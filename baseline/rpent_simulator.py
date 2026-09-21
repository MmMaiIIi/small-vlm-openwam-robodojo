"""Native Isaac worker: receives bounded tool operations, never plans."""
import argparse
import io
import json
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from types import MethodType

import yaml
from common import event, timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--actor-port', type=int, required=True)
    parser.add_argument('--fd', type=int, required=True)
    args = parser.parse_args()
    run = Path(args.run); cfg = yaml.safe_load((run / 'config.yaml').read_text())
    conn = Connection(args.fd)
    root = Path(cfg['paths']['robodojo'])
    task, seed = cfg['environment']['task'], cfg['environment']['seed']
    app = env = writer = None
    started = time.monotonic()
    result = dict(task=task, seed=seed, success=False, env_steps=0, clean_exit=False)
    try:
        from isaaclab.app import AppLauncher
        from env.camera_manager.capture.render_sync import zero_delay_kit_args
        app = AppLauncher(headless=True, enable_cameras=True,
            kit_args=' --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera '
            + zero_delay_kit_args() + ' --/log/file=' + str(run / 'kit.log')).app
        import numpy as np
        from PIL import Image, ImageDraw
        from omegaconf import OmegaConf
        from utils.pipeline_utils import process_config, process_randomization
        from utils.save_file import VideoStreamWriter
        from src.eval_client.eval_env import create_eval_env
        from runtime_compat import install_evaluator_boundary
        def load(relative):
            return OmegaConf.load(root / relative)
        base = load('env_cfg/arx_x5.yml')
        base.update(task_name=task, seed=seed, num_envs=1, eval_num=1, policy_name='OpenWAM')
        native = OmegaConf.create({name: load('env_cfg/' + name + '/' + base.config[name] + '.yml')
                                  for name in ('sim', 'scene', 'robot', 'camera')})
        native.update(task_env=load('task/RoboDojo/config/' + task + '.yml'), eval_cfg=base,
            deploy_cfg={'policy_name': 'OpenWAM', 'port': args.actor_port, 'host': '127.0.0.1'})
        native.sim.scene.num_envs = 1; native.sim.seed = [seed]; native.sim.device = 'cuda:0'
        native, _ = process_config(process_randomization(native), task)
        native.camera.default_frequency = base.observation.collect_freq
        assert base.observation.collect_freq == cfg['environment']['control_frequency']
        env = create_eval_env(native, app, resume_state={'save_dir': str(run)})
        install_evaluator_boundary(env, run)
        env.model_client._client.config.request_timeout_s = cfg['actor']['timeout']
        last_video_step = -1
        cameras = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
        def stream_video(self, env_idx, frame):
            nonlocal writer, last_video_step
            step = self.take_action_cnt[0]
            if step == last_video_step:
                return
            rgb = np.concatenate([frame['vision'][n]['color'][:, :, :3] for n in cameras], axis=1)
            if writer is None:
                writer = VideoStreamWriter(str(run / 'video.mp4'), *rgb.shape,
                                           fps=cfg['logging']['video_fps'], is_rgb=True)
            writer.append(rgb); last_video_step = step
        env._stream_vision = MethodType(stream_video, env)
        env.reset(seed=[seed]); env.run_reward()
        if hasattr(env, 'get_score'):
            env.get_score()
        if getattr(env, 'interact', False) and hasattr(env, 'query_support_arm_traj'):
            env.query_support_arm_traj(env_idx=0)
        obs = env.get_obs()
        def public_observation():
            panel = Image.new('RGB', (1152, 312))
            draw = ImageDraw.Draw(panel)
            for i, name in enumerate(cameras):
                panel.paste(Image.fromarray(obs['vision'][name]['color'][:, :, :3]).resize((384, 288)), (384*i,24))
                draw.text((384*i+5,5), name, fill='white')
            buf = io.BytesIO(); panel.save(buf, format='PNG')
            return {'_image_bytes': buf.getvalue(), 'observation_timestamp': timestamp(),
                    'env_steps': int(env.take_action_cnt[0])}
        conn.send({'instruction': obs['instruction'], **public_observation()})
        event(run, 'environment_ready', task=task, instruction=obs['instruction'])
        generation = 0
        while True:
            request = conn.recv()
            name, arguments = request['tool'], request['arguments']
            if name == 'close':
                break
            if name == 'finish':
                conn.send({'_finish': True}); continue
            if name not in ('execute_subgoal', 'wait'):
                raise ValueError('Invalid robot operation')
            generation += 1
            window = min(cfg['tool']['high_level_window'],
                         cfg['environment']['max_steps'] - env.take_action_cnt[0])
            instruction = arguments.get('instruction')
            if name == 'execute_subgoal':
                ack = env.model_client.call(func_name='begin_subgoal', obs={
                    'generation': generation, 'subgoal': instruction})
                assert ack['generation'] == generation and ack['cache_empty'] and ack['subgoal'] == instruction
            executed_total = stale = inferences = 0
            inference_time = 0.0
            while executed_total < window and not env.is_episode_end():
                if name == 'execute_subgoal':
                    env.model_client.call(func_name='update_obs', obs=dict(obs, instruction=instruction))
                    tic = time.monotonic()
                    actions = env.model_client.call(func_name='get_action')
                    inference_time += time.monotonic()-tic; inferences += 1
                    assert isinstance(actions, list) and actions
                else:
                    actions = [{k: np.asarray(obs['state'][k]).copy() for k in
                        ('left_ee_pose', 'left_ee_joint_state', 'right_ee_pose', 'right_ee_joint_state')}]
                executed = 0
                try:
                    for action in actions[:window-executed_total]:
                        env.take_action(action); executed += 1; executed_total += 1
                        obs = env.get_obs()
                        if not all(np.isfinite(np.asarray(v)).all() for v in obs['state'].values()):
                            raise RuntimeError('Nonfinite robot observation')
                        if env.is_episode_end():
                            break
                finally:
                    if name == 'execute_subgoal':
                        ack = env.model_client.call(func_name='account_chunk', obs={
                            'generation': generation, 'executed_action_count': executed})
                        assert ack['generation'] == generation and ack['executed_action_count'] == executed
                        stale += len(actions)-executed
                    actions = []
            # Explicit allow-list: evaluator success, reward, poses and end flags never leave this worker.
            feedback = dict(public_observation(), executed_env_steps=executed_total,
                OpenWAM_inference_count=inferences, OpenWAM_inference_latency=inference_time,
                stale_actions_dropped=stale)
            conn.send(feedback)
            event(run, 'observation_returned', generation=generation, env_step=env.take_action_cnt[0],
                  client_chunk_empty=True, subgoal=instruction)
        reward = env.reward_manager.get_reward(final_check=True)[0]
        result.update(success=bool(reward > 1-1e-3), env_steps=int(env.take_action_cnt[0]),
            official_ended=bool(env.is_episode_end()))
    except Exception as exc:
        result['error'] = repr(exc); traceback.print_exc()
        event(run, 'simulator_error', traceback=traceback.format_exc())
    finally:
        if env is not None:
            result['env_steps'] = int(env.take_action_cnt[0])
        try:
            if writer is not None: writer.close()
            if env is not None: env.model_client.close(); env.close()
            result['clean_exit'] = True
        except Exception as exc:
            result['close_error'] = repr(exc)
        result.update(wall_time=time.monotonic()-started,
                      sim_time=result['env_steps']/cfg['environment']['control_frequency'])
        (run / 'simulator.json').write_text(json.dumps(result, indent=2))
        conn.close()
        # Isaac's configured fast shutdown may exit the process here.
        if app is not None: app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
