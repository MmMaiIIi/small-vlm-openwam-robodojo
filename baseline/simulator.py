"""One bounded episode through unmodified official environment and actor APIs."""
import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path
from types import MethodType

import yaml
from common import event, timestamp

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--actor-port', type=int, required=True)
parser.add_argument('--qwen-port', type=int, required=True)
args = parser.parse_args()
run = Path(args.run)
cfg = yaml.safe_load((run / 'config.yaml').read_text())
root = Path(cfg['paths']['robodojo'])
task, seed = cfg['environment']['task'], cfg['environment']['seed']
app = env = writer = planner = None
started = time.monotonic()
episode_start = None
result = dict(task=task, seed=seed, completed=False, success=False, env_steps=0,
              failure_reason=None, failure_category=None, clean_exit=False)

try:
    from isaaclab.app import AppLauncher
    from env.camera_manager.capture.render_sync import zero_delay_kit_args
    app = AppLauncher(headless=True, enable_cameras=True,
        kit_args=' --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera '
                 + zero_delay_kit_args() + ' --/log/file=' + str(run / 'kit.log')).app
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from utils.pipeline_utils import process_config, process_randomization
    from utils.save_file import VideoStreamWriter
    from src.eval_client.eval_env import create_eval_env
    from planner import Planner

    def load(relative):
        return OmegaConf.load(root / relative)
    base = load('env_cfg/arx_x5.yml')
    base.update(task_name=task, seed=seed, num_envs=1, eval_num=1, policy_name='OpenWAM')
    native = OmegaConf.create({
        'sim': load('env_cfg/sim/' + base.config.sim + '.yml'),
        'scene': load('env_cfg/scene/' + base.config.scene + '.yml'),
        'robot': load('env_cfg/robot/' + base.config.robot + '.yml'),
        'camera': load('env_cfg/camera/' + base.config.camera + '.yml'),
        'task_env': load('task/RoboDojo/config/' + task + '.yml'), 'eval_cfg': base,
        'deploy_cfg': {'policy_name': 'OpenWAM', 'port': args.actor_port, 'host': '127.0.0.1'}})
    native.sim.scene.num_envs = 1
    native.sim.seed = [seed]
    native.sim.device = 'cuda:0'
    native, _ = process_config(process_randomization(native), task)
    native.camera.default_frequency = base.observation.collect_freq
    assert base.observation.collect_freq == cfg['environment']['control_frequency']
    env = create_eval_env(native, app, resume_state={'save_dir': str(run)})
    from runtime_compat import install_evaluator_boundary
    install_evaluator_boundary(env, run)
    env.model_client._client.config.request_timeout_s = cfg['actor']['timeout']
    last_video_step = -1

    def stream_video(self, env_idx, frame):
        global writer, last_video_step
        step = self.take_action_cnt[0]
        if step == last_video_step:
            return
        rgb = np.concatenate([frame['vision'][name]['color'][:, :, :3]
            for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')], axis=1)
        if writer is None:
            writer = VideoStreamWriter(str(run / 'video.mp4'), *rgb.shape,
                                       fps=cfg['logging']['video_fps'], is_rgb=True)
        writer.append(rgb)
        last_video_step = step
    # Run-local recording hook only: no environment state/control changes.
    env._stream_vision = MethodType(stream_video, env)
    env.reset(seed=[seed])
    # Identical initialization to the official run_eval(), without its policy loop.
    env.run_reward()
    if hasattr(env, 'get_score'):
        env.get_score()
    if getattr(env, 'interact', False) and hasattr(env, 'query_support_arm_traj'):
        env.query_support_arm_traj(env_idx=0)
    obs = env.get_obs()
    observed_at = timestamp()
    instruction = obs['instruction']
    result['task_instruction'] = instruction
    result['official_step_limit'] = env.step_lim
    planner = Planner(cfg['planner'], run, args.qwen_port)
    history = []
    episode_start = time.monotonic()
    event(run, 'environment_ready', task=task, instruction=instruction,
          cameras={k: list(v['color'].shape) for k, v in obs['vision'].items()},
          control_frequency=base.observation.collect_freq, official_step_limit=env.step_lim,
          torch_allocated_bytes=torch.cuda.memory_allocated())
    resident_mib = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], text=True).strip())
    event(run, 'coexistence_before_episode', total_memory_mib=resident_mib,
          services=['Qwen reasoning server', 'OpenWAM CUDA model', 'RoboDojo reset scene'])
    generation = 0
    reason = 'harness_budget'
    while (env.take_action_cnt[0] < cfg['harness']['max_episode_steps']
           and planner.calls < cfg['harness']['max_planner_calls'] and not env.is_episode_end()):
        # Retry is also an actual API call and must fit the total call budget.
        planner.cfg['invalid_retries'] = min(cfg['planner']['invalid_retries'],
            cfg['harness']['max_planner_calls'] - planner.calls - 1)
        decision = planner.decide(instruction, obs, observed_at, history)
        generation += 1
        event(run, 'decision', generation=generation, env_step=env.take_action_cnt[0], **decision)
        history.append(decision)
        if decision['action'] == 'finish':
            reason = 'planner_finish'
            break
        window = min(cfg['harness']['high_level_window'],
                     cfg['harness']['max_episode_steps'] - env.take_action_cnt[0])
        if decision['action'] == 'execute':
            reply = env.model_client.call(func_name='begin_subgoal', obs={
                'generation': generation, 'subgoal': decision['subgoal']})
            assert reply['generation'] == generation and reply['cache_empty']
            assert reply['subgoal'] == decision['subgoal']
        done = 0
        while done < window and not env.is_episode_end():
            if decision['action'] == 'execute':
                actor_obs = dict(obs, instruction=decision['subgoal'])
                env.model_client.call(func_name='update_obs', obs=actor_obs)
                actions = env.model_client.call(func_name='get_action')
                if not isinstance(actions, list) or not actions:
                    raise RuntimeError('Empty or invalid actor action chunk')
            else:
                # WAIT holds the observed absolute EE targets using the native controller.
                actions = [{key: np.asarray(obs['state'][key]).copy() for key in
                    ('left_ee_pose', 'left_ee_joint_state', 'right_ee_pose', 'right_ee_joint_state')}]
            executed = 0
            try:
                for action in actions[:window - done]:
                    env.take_action(action)
                    executed += 1
                    done += 1
                    obs = env.get_obs()
                    observed_at = timestamp()
                    if not all(np.isfinite(np.asarray(value)).all() for value in obs['state'].values()):
                        raise RuntimeError('Nonfinite robot observation')
                    if env.is_episode_end():
                        break
            finally:
                if decision['action'] == 'execute':
                    account = env.model_client.call(func_name='account_chunk', obs={
                        'generation': generation, 'executed_action_count': executed})
                    assert account['generation'] == generation
                    assert account['executed_action_count'] == executed
                # Drop the entire client-owned chunk before returning to the planner.
                actions = []
        event(run, 'observation_returned', generation=generation, env_step=env.take_action_cnt[0],
              observation_timestamp=observed_at, client_chunk_empty=True)
    if env.end_flag[0]:
        reason = 'environment_success' if env.success[0] else 'environment_failure'
    # Evaluator-only label: never fed into the planner context or actor observation.
    reward = env.reward_manager.get_reward(final_check=True)[0]
    result.update(completed=True, success=bool(reward > 1 - 1e-3), termination=reason,
                  env_steps=env.take_action_cnt[0], planner_calls=planner.calls,
                  high_level_decisions=len(history), failure_category=None if reward > 1-1e-3 else 'UNKNOWN',
                  failure_reason=None if reward > 1-1e-3 else reason)
except Exception as exc:
    traceback.print_exc()
    text = repr(exc)
    result.update(failure_reason=text,
        failure_category='TIMING' if 'timeout' in text.lower() else
                         'HIGH_LEVEL_REASONING' if 'planner_failure' in text else
                         'PHYSICAL_EXECUTION' if 'Nonfinite robot' in text else 'INTERFACE')
    if env is not None:
        result['env_steps'] = env.take_action_cnt[0]
    event(run, 'episode_error', error=text, traceback=traceback.format_exc())
finally:
    result['wall_clock_time'] = time.monotonic() - (episode_start or started)
    result['sim_time'] = result['env_steps'] / cfg['environment']['control_frequency']
    try:
        if writer is not None:
            writer.close()
        if planner is not None:
            planner.client.close()
        if env is not None:
            env.model_client.close()
            env.close()
        result['clean_exit'] = True
    except Exception as exc:
        result['close_error'] = repr(exc)
        traceback.print_exc()
    (run / 'summary.json').write_text(json.dumps(result, indent=2, default=str))
    event(run, 'episode_end', **result)
    print('BASELINE_EPISODE_RESULT', json.dumps(result), flush=True)
    if app is not None:
        app.close(wait_for_replicator=False)
