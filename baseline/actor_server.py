"""Official XPolicyLab model/server with run-local timing and generation tags."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from common import append, event, timestamp
from XPolicyLab.policy.OpenWAM.model import Model
from client_server.ws.model_server import PolicyServer, PolicyServerConfig


class RecordedModel(Model):
    def __init__(self, cfg, run):
        self.run, self.generation, self.inference_id = Path(run), 0, 0
        self.current_subgoal, self.pending = None, None
        torch.manual_seed(cfg['environment']['seed'])
        super().__init__({'env_cfg_type': cfg['environment']['env_cfg_type'], 'action_type': 'ee',
                         'ckpt_dir': cfg['actor']['checkpoint'], 'device': 'cuda',
                         'replan_steps': cfg['actor']['replan_steps'], 'allow_dummy_policy': False})
        self.predicted_length = cfg['actor']['predicted_chunk_length']
        generate = self._engine.generate_batch
        def measured_generate(*args, **kwargs):
            result = generate(*args, **kwargs)
            self.predicted_length = int(result['actions'].shape[1])
            return result
        self._engine.generate_batch = measured_generate
        event(self.run, 'actor_loaded', allocated_bytes=torch.cuda.memory_allocated())

    def begin_subgoal(self, payload):
        if self.pending is not None:
            raise RuntimeError('Previous action chunk must be accounted before a new subgoal')
        self.reset()
        self.generation = int(payload['generation'])
        self.current_subgoal = payload['subgoal']
        event(self.run, 'actor_instruction_reset', generation=self.generation, subgoal=self.current_subgoal)
        return {'generation': self.generation, 'subgoal': self.current_subgoal, 'cache_empty': not self._batch}

    def update_obs(self, obs):
        if self.current_subgoal is not None and obs.get('instruction') != self.current_subgoal:
            raise ValueError('Instruction/generation mismatch')
        return super().update_obs(obs)

    def get_action(self):
        if self.pending is not None:
            raise RuntimeError('Unaccounted previous chunk')
        self.inference_id += 1
        started, stamp = time.monotonic(), timestamp()
        torch.cuda.reset_peak_memory_stats()
        try:
            actions = super().get_action()
            torch.cuda.synchronize()
            assert actions and all(np.isfinite(v).all() for a in actions for v in a.values())
            self.pending = {'inference_id': self.inference_id, 'generation': self.generation,
                'subgoal': self.current_subgoal, 'start_timestamp': stamp, 'end_timestamp': timestamp(),
                'latency': time.monotonic() - started, 'predicted_action_count': self.predicted_length,
                'returned_action_count': len(actions), 'executed_action_count': None,
                'stale_actions_dropped': None, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            event(self.run, 'actor_inference', **self.pending)
            return actions
        except Exception as exc:
            append(self.run / 'actor.jsonl', dict(inference_id=self.inference_id, generation=self.generation,
                subgoal=self.current_subgoal, start_timestamp=stamp, end_timestamp=timestamp(),
                latency=time.monotonic()-started, predicted_action_count=None, executed_action_count=0,
                stale_actions_dropped=0, error=repr(exc)))
            raise

    def account_chunk(self, payload):
        if self.pending is None or payload['generation'] != self.generation:
            raise RuntimeError('Stale or missing chunk accounting')
        count = int(payload['executed_action_count'])
        assert 0 <= count <= self.pending['returned_action_count']
        self.pending.update(executed_action_count=count,
            stale_actions_dropped=self.pending['returned_action_count'] - count,
            replan_tail_dropped=self.pending['predicted_action_count'] - self.pending['returned_action_count'])
        append(self.run / 'actor.jsonl', self.pending)
        result, self.pending = self.pending, None
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load((Path(args.run) / 'config.yaml').read_text())
    model = RecordedModel(cfg, args.run)
    server = PolicyServer(model, PolicyServerConfig(host='127.0.0.1', port=args.port))
    async def serve():
        await server.start()
        event(args.run, 'actor_socket_ready', port=args.port)
        await server.serve_forever()
    asyncio.run(serve())
