import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
import torch
from generation_telemetry import GenerationTelemetry, native_context, remaining_context
from rpent_launch import RobotConnection


class UnrestrictedTests(unittest.TestCase):
    def test_native_context_from_checkpoint(self):
        from transformers import AutoConfig, AutoTokenizer
        model = '/root/gpufree-data/models/Qwen/Qwen3.5-2B'
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        limit = native_context(config, tokenizer)
        self.assertEqual(limit, config.text_config.max_position_embeddings)
        self.assertEqual(limit, tokenizer.model_max_length)
        self.assertEqual(remaining_context(limit, 952), 261191)
        self.assertEqual(remaining_context(limit, limit), 0)
        with self.assertRaises(ValueError): remaining_context(limit, 1, 32)

    def test_incremental_streamer_does_not_stop(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('/root/gpufree-data/models/Qwen/Qwen3.5-2B', local_files_only=True)
        import time
        with tempfile.TemporaryDirectory() as tmp, patch('generation_telemetry.subprocess.check_output', return_value='123, 25'):
            stream = GenerationTelemetry(tok, tmp, 1, 3, 262144, time.monotonic(), progress_tokens=2)
            stream.put(torch.tensor([[1, 2, 3]]))
            ids = tok.encode('Reasoning.\n</think>\n<tool_call>\n<function=wait></function>\n</tool_call>', add_special_tokens=False)
            for token in ids:
                self.assertIsNone(stream.put(torch.tensor([token])))
            stream.end()
            rows = [json.loads(x) for x in Path(tmp, 'planner_progress.jsonl').read_text().splitlines()]
            self.assertGreater(len(rows), 1)
            self.assertEqual(rows[-1]['generated_tokens'], len(ids))
            self.assertIsNotNone(stream.times['time_to_think_end'])
            self.assertIsNotNone(stream.times['time_to_first_tool_call'])
            self.assertGreater(stream.observed_reasoning_tokens, 0)
            persisted = [json.loads(x) for x in stream.token_path.read_text().splitlines()]
            self.assertEqual([t for row in persisted for t in row['token_ids']], ids)
            self.assertIn('</think>', stream.raw_path.read_text())

    def test_lifecycle_is_not_policy_feedback(self):
        class Conn:
            def send(self, x): pass
            def recv(self): return {'feedback': {'executed_env_steps': 12}, 'lifecycle_ended': True}
        robot = RobotConnection(Conn())
        self.assertEqual(robot.call('wait', {}), {'executed_env_steps': 12})
        with self.assertRaisesRegex(RuntimeError, 'OFFICIAL_EPISODE_ENDED'): robot.before_planner()


if __name__ == '__main__': unittest.main()
