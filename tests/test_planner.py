import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from planner import Planner, PlannerFailure, parse_decision


class Reply:
    def __init__(self, content):
        self.text = json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                               'usage': {'prompt_tokens': 12, 'completion_tokens': 5}})
    def raise_for_status(self):
        pass
    def json(self):
        return json.loads(self.text)


class Client:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
    def post(self, url, json):
        self.requests.append(json)
        return Reply(self.replies.pop(0))


class Tests(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(parse_decision('{"action":"execute","subgoal":"lift pen"}')['subgoal'], 'lift pen')
        for bad in ['[]', '{}', '{"action":"wait","subgoal":"x"}',
                    '{"action":"execute","subgoal":" "}', '{"action":"wait","action":"finish"}',
                    '```json\n{"action":"finish"}\n```', '{"action":"execute","subgoal":"x"']:
            with self.assertRaises((ValueError, TypeError)):
                parse_decision(bad)

    def test_retry_and_privilege_boundary(self):
        with tempfile.TemporaryDirectory(dir='/root/gpufree-data/.tmp') as directory:
            cfg = dict(timeout=1, history_length=3, invalid_retries=1, model='test', temperature=0,
                       max_tokens=100, reasoning=True)
            p = Planner(cfg, directory, 1)
            p.client.close()
            p.client = Client(['bad JSON', '{"action":"wait"}'])
            obs = {'vision': {name: {'color': np.zeros((10, 10, 3), dtype=np.uint8)}
                for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
                'state': {'secret_pose': [99]}, 'reward': 'PRIVILEGED_SENTINEL', 'success': True}
            self.assertEqual(p.decide('put pens in holder', obs, 'now', [])['action'], 'wait')
            self.assertEqual(p.calls, 2)
            request = json.dumps(p.client.requests)
            self.assertNotIn('PRIVILEGED_SENTINEL', request)
            self.assertNotIn('secret_pose', request)
            self.assertTrue(p.client.requests[0]['chat_template_kwargs']['enable_thinking'])
            records = [json.loads(x) for x in Path(directory, 'planner.jsonl').read_text().splitlines()]
            self.assertTrue(records[0]['parse_failure'])
            self.assertIsNone(records[1]['reasoning_tokens'])

    def test_never_execute_partial_output(self):
        with tempfile.TemporaryDirectory(dir='/root/gpufree-data/.tmp') as directory:
            cfg = dict(timeout=1, history_length=3, invalid_retries=1, model='test', temperature=0,
                       max_tokens=100, reasoning=True)
            p = Planner(cfg, directory, 1)
            p.client.close()
            p.client = Client(['{"action":"execute","subgoal":"pick', 'garbage'])
            obs = {'vision': {name: {'color': np.zeros((10, 10, 3), dtype=np.uint8)}
                for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}}
            with self.assertRaises(PlannerFailure):
                p.decide('test', obs, 'now', [])
            self.assertEqual(p.calls, 2)


if __name__ == '__main__':
    unittest.main()
