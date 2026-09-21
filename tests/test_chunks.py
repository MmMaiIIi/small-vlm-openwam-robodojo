import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, '/root/gpufree-data/RoboDojo')
sys.path.insert(0, '/root/gpufree-data/RoboDojo/XPolicyLab')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from actor_server import RecordedModel


class Tests(unittest.TestCase):
    def test_chunk_discard_and_reset_contract_without_loading_weights(self):
        with tempfile.TemporaryDirectory(dir='/root/gpufree-data/.tmp') as directory:
            model = RecordedModel.__new__(RecordedModel)
            model.run = Path(directory)
            model.generation = 1
            model.current_subgoal = 'old instruction'
            model._batch, model._order = {0: 'old observation'}, [0]
            model.pending = dict(generation=1, predicted_action_count=32, returned_action_count=32)
            with self.assertRaises(RuntimeError):
                model.begin_subgoal(dict(generation=2, subgoal='new instruction'))
            with self.assertRaises(RuntimeError):
                model.account_chunk(dict(generation=2, executed_action_count=10))
            record = model.account_chunk(dict(generation=1, executed_action_count=10))
            self.assertEqual(record['stale_actions_dropped'], 22)
            self.assertIsNone(model.pending)
            ack = model.begin_subgoal(dict(generation=2, subgoal='new instruction'))
            self.assertTrue(ack['cache_empty'])
            self.assertEqual(model._order, [])
            self.assertEqual(model.current_subgoal, 'new instruction')
            with self.assertRaises(ValueError):
                model.update_obs({'instruction': 'old instruction'})


if __name__ == '__main__':
    unittest.main()
