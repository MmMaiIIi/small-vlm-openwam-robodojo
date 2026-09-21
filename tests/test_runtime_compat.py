import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from runtime_compat import NumpyEvaluatorLayoutView, install_evaluator_boundary


class Layout:
    def __init__(self):
        self.position = torch.tensor([1., 2., 3.], dtype=torch.float32)
        self.rotation = torch.tensor([1., 0., 0., 0.], dtype=torch.float32)
        self.marker = object()
    def get_instance_pose(self, **kwargs):
        return self.position, self.rotation
    def untouched(self):
        return self.marker


class Tests(unittest.TestCase):
    def test_deferred_scene_initialization(self):
        class Parser:
            def initialize(self, environment):
                self.layout_manager = environment.layout
                return 'unchanged return'
        with tempfile.TemporaryDirectory(dir='/root/gpufree-data/.tmp') as directory:
            parser = Parser()
            environment = SimpleNamespace(layout=Layout(), reward_manager=SimpleNamespace(func_parser=parser))
            install_evaluator_boundary(environment, directory)
            self.assertFalse(hasattr(parser, 'layout_manager'))
            self.assertEqual(parser.initialize(environment), 'unchanged return')
            self.assertIsInstance(parser.layout_manager, NumpyEvaluatorLayoutView)
            self.assertIsInstance(parser.layout_manager.get_instance_pose()[0], np.ndarray)

    def test_values_dtype_and_scope(self):
        with tempfile.TemporaryDirectory(dir='/root/gpufree-data/.tmp') as directory:
            native = Layout()
            view = NumpyEvaluatorLayoutView(native, directory)
            position, rotation = view.get_instance_pose(env_idx=0)
            self.assertEqual(position.dtype, np.float32)
            np.testing.assert_array_equal(position, native.position.numpy())
            np.testing.assert_array_equal(rotation, native.rotation.numpy())
            self.assertIsInstance(native.get_instance_pose()[0], torch.Tensor)
            self.assertIs(view.untouched(), native.marker)
            self.assertIs(view.marker, native.marker)


if __name__ == '__main__':
    unittest.main()
