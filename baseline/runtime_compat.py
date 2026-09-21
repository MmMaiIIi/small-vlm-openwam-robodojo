"""Run-local NumPy evaluator boundary; does not modify the simulator's layout API."""
import torch
from common import event


class NumpyEvaluatorLayoutView:
    """Delegate everything except the pose type consumed by NumPy reward functions.

    Func_Parser has 53 pose calls used by NumPy / transforms3d / Shapely functions.
    The underlying layout manager must retain its native Torch return contract for
    all other consumers. Only this one evaluator instance receives this view.
    """
    def __init__(self, layout_manager, run):
        self._layout_manager, self._run, self._reported = layout_manager, run, False

    def __getattr__(self, name):
        return getattr(self._layout_manager, name)

    def get_instance_pose(self, *args, **kwargs):
        values = self._layout_manager.get_instance_pose(*args, **kwargs)
        converted = tuple(v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values)
        if not self._reported:
            def describe(v):
                return dict(type=type(v).__name__, dtype=str(getattr(v, 'dtype', None)),
                            device=str(getattr(v, 'device', 'cpu')), shape=list(v.shape) if hasattr(v, 'shape') else None)
            event(self._run, 'evaluator_numpy_boundary', source=[describe(v) for v in values],
                  target=[describe(v) for v in converted], consumer='Func_Parser NumPy evaluator')
            self._reported = True
        return converted


def install_evaluator_boundary(env, run):
    parser = env.reward_manager.func_parser
    initialize = parser.initialize
    def initialize_with_numpy_view(environment):
        result = initialize(environment)
        parser.layout_manager = NumpyEvaluatorLayoutView(parser.layout_manager, run)
        return result
    # Native scene setup initializes this parser during reset, not construction.
    parser.initialize = initialize_with_numpy_view
    if hasattr(parser, 'layout_manager'):
        parser.layout_manager = NumpyEvaluatorLayoutView(parser.layout_manager, run)
