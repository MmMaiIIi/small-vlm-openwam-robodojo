"""Three tools and an agent factory; RPent owns all iteration and history."""
import io
import time
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory, Thinking
from rpent.planner.api_loop import ApiAgentLoop, _build_tools, _prune_history_images
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly
from common import append, timestamp

RULES = """You control a robot through a low-level language-conditioned actor.
Use the original task and current camera views to choose only the next short subgoal.
The panels are head, left wrist, right wrist, in that order. Images are the current scene.
Use your recent decisions only as attempted actions, never as proof of completion.
execute_subgoal provides one concise textual subgoal, which may include holding an object
with one hand while manipulating with the other. It runs for a fixed short window.
wait lets physics advance while holding the current robot target. finish ends the episode
only when the visible task appears complete. Inspect each new observation to decide.
Do not output a full plan, joint values, EE deltas, distances, coordinates, or trajectories.
After reasoning, call exactly one of the provided tools.
"""


class RobotAgentLoop(ApiAgentLoop):
    def _build_agent(self, system_prompt, toolkit):
        # RPent adds a filesystem read_image tool by default. Only expose the
        # three registered robot tools; solve/_solve and image pruning unchanged.
        names = {t['name'] for t in toolkit.get_tools_spec()}
        return Agent(self._model, instructions=system_prompt,
            tools=[t for t in _build_tools(toolkit) if t.name in names],
            model_settings={},
            capabilities=[Thinking(effort='high'), ProcessHistory(processor=_prune_history_images)])


class RobotToolkit(Toolkit):
    def _register_common_tools(self):
        # No filesystem tools, memory store, or task skills.
        pass

    def __init__(self, driver, run, sink):
        self.driver, self.run = driver, Path(run)
        from rpent.utils.logging import init_output_dir
        init_output_dir(self.run / 'rpent')
        super().__init__(dashboard_events=sink, state=EnvState(self.run / 'rpent-state'), memory=None)
        self.add_tool('execute_subgoal', {'name': 'execute_subgoal',
            'description': 'Execute one short natural-language subgoal for a fixed finite window, then observe.',
            'input_schema': {'type': 'object', 'properties': {'instruction':
                {'type': 'string', 'minLength': 1, 'maxLength': 240}},
                'required': ['instruction'], 'additionalProperties': False}}, self.execute_subgoal)
        for name, handler, description in [('wait', self.wait, 'Hold current targets for one window, then observe.'),
                                          ('finish', self.finish, 'End the episode when the visible task appears complete.')]:
            self.add_tool(name, {'name': name, 'description': description,
                'input_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}}, handler)

    def _invoke(self, name, arguments):
        start, stamp = time.monotonic(), timestamp()
        result = self.driver.call(name, arguments)
        append(self.run / 'tools.jsonl', dict(tool=name, arguments=arguments,
            start_timestamp=stamp, end_timestamp=timestamp(), execution_latency=time.monotonic()-start,
            **{k:v for k,v in result.items() if k != '_image_bytes'}))
        return result

    def execute_subgoal(self, instruction):
        return self._invoke('execute_subgoal', {'instruction': instruction})

    def wait(self):
        return self._invoke('wait', {})

    @readonly
    def finish(self):
        return self._invoke('finish', {})

    def get_env_state(self, *, command=None, result=None, elapsed_s=None):
        result = dict(result or {})
        metadata = {k:v for k,v in result.items() if k != '_image_bytes'}
        with self.state.record_step(state=metadata, command=command, result=metadata, elapsed_s=elapsed_s):
            if result.get('_image_bytes'):
                import numpy as np
                from PIL import Image
                self.state.save('camera.png', np.asarray(Image.open(io.BytesIO(result['_image_bytes']))))
        return result
