import io
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from PIL import Image
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelRequest, UserPromptPart, ToolReturnPart
from rpent.planner.api_loop import ApiAgentLoop
from rpent.dashboard.events import NullDashboardEventSink
from local_qwen_planner import LocalQwenPlanner, parse_native_tool, render_messages
from rpent_tools import RobotAgentLoop, RobotToolkit


class AdapterTests(unittest.TestCase):
    def definitions(self):
        return [{'function': {'name': 'wait', 'parameters': {
            'type': 'object', 'properties': {}, 'additionalProperties': False}}}]

    def test_native_call(self):
        self.assertEqual(parse_native_tool('thinking</think><tool_call><function=wait></function></tool_call>',
            self.definitions())[1:], ('wait', {}))

    def test_no_partial_or_multiple_execution(self):
        for text in ('<tool_call><function=wait></function></tool_call>',
                     '</think><tool_call><function=wait></function></tool_call>suffix',
                     '</think><tool_call><function=unknown></function></tool_call>',
                     '</think><tool_call><function=wait>',
                     '</think>' + '<tool_call><function=wait></function></tool_call>'*2):
            with self.assertRaises(ValueError): parse_native_tool(text, self.definitions())

    def test_native_loop_unchanged(self):
        self.assertIs(RobotAgentLoop.solve, ApiAgentLoop.solve)
        self.assertIs(RobotAgentLoop._solve, ApiAgentLoop._solve)
        self.assertNotIn('execute_tool', RobotToolkit.__dict__)

    def test_rpent_turn_counter_excludes_standalone_calls(self):
        from pydantic_ai.models import ModelRequestParameters
        adapter = LocalQwenPlanner.__new__(LocalQwenPlanner)
        adapter.rpent_turns = 0
        adapter.calls = 2
        adapter.generate = lambda messages, tools: adapter.rpent_turns
        messages = [ModelRequest(parts=[UserPromptPart('test')])]
        parameters = ModelRequestParameters(function_tools=[])
        self.assertEqual(asyncio.run(adapter.request(messages, {}, parameters)), 1)
        self.assertEqual(asyncio.run(adapter.request(messages, {}, parameters)), 2)

    def test_images_and_tool_results(self):
        buf = io.BytesIO(); Image.new('RGB', (8,8)).save(buf, format='PNG')
        messages = [ModelRequest(parts=[UserPromptPart(['test', BinaryContent(buf.getvalue(), media_type='image/png')]),
            ToolReturnPart('wait', {'executed_env_steps':64}, 'a')])]
        out = render_messages(messages, None)
        self.assertIsInstance(out[0]['content'][1]['image'], Image.Image)
        self.assertEqual(out[1]['role'], 'tool')

    def test_exact_three_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            kit = RobotToolkit(None, tmp, NullDashboardEventSink())
            self.assertEqual({x['name'] for x in kit.get_tools_spec()}, {'execute_subgoal','wait','finish'})
            self.assertIsNone(kit.memory)


if __name__ == '__main__': unittest.main()
