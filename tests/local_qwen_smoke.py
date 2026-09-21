"""Real local model tests and real RPent loop with synthetic robot only."""
import io
import json
import subprocess
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
import yaml
from PIL import Image, ImageDraw
from pydantic_ai import BinaryContent
from rpent.dashboard.events import NullDashboardEventSink
from local_qwen_planner import LocalQwenPlanner
from rpent_tools import RobotToolkit, RobotAgentLoop


def listeners():
    # Normalize whitespace; ss alignment changes when new listeners appear.
    return sorted({' '.join(row.split()[3:5]) for row in subprocess.check_output(
        ['ss', '-H', '-ltn'], text=True).splitlines()})


class MockRobot:
    def __init__(self):
        self.calls = 0

    def image(self):
        image = Image.new('RGB', (384, 288), 'white')
        ImageDraw.Draw(image).rectangle((100, 80, 200, 180), fill='red')
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue()

    def call(self, name, args):
        self.calls += 1
        return {'_finish': True} if name == 'finish' else {
            'mock': True, 'completed_test_operations': self.calls,
            'test_operations_requested': 3, '_image_bytes': self.image(), 'executed_env_steps': 0}


if __name__ == '__main__':
    def forbid_network(event, args):
        if event in ('socket.connect', 'socket.bind') and args[0].family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('Local Qwen test forbids TCP/HTTP/API connections and listeners')
    sys.addaudithook(forbid_network)
    root = Path(__file__).resolve().parents[1]
    run = Path(sys.argv[1]); run.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((root / 'configs/rpent_qwen2b_openwam.yaml').read_text())
    before = listeners()
    model = LocalQwenPlanner(cfg['planner'], run)
    robot = MockRobot()
    model.generate([{'role': 'user', 'content': 'What is 2 + 2?'}])
    model.generate([{'role': 'user', 'content': [
        {'type': 'image', 'image': Image.open(io.BytesIO(robot.image()))},
        {'type': 'text', 'text': 'Describe the shape and color in this image.'}]}])
    sink = NullDashboardEventSink()
    toolkit = RobotToolkit(robot, run, sink)
    loop = RobotAgentLoop(model, max_tokens=cfg['planner']['max_generation_tokens'],
        reasoning_effort='high', dashboard_events=sink)
    # Interface fixture only, not a robotic task plan or replayed model answer.
    result = loop.solve(system_prompt='This is a mock tool-interface test, not a robotics task. '
        'Request three observation checks using execute_subgoal, one per turn, then finish. '
        'Inspect the returned image and operation counter after each check. Call only one tool per turn.',
        user_message=['Check the red shape in this test image.', BinaryContent(robot.image(), media_type='image/png')],
        toolkit=toolkit, max_turns=4)
    after = listeners()
    record = {'stats': result.stats, 'error': result.error, 'finish': result.finish_result,
              'load_count': model.load_count, 'generations': model.calls,
              'before_listeners': before, 'after_listeners': after,
              'new_listeners': sorted(set(after)-set(before)), 'mock_tool_calls': robot.calls}
    (run / 'smoke.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    assert not result.error and robot.calls >= 3 and result.finish_result
    assert not record['new_listeners'] and model.load_count == 1
