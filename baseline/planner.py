"""A bounded vision -> one JSON decision call; no simulator state input."""
import base64
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw
from common import append, timestamp

SCHEMA = {'oneOf': [
    {'type': 'object', 'properties': {'action': {'const': 'execute'},
      'subgoal': {'type': 'string', 'minLength': 1, 'maxLength': 240}},
     'required': ['action', 'subgoal'], 'additionalProperties': False},
    {'type': 'object', 'properties': {'action': {'enum': ['wait', 'finish']}},
     'required': ['action'], 'additionalProperties': False},
]}
RULES = """You control a robot through a low-level language-conditioned actor.
Use the original task and current camera views to choose only the next short subgoal.
The panels are head, left wrist, right wrist, in that order. Images are the current scene.
Use your recent decisions only as attempted actions, never as proof of completion.
EXECUTE provides one concise textual subgoal, which may include holding an object
with one hand while manipulating with the other. It runs for a fixed short window.
WAIT lets physics advance while holding the current robot target. FINISH ends the episode
only when the visible task appears complete. Inspect each new observation to decide.
Do not output a full plan, joint values, EE deltas, distances, coordinates, or trajectories.
After reasoning, output exactly one JSON object matching the schema:
{"action":"execute","subgoal":"..."}, {"action":"wait"}, or {"action":"finish"}.
"""


class PlannerFailure(RuntimeError):
    pass


def parse_decision(content):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate JSON key')
            obj[key] = value
        return obj
    d = json.loads(content, object_pairs_hook=unique)
    if not isinstance(d, dict):
        raise ValueError('decision must be an object')
    if d.get('action') == 'execute':
        if set(d) != {'action', 'subgoal'} or not isinstance(d['subgoal'], str) or not 1 <= len(d['subgoal'].strip()) <= 240:
            raise ValueError('invalid execute schema')
        d['subgoal'] = d['subgoal'].strip()
    elif d.get('action') in ('wait', 'finish'):
        if set(d) != {'action'}:
            raise ValueError('invalid terminal/wait schema')
    else:
        raise ValueError('unknown action')
    return d


def camera_panel(obs):
    names = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']
    frames = [Image.fromarray(obs['vision'][n]['color'][:, :, :3]).resize((384, 288)) for n in names]
    canvas = Image.new('RGB', (1152, 312))
    draw = ImageDraw.Draw(canvas)
    for i, (name, frame) in enumerate(zip(names, frames)):
        canvas.paste(frame, (384 * i, 24))
        draw.text((384 * i + 5, 5), name, fill='white')
    return canvas


class Planner:
    def __init__(self, cfg, run, port):
        self.cfg, self.run, self.port = cfg, Path(run), port
        self.calls = 0
        self.client = httpx.Client(trust_env=False, timeout=cfg['timeout'])

    def decide(self, instruction, obs, observation_timestamp, history):
        image = camera_panel(obs)
        image.save(self.run / f'planner_observation_{self.calls + 1:03d}.jpg')
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=90)
        data_url = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
        # Progress is only the preceding model decisions, never rewards/poses/success.
        context = {'original_task_instruction': instruction,
                   'progress': 'No previous decisions.' if not history else 'Recent attempted decisions; completion is unverified.',
                   'recent_decisions': history[-self.cfg['history_length']:]}
        start = time.monotonic()
        for attempt in range(self.cfg['invalid_retries'] + 1):
            self.calls += 1
            record = dict(call_id=self.calls, timestamp=timestamp(),
                          observation_timestamp=observation_timestamp, attempt=attempt,
                          input_tokens=None, output_tokens=None, reasoning_tokens=None, cached_tokens=None,
                          timeout=False, parse_failure=False, raw_response=None, parsed_action=None, subgoal=None)
            request_start = time.monotonic()
            try:
                prompt = json.dumps(context, ensure_ascii=False)
                if attempt:
                    prompt += '\nPrevious output failed validation. Return only a complete schema-valid decision.'
                response = self.client.post(f'http://127.0.0.1:{self.port}/v1/chat/completions', json={
                    'model': self.cfg['model'], 'temperature': self.cfg['temperature'],
                    'max_tokens': self.cfg['max_tokens'], 'seed': 0,
                    'chat_template_kwargs': {'enable_thinking': self.cfg['reasoning']},
                    'response_format': {'type': 'json_schema', 'json_schema': {'name': 'decision', 'strict': True, 'schema': SCHEMA}},
                    'messages': [{'role': 'system', 'content': RULES}, {'role': 'user', 'content': [
                        {'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': data_url}}]}]})
                record['raw_response'] = response.text
                response.raise_for_status()
                data = response.json()
                record['raw_response'] = data
                usage = data.get('usage', {})
                record.update(input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'),
                    reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                    cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'))
                choice = data['choices'][0]
                if choice.get('finish_reason') == 'length':
                    record['timeout'] = True
                    raise PlannerFailure('planner_timeout: generation token budget exhausted')
                decision = parse_decision(choice['message']['content'])
                record.update(parsed_action=decision['action'], subgoal=decision.get('subgoal'),
                              time_to_valid_decision=time.monotonic() - start)
                return decision
            except httpx.TimeoutException as exc:
                record['timeout'] = True
                raise PlannerFailure('planner_timeout') from exc
            except (ValueError, KeyError, TypeError) as exc:
                record.update(parse_failure=True, parse_error=str(exc))
                if attempt == self.cfg['invalid_retries']:
                    raise PlannerFailure('planner_failure: invalid JSON/schema after retry') from exc
            finally:
                record['request_latency'] = time.monotonic() - request_start
                record.setdefault('time_to_valid_decision', None)
                append(self.run / 'planner.jsonl', record)
