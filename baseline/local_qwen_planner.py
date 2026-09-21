"""Local-only Qwen transport for RPent's unmodified PydanticAI tool loop.

No provider, client, server, remote images, or history manager is constructed.
The checkpoint's native tool-call template is used without task demonstrations.
"""
import asyncio
import io
import json
import re
import time
from pathlib import Path

from PIL import Image
from pydantic_ai.messages import (BinaryContent, ModelResponse, TextPart,
    ThinkingPart, ToolCallPart)
from pydantic_ai.models import Model
from pydantic_ai.usage import RequestUsage
from common import append, event, timestamp


def content_items(content):
    if isinstance(content, str):
        return content
    items = []
    for item in content:
        if isinstance(item, str):
            items.append({'type': 'text', 'text': item})
        elif isinstance(item, BinaryContent) and item.media_type.startswith('image/'):
            items.append({'type': 'image', 'image': Image.open(io.BytesIO(item.data)).convert('RGB')})
        else:
            raise TypeError(f'Unsupported local content: {type(item)}')
    return items


def render_messages(messages, instructions):
    out = []
    if instructions:
        out.append({'role': 'system', 'content': '\n\n'.join(x.content for x in instructions)})
    for message in messages:
        if message.kind == 'response':
            row = {'role': 'assistant', 'content': '', 'reasoning_content': '', 'tool_calls': []}
            for part in message.parts:
                if part.part_kind == 'thinking':
                    row['reasoning_content'] += part.content
                elif part.part_kind == 'text':
                    row['content'] += part.content
                elif part.part_kind == 'tool-call':
                    row['tool_calls'].append({'type': 'function', 'id': part.tool_call_id,
                        'function': {'name': part.tool_name, 'arguments': part.args_as_dict()}})
                else:
                    raise TypeError(f'Unsupported response part: {part.part_kind}')
            out.append(row)
        else:
            for part in message.parts:
                if part.part_kind == 'user-prompt':
                    out.append({'role': 'user', 'content': content_items(part.content)})
                elif part.part_kind == 'system-prompt':
                    if out and out[0]['role'] == 'system':
                        out[0]['content'] += '\n' + part.content
                    else:
                        out.insert(0, {'role': 'system', 'content': part.content})
                elif part.part_kind == 'tool-return':
                    out.append({'role': 'tool', 'tool_call_id': part.tool_call_id,
                                'name': part.tool_name, 'content': part.model_response_str()})
                elif part.part_kind == 'retry-prompt':
                    out.append({'role': 'user', 'content': part.model_response()})
                else:
                    raise TypeError(f'Unsupported request part: {part.part_kind}')
    return out


def parse_native_tool(raw, definitions):
    """Decode exactly one native Qwen XML call; validate complete arguments."""
    import jsonschema
    if '</think>' not in raw:
        raise ValueError('Missing reasoning closure')
    reasoning, final = raw.split('</think>', 1)
    matches = list(re.finditer(r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>', final, re.S))
    if len(matches) != 1 or final[matches[0].end():].strip():
        raise ValueError('Expected exactly one complete native tool call with no suffix')
    match = matches[0]
    name, body = match.group(1), match.group(2)
    schema = next((x['function']['parameters'] for x in definitions if x['function']['name'] == name), None)
    if schema is None:
        raise ValueError('Unknown tool: ' + name)
    args = {}
    pattern = r'<parameter=(\w+)>(.*?)</parameter>'
    for parameter in re.finditer(pattern, body, re.S):
        key, value = parameter.group(1), parameter.group(2).strip()
        if key in args:
            raise ValueError('Duplicate argument')
        args[key] = value
    if re.sub(pattern, '', body, flags=re.S).strip():
        raise ValueError('Malformed tool arguments')
    jsonschema.validate(args, schema)
    return reasoning.removeprefix('<think>'), name, args


class LocalQwenPlanner(Model):
    def __init__(self, cfg, run):
        super().__init__()
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.cfg, self.run, self.calls = cfg, Path(run), 0
        self.rpent_turns = 0
        assert cfg['reasoning'] and cfg['dtype'] == 'bfloat16'
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(cfg['model_path'], local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg['model_path'],
            dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa', local_files_only=True).eval()
        assert all(p.device.type == 'cuda' for p in self.model.parameters())
        self.load_count = 1
        from generation_telemetry import native_context
        self.native_context_limit = native_context(self.model.config, self.processor.tokenizer)
        self.before_request = None
        audit = dict(native_context_limit=self.native_context_limit,
            tokenizer_context=self.processor.tokenizer.model_max_length,
            generation_config=self.model.generation_config.to_dict(),
            checkpoint_generation_file_exists=Path(cfg['model_path'], 'generation_config.json').exists(),
            chat_template=self.processor.chat_template,
            model_eos=self.model.generation_config.eos_token_id,
            tokenizer_eos=self.processor.tokenizer.eos_token_id)
        (self.run / 'inference_audit.json').write_text(json.dumps(audit, indent=2, default=str))
        event(self.run, 'local_qwen_loaded', model_object_id=id(self.model), load_count=1,
              seconds=time.monotonic()-started, allocated_bytes=torch.cuda.memory_allocated(),
              backend='transformers', reasoning=True, native_context_limit=self.native_context_limit)

    @property
    def model_name(self):
        return 'local-Qwen3.5-2B'

    @property
    def system(self):
        return 'local-transformers'

    def generate(self, messages, tools=None):
        import torch
        from generation_telemetry import GenerationTelemetry, remaining_context
        self.calls += 1
        start = time.monotonic()
        record = dict(call_id=self.calls, rpent_turn_id=self.rpent_turns or None, timestamp=timestamp(),
            input_tokens=None, output_tokens=None, reasoning_tokens=None, cached_tokens=None,
            model_object_id=id(self.model), load_count=self.load_count, parsed_tool=None,
            parse_failure=None, generation_limit_hit=False, timeout=False, raw_response=None,
            time_to_valid_decision=None)
        stream = None
        record.update(native_context_limit=self.native_context_limit, finish_reason=None,
                      observed_reasoning_tokens=None)
        try:
            messages = [dict(m, content=[{'type': 'text', 'text': m['content']}])
                        if isinstance(m.get('content'), str) else m for m in messages]
            inputs = self.processor.apply_chat_template(messages, tools=tools,
                add_generation_prompt=True, enable_thinking=True, tokenize=True,
                return_dict=True, return_tensors='pt').to('cuda:0')
            n = inputs['input_ids'].shape[1]
            (self.run / f'planner_input_{self.calls:03d}.txt').write_text(
                self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False))
            budget = remaining_context(self.native_context_limit, n, self.cfg['context_margin'])
            record.update(input_tokens=n, native_remaining_generation_tokens=budget)
            if budget <= 0:
                record['finish_reason'] = 'native_context_limit'
                raise RuntimeError('PLANNER_DELIBERATION_CONTEXT_EXHAUSTED')
            stream = GenerationTelemetry(self.processor.tokenizer, self.run, self.calls,
                n, self.native_context_limit, start, self.cfg['progress_tokens'], self.cfg['progress_seconds'],
                validate=(lambda raw: parse_native_tool(raw, tools)) if tools else None)
            stream.progress()
            with torch.inference_mode():
                # This checkpoint has no generation_config.json. Its model
                # fallback EOS is endoftext, but the tokenizer declares im_end
                # as the chat EOS. Respect both; never continue as a fake user.
                eos = self.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                eos = sorted({x for x in [*eos, self.processor.tokenizer.eos_token_id] if x is not None})
                ids = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                    eos_token_id=eos, pad_token_id=self.processor.tokenizer.pad_token_id,
                    streamer=stream)
            torch.cuda.synchronize()
            tokens = ids[0, n:]
            raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            record.update(output_tokens=len(tokens), raw_response_file=str(stream.raw_path),
                eos_token_ids=eos, final_token_id=int(tokens[-1]),
                generation_limit_hit=len(tokens) >= budget and int(tokens[-1]) not in eos,
                finish_reason='eos' if int(tokens[-1]) in eos else 'native_context_limit',
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
            if tools:
                try:
                    thinking, name, args = parse_native_tool(raw, tools)
                    record.update(parsed_tool=name, arguments=args, subgoal=args.get('instruction'))
                    parts = [ThinkingPart(thinking), ToolCallPart(name, args, f'local-{self.calls}')]
                except Exception as exc:
                    record['parse_failure'] = str(exc)
                    if record['generation_limit_hit']:
                        raise RuntimeError('PLANNER_DELIBERATION_CONTEXT_EXHAUSTED') from exc
                    raise
            else:
                thinking, final = raw.split('</think>', 1)
                parts = [ThinkingPart(thinking), TextPart(final.strip())]
            record['time_to_valid_decision'] = stream.time_to_valid_decision or time.monotonic() - start
            return ModelResponse(parts=parts, model_name=self.model_name,
                usage=RequestUsage(input_tokens=n, output_tokens=len(tokens)))
        except Exception as exc:
            record['error'] = repr(exc)
            if isinstance(exc, torch.OutOfMemoryError):
                record['finish_reason'] = 'cuda_oom'
            elif record['finish_reason'] is None:
                record['finish_reason'] = 'fatal_exception'
            raise
        finally:
            if stream is not None:
                stream.end()
                record.update(output_tokens=len(stream.generated), raw_response_file=str(stream.raw_path),
                    raw_reasoning_length_chars=len(self.processor.tokenizer.decode(stream.generated,
                        skip_special_tokens=True).split('</think>', 1)[0]),
                    observed_reasoning_tokens=stream.observed_reasoning_tokens if stream.observed_reasoning_tokens is not None
                        else len(stream.generated),
                    time_to_valid_decision=stream.time_to_valid_decision,
                    context_utilization=((record['input_tokens'] or 0)+len(stream.generated))/self.native_context_limit,
                    **stream.times)
            record['request_latency'] = time.monotonic() - start
            record['generation_duration'] = record['request_latency']
            record['tokens_per_second'] = (record.get('output_tokens') or 0)/record['request_latency']
            append(self.run / 'planner.jsonl', record)
            print('LOCAL_QWEN_CALL', json.dumps({k:v for k,v in record.items() if k != 'raw_response'}), flush=True)

    async def request(self, messages, model_settings, model_request_parameters):
        if getattr(self, 'before_request', None):
            self.before_request()
        self.rpent_turns += 1
        tools = [{'type': 'function', 'function': {'name': t.name, 'description': t.description,
                    'parameters': t.parameters_json_schema}}
                 for t in model_request_parameters.function_tools]
        rendered = render_messages(messages, self._get_instruction_parts(messages, model_request_parameters))
        return await asyncio.to_thread(self.generate, rendered, tools)
