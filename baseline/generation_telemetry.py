"""Observational token streamer. Never stops or changes generation."""
import json
import subprocess
import time
from pathlib import Path

from transformers import TextStreamer
from common import append, timestamp


def native_context(model_config, tokenizer):
    text = model_config.get_text_config()
    limit = int(text.max_position_embeddings)
    token_limit = int(tokenizer.model_max_length)
    if limit <= 0 or (token_limit < 10**12 and token_limit != limit):
        raise ValueError(f'Ambiguous native context: model={limit}, tokenizer={token_limit}')
    return limit


def remaining_context(limit, input_tokens, margin=1):
    if margin not in (0, 1):
        raise ValueError('Only a zero/one-token bounds margin is permitted')
    return max(0, limit - input_tokens - margin)


class GenerationTelemetry(TextStreamer):
    def __init__(self, tokenizer, run, call_id, input_tokens, limit, started,
                 progress_tokens=512, progress_seconds=10, validate=None):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=False)
        self.run, self.call_id = Path(run), call_id
        self.input_tokens, self.limit, self.started = input_tokens, limit, started
        self.progress_tokens, self.progress_seconds = progress_tokens, progress_seconds
        self.last_progress, self.last_count, self.persisted = started, 0, 0
        self.generated = []
        self.raw_path = self.run / f'planner_raw_{call_id:03d}.txt'
        self.token_path = self.run / f'planner_tokens_{call_id:03d}.jsonl'
        self.raw = self.raw_path.open('w', encoding='utf-8')
        self.markers = {key: tokenizer.encode(value, add_special_tokens=False) for key, value in {
            'time_to_think_end': '</think>', 'time_to_first_tool_call': '<tool_call>',
            'time_to_complete_tool_call': '</tool_call>'}.items()}
        self.times = {key: None for key in self.markers}
        self.observed_reasoning_tokens = None
        self.time_to_valid_decision = None
        self.validate = validate
        self.ended = False

    def on_finalized_text(self, text, stream_end=False):
        self.raw.write(text)
        self.raw.flush()

    def put(self, value):
        prompt = self.next_tokens_are_prompt
        super().put(value)
        if prompt:
            return
        self.generated.extend(value.reshape(-1).tolist())
        now = time.monotonic()
        for key, marker in self.markers.items():
            if self.times[key] is None and self.generated[-len(marker):] == marker:
                self.times[key] = now - self.started
                if key == 'time_to_think_end':
                    self.observed_reasoning_tokens = len(self.generated)-len(marker)
                if key == 'time_to_complete_tool_call' and self.validate:
                    try:
                        self.validate(self.tokenizer.decode(self.generated, skip_special_tokens=True))
                        self.time_to_valid_decision = now-self.started
                    except Exception:
                        pass  # observation only; final adapter parsing remains authoritative
        if len(self.generated)-self.last_count >= self.progress_tokens or now-self.last_progress >= self.progress_seconds:
            self.progress()

    def progress(self):
        now = time.monotonic(); elapsed = now-self.started
        if self.persisted < len(self.generated):
            append(self.token_path, {'offset': self.persisted, 'token_ids': self.generated[self.persisted:]})
            self.persisted = len(self.generated)
        gpu = {}
        try:
            memory, utilization = subprocess.check_output(['nvidia-smi',
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
            gpu = dict(gpu_memory_mib=int(memory), gpu_utilization_percent=int(utilization))
        except Exception as exc:
            gpu['gpu_probe_error'] = repr(exc)
        append(self.run / 'planner_progress.jsonl', dict(timestamp=timestamp(), call_id=self.call_id,
            input_tokens=self.input_tokens, generated_tokens=len(self.generated), elapsed_seconds=elapsed,
            tokens_per_second=len(self.generated)/elapsed if elapsed else None,
            native_context_limit=self.limit, context_utilization=(self.input_tokens+len(self.generated))/self.limit,
            **self.times, time_to_valid_decision=self.time_to_valid_decision, **gpu))
        self.last_count, self.last_progress = len(self.generated), now

    def end(self):
        if not self.ended:
            super().end()
            self.progress()
            self.raw.close()
            self.ended = True
