"""One-off feasibility diagnostic, not a maintained planner backend."""
import json
import os
import subprocess
from pathlib import Path

if __name__ == '__main__':
    from vllm import LLM, SamplingParams
    run = Path(os.environ['PROBE_RUN'])
    run.mkdir(parents=True, exist_ok=True)
    before = subprocess.check_output(['ss', '-H', '-ltn'], text=True)
    llm = LLM(model='/root/gpufree-data/models/Qwen/Qwen3.5-2B', dtype='bfloat16',
              max_model_len=8192, max_num_seqs=1, gpu_memory_utilization=.35,
              kv_cache_memory_bytes=1073741824, enforce_eager=True,
              limit_mm_per_prompt={'image': 1, 'video': 0})
    during = subprocess.check_output(['ss', '-H', '-ltn'], text=True)
    output = llm.chat([{'role': 'user', 'content': 'What is 2 + 2?'}],
        SamplingParams(max_tokens=6144, temperature=0),
        chat_template_kwargs={'enable_thinking': True}, use_tqdm=False)[0]
    after = subprocess.check_output(['ss', '-H', '-ltn'], text=True)
    result = dict(before=before, during=during, after=after,
                  new_listeners=sorted({' '.join(r.split()[3:5]) for r in during.splitlines()}
                      - {' '.join(r.split()[3:5]) for r in before.splitlines()}),
                  input_tokens=len(output.prompt_token_ids), output_tokens=len(output.outputs[0].token_ids),
                  raw_output=output.outputs[0].text)
    (run / 'probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
