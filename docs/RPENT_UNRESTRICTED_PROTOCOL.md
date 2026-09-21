# Raw unrestricted RPent baseline

This protocol supersedes the artificial limits in the earlier report; it does
not overwrite earlier runs or their conclusions. Task: fill_pen_holder, seed 0.

## Limits derived from the installed artifacts

- Qwen config.json: text_config.max_position_embeddings = 262144.
- tokenizer_config.json: model_max_length = 262144.
- No checkpoint generation_config.json; model EOS = 248044, tokenizer chat
  EOS = 248046. Both official end tokens are respected, as in the corrected run.
- No RoPE extension, model patch, quantization, CPU offload or alternate backend.
- For each actual tokenized prompt: max_new_tokens = 262144 - input_tokens - 1.
  The one-token margin is solely a bounds guard. No smaller output cap, max_time,
  deadline StoppingCriteria, hidden stop string or reasoning truncation.
- Official task class FillPenHolderCommon sets step_lim = 1100. The worker reads
  the actual initialized env.step_lim; it does not install a replacement limit.
- RPent requires an integer max_turns. It receives actual env.step_lim + 1;
  each successful nonterminal tool advances up to 64 steps, so this sentinel
  cannot preempt the official horizon. At official termination a control-only
  lifecycle signal prevents another model request. It is not planner feedback.

The original planner prompt and three tool schemas remain unchanged. RPent
still owns its loop/history/image pruning; no new memory or planning policy.
OpenWAM checkpoint, precision, inference settings, replan_steps=32 and the
64-step execution window are unchanged. The artificial actor RPC request
deadline is disabled at the run-local transport boundary, not in model code.
Startup and robot-IPC wall-time deadlines are removed. Natural EOS/finish,
official environment termination, OOM/fatal exception, native context
exhaustion, or a user stop determine the outcome. Slowness alone does not.

## Observability

`GenerationTelemetry` is a standard Transformers TextStreamer. It neither
returns a stop decision nor changes tokens/logits. It writes every decoded
fragment to planner_raw_NNN.txt and flushes it. Every 512 generated tokens or
10 seconds (whichever comes first while tokens advance), it appends:

- planner_progress.jsonl: token count, elapsed time, mean tokens/s, GPU memory
  and utilization, context utilization, observed marker timestamps;
- planner_tokens_NNN.jsonl: exact generated token IDs since the last progress.

No complete reasoning is printed to the terminal. RPent retains its existing
short console preview. Final planner.jsonl records per-call actual input/output,
raw file path, generation duration, tokens/s, time to thinking closure, first
tool-call opening, complete call, schema-valid call, EOS/termination and context
utilization. reasoning_tokens remains null; observed_reasoning_tokens counts
generated tokens before the exact `</think>` token sequence, or all generated
tokens if closure never arrives. This is not backend-reported usage.

Markers are observations, **not stop sequences**. Only official EOS ends an
ordinary model generation; a completed tool call is validated before RPent
executes it. No half-parsed prefix is executed. No automatic hang detector
equates low token throughput with a deadlock. A suspected stall must be
investigated using both token progress and GPU/process state.

## Reproduce

```bash
cd /root/gpufree-data/robot-agent-exp
./run_baseline.sh --task fill_pen_holder --seed 0
```

Default config: configs/rpent_qwen2b_openwam_unrestricted.yaml. Each attempt
uses a new `_rpent_unrestricted_fill_pen_holder_0` run directory. Previous
6144-cap run `20260921T222051_rpent_fill_pen_holder_0` is retained unchanged.

The standalone mock diagnostic has a small fixture turn count to test tool
termination; that fixture is not used as an episode or planner budget in the
real baseline. Its real model generation uses the same native context limit.
