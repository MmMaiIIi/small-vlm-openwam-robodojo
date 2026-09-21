# Raw unrestricted RPent baseline — completed lifecycle

Run: `runs/20260921T224427_rpent_unrestricted_fill_pen_holder_0/`.
Task: **fill_pen_holder / seed 0**, 2026-09-21.

**The complete multi-decision episode reached the official horizon. The task
was not successful.** No additional episode, sweep, prompt optimization,
model change, reasoning suppression or training followed this run.

## Actual configuration

Qwen3.5-2B, local Transformers 5.5.4, BF16 CUDA, reasoning ON, one model load,
no HTTP/API. RPent owns the unchanged planner/tool/history loop.

- Native context: **262144**, read from both checkpoint text_config.max_position_embeddings
  and tokenizer.model_max_length. No generation_config.json is present.
- For each call: max_new_tokens = native context - actual input tokens - 1.
  No 6144 cap, 8192 context override, planner timeout, or deadline stop criterion.
- Both official EOS IDs 248044 / 248046 are enabled. Every real turn ended
  naturally on **248046**. Marker timestamps are observed, not stop sequences.
- Official env.step_lim: **1100**. RPent integer sentinel: 1101, never reached.
  The old artificial 8-turn and 512-step limits were removed.
- OpenWAM remains native replan=32; high-level execution window=64. No change
  to actor checkpoint, inference algorithm, physics, reward, success or assets.

See [exact protocol and progress telemetry](RPENT_UNRESTRICTED_PROTOCOL.md).

## Episode result

| Metric | Actual result |
|---|---|
| RPent turns / accepted tool calls | 18 / 18 |
| Planner input / output tokens, summed across calls | 86563 / 2352 |
| Backend reasoning tokens / cached tokens | null / null |
| Observed reasoning tokens, separate token-ID measurement | 1659 |
| First decision | 441 output tokens, 402 observed reasoning tokens, 13.804 s |
| Planner total generation time | 83.872 s |
| Planner mean / p50 latency | 4.660 / 3.796 s |
| Overall output tokens / total generation time | 28.043 tokens/s |
| OpenWAM executions | 18 windows; 35 native inference calls |
| OpenWAM mean / total inference time | 0.962 / 33.665 s |
| Executed env steps | 1100 = 17 × 64 + 12 |
| Unexecuted stale actions discarded | 20 in the final 32-action chunk |
| Episode wall time, including shutdown, excluding startup | 286.697 s |
| End-to-end wall time, including all startup/cleanup | 486.393 s |
| Simulated time | 44.000 s |
| Peak total GPU memory, sampled 1 Hz | 39815 MiB = 38.882 GiB |
| OOM / fatal crash | No / No |
| Final task success | **false** |
| Actual termination | **official_horizon** at env step 1100 |
| Video | 1101 frames, 1920×480, 25 FPS, 44.040 s including initial frame |
| Model load count | 1; same Python model object for all 18 decisions |
| Cleanup | Recorded planner/actor/Isaac processes exited; GPU returned to 136 MiB |

All 18 planner calls produced execute_subgoal. The 18 instruction-reset
acknowledgements, fresh observations and chunk accounting are in the run.
All 35 actor records sum to 1100 executed actions. No partial output was used.
No context boundary was approached and no latency-based stop occurred.

The control-only lifecycle signal `OFFICIAL_EPISODE_ENDED` is deliberately
caught at RPent's next request boundary after the final tool. The raw summary
retains it in `planner_error`; it is **not** a model failure or process crash.
The guard runs before model generation/call counting, so no 19th Qwen call was
made. Benchmark end state never becomes model-visible feedback.

## Per-turn measurements

Times below are seconds from the start of each planner call, including prompt
preprocessing/prefill. Generation duration includes telemetry overhead. The
valid-call timestamp is the first complete schema-valid tool call observed;
execution waits for natural EOS and the adapter's final validation.

| Turn | Input | Output | Observed reasoning | Duration s | Tokens/s | Think end s | First tool start s | Valid tool s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 952 | 441 | 402 | 13.804 | 31.947 | 12.687 | 12.744 | 13.741 |
| 2 | 1488 | 87 | 49 | 3.108 | 27.989 | 1.989 | 2.048 | 3.032 |
| 3 | 2024 | 149 | 110 | 4.924 | 30.259 | 3.736 | 3.793 | 4.808 |
| 4 | 2561 | 130 | 91 | 4.543 | 28.613 | 3.367 | 3.423 | 4.427 |
| 5 | 3098 | 94 | 55 | 3.291 | 28.563 | 2.203 | 2.262 | 3.234 |
| 6 | 3634 | 151 | 111 | 4.944 | 30.544 | 3.797 | 3.853 | 4.864 |
| 7 | 4171 | 109 | 69 | 3.814 | 28.580 | 2.668 | 2.724 | 3.741 |
| 8 | 4709 | 104 | 65 | 3.826 | 27.186 | 2.712 | 2.767 | 3.749 |
| 9 | 5246 | 83 | 47 | 3.390 | 24.482 | 2.296 | 2.354 | 3.275 |
| 10 | 5780 | 179 | 141 | 6.126 | 29.217 | 5.015 | 5.072 | 6.030 |
| 11 | 5963 | 157 | 118 | 5.383 | 29.165 | 4.267 | 4.323 | 5.291 |
| 12 | 6149 | 94 | 55 | 3.691 | 25.466 | 2.587 | 2.642 | 3.614 |
| 13 | 6335 | 70 | 31 | 3.011 | 23.245 | 1.897 | 1.951 | 2.918 |
| 14 | 6521 | 70 | 31 | 3.227 | 21.690 | 2.057 | 2.117 | 3.135 |
| 15 | 6707 | 80 | 44 | 3.318 | 24.109 | 2.297 | 2.352 | 3.247 |
| 16 | 6889 | 186 | 147 | 6.379 | 29.157 | 5.288 | 5.343 | 6.310 |
| 17 | 7076 | 77 | 41 | 3.312 | 23.248 | 2.254 | 2.311 | 3.218 |
| 18 | 7260 | 91 | 52 | 3.778 | 24.085 | 2.654 | 2.710 | 3.703 |

`reasoning_tokens` is null in every row's source record. The observed column
counts actual generated token IDs preceding the native </think> marker; it is
not a claimed backend usage field. Exact token IDs and streamed raw output are
saved separately per call. Context utilization and finish/EOS are also in
planner.jsonl and planner_progress.jsonl.

## Comparison boundary: do not overinterpret the shorter first decision

The older 6144-token blocked run is intact. The first tokenized text/template
is byte-identical between that run and this run, and both have 952 input tokens.
However, the actual lossless initial PNG observations are **not pixel-identical**,
even with the same task/seed. From the stored RPent BinaryContent inputs:

- Old PNG SHA256: `3dd7cd382036a24b24e6766ceb890fa1973845d3920a3830b26dc91430392cd6`
- New PNG SHA256: `2afa6e4faee250648c97f36d5a634232fb22e07bfcee72023fac6fd8c4ee15be`
- Shape: 312×1152×3 in both; mean absolute uint8 pixel difference:
  0.316027 on the 0–255 scale.

Thus this is evidence of a working unrestricted lifecycle, **not** a controlled
causal demonstration that removing a token cap shortened reasoning. No hidden
prompt change, replayed plan, early thinking exit, or image modification was used.
The source of the small observation difference was not altered or optimized.

## Validation and artifacts

- 3 unrestricted unit tests: checkpoint-derived context, non-stopping incremental
  streamer/exact token persistence, lifecycle signal not exposed to policy.
- 6 RPent adapter tests and 3 actor/runtime tests passed.
- Actual local text/image plus 3-turn RPent mock passed with AF_INET/AF_INET6
  connect/bind forbidden. Model loaded once; no Qwen listening ports.
- Real run checks: 18 valid EOS decisions, same model object, exact token counts
  match persisted token-ID arrays, 35 actor calls/1100 actions, full readable MP4.
- Old reports/runs retained. No upstream repo or model/assets were changed.

Raw artifacts: config.yaml, inference_audit.json, versions.json, planner.jsonl,
planner_progress.jsonl, planner_raw_NNN.txt, planner_tokens_NNN.jsonl,
planner_input_NNN.txt, actor.jsonl, tools.jsonl, events.jsonl, gpu.jsonl,
rpent_result.json, simulator.json, summary.json, timeline.md, stdout.log,
video.mp4. Runs/models/assets/logs/videos are excluded from Git.

**RAW UNRESTRICTED RPENT BASELINE COMPLETED**

