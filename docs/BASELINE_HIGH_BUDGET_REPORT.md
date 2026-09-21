# High-budget baseline retry — PLANNER_DELIBERATION / TIMING

Date: 2026-09-21. Task: **fill_pen_holder / seed 0 only**.

Two complete decision/execution/observation cycles succeeded. The third planner
request consumed all 6144 generated tokens without producing a final JSON decision.
The episode stopped safely after 128 control steps. **It did not naturally complete.**
No prompt optimization, non-thinking fallback, extra mechanism, or 8-task sweep was performed.

## Exactly what changed

The user authorized two config changes relative to the original raw attempt:

| Setting | Previous | This run |
|---|---:|---:|
| planner.max_tokens | 1536 | 6144 |
| planner.timeout | 90 s | 180 s |

Context stays 8192, leaving 2048 tokens reserved for input. All other YAML settings
are identical: reasoning ON, same Qwen and OpenWAM weights, actor replan 32,
high-level window 64, 25 Hz, task/seed, history, temperature, 8-call/512-step cap.
The planner, actor, and runtime compatibility source snapshots are byte-identical
to the previous real attempt; prompts were not changed. Upstream commits and dirty
diff hashes are also unchanged. Previously added recording/cleanup improvements do
not alter policy or simulator behavior and are retained in the source snapshot.

## Evidence

Run directory: `runs/20260921T214249_fill_pen_holder_0/`.

- [Summary](../runs/20260921T214249_fill_pen_holder_0/summary.json)
- [Planner responses and usage](../runs/20260921T214249_fill_pen_holder_0/planner.jsonl)
- [Actor accounting](../runs/20260921T214249_fill_pen_holder_0/actor.jsonl)
- [Timeline](../runs/20260921T214249_fill_pen_holder_0/timeline.md)
- [Video](../runs/20260921T214249_fill_pen_holder_0/video.mp4)
- [Config](../runs/20260921T214249_fill_pen_holder_0/config.yaml)
- [Versions](../runs/20260921T214249_fill_pen_holder_0/versions.json)

Original raw responses and events remain unchanged. The summary's
`post_run_failure_review` adds the requested failure subtype and an offline review
of the saved response; it is not an online deliberation detector or a planner change.

## Actual planner consumption

| Call | Input tokens | Output tokens | Reasoning tokens | Request latency | Time to valid decision | Result |
|---|---:|---:|---|---:|---|---|
| 1 | 617 | 643 | null | 18.330 s | 18.330 s | EXECUTE: Grasp the pen holder |
| 2 | 638 | 901 | null | 22.500 s | 22.500 s | EXECUTE: Grasp a pen |
| 3 | 654 | 6144 | null | 146.735 s | null | No final JSON; finish_reason=length |
| Total | 1909 | 7688 | null | 187.565 s | — | 3 calls, 2 valid decisions |

All three responses contain reasoning text. The API does not report separate
reasoning-token or cached-token usage, so those values remain null. Output usage is
the API's completion-token count, not an inferred split between reasoning and answer.

The first two calls finished naturally well below the high cap. The third did not:
its response contains 23,228 reasoning characters and visibly repeated passages.
An offline count found one identical nontrivial line repeated 82 times (434
nontrivial lines, only 36 distinct). No truncated fragment was parsed or executed.

**Failure classification: PLANNER_DELIBERATION / TIMING.**

- Concrete stopping mechanism: generation budget exhausted at 6144 output tokens.
- Not the 180-second wall-clock timeout: the response returned after 146.735 s.
- Not full context exhaustion: 654 input + 6144 output = 6798 total tokens, leaving
  1394 of the 8192-token context window unused.
- No evidence of OOM or an interface crash in this attempt.
- This finite trace demonstrates repetitive deliberation and failure to finalize;
  it does not establish that an unlimited run would literally continue forever.

## Episode and system metrics

| Metric | Result |
|---|---:|
| Natural episode completion / task success | No / No |
| Valid high-level decisions | 2 |
| Environment steps / simulated control time | 128 / 5.12 s |
| Episode wall-clock time | 210.492 s |
| Full launch → cleanup wall-clock time | 449.186 s |
| Planner mean latency, including failed call | 62.522 s |
| Planner mean latency for the two valid decisions | 20.415 s |
| OpenWAM inference calls | 4 |
| OpenWAM total / mean inference latency | 5.115 s / 1.279 s |
| Predicted / executed actions | 128 / 128 |
| Stale actions remaining / dropped | 0 / 0 |
| Peak total GPU memory, 1-second samples | 40,005 MiB = 39.07 GiB |
| OOM | No |
| Video | 129 frames, 1920×480, 5.16 s |
| Clean exit / process cleanup | Yes; GPU returned to 136 MiB |

Both generations have explicit instruction-reset acknowledgements. Each produced
two 32-action chunks, all fully consumed before the next planner request. This now
verifies instruction replacement across **two real subgoals**, in addition to the
existing unit tests. Updated RGB reached the next planner call after both windows.

The episode ended on the third request's deliberation failure, not on the unchanged
8-call or 512-step harness cap. Neither reaching that cap nor a clean process exit
would itself establish natural task completion. `completed` and `pipeline_valid`
therefore remain false for this run.

Per-process NVML memory is unavailable in this container. Startup-delta estimates
are Qwen 6252 MiB, OpenWAM 24299 MiB, Isaac 8764 MiB; they are not precise process
peaks. Raw 1 Hz memory/utilization samples are retained in `gpu.jsonl`.

## Scope and next state

The minimal hierarchy can execute consecutive real decisions, but the requested
full naturally ended episode is still not verified. The high-budget failure is
preserved as a raw baseline result. No additional run or automatic budget escalation
was started, and no other task was launched.
