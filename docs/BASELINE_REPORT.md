# Raw baseline: partial pipeline verified, completion gate NOT passed

> Latest authorized retry: [high-budget report](BASELINE_HIGH_BUDGET_REPORT.md).
> With 6144 output tokens / 180 s, two valid decisions executed 128 steps, but the
> third call exhausted the generation budget. Natural completion is still unverified.
> The rest of this document preserves the earlier 1536-token attempt unchanged.

Date: 2026-09-21. No optimization, additional model, training, or benchmark was performed.

The real chain ran Qwen reasoning → EXECUTE → OpenWAM → 64 native control steps →
new RGB → second Qwen reasoning request. The second request exhausted the configured
1536-token generation budget and returned `finish_reason=length`, `content=null`.
It was recorded as `planner_timeout` and execution stopped safely. This was **not**
the 90-second HTTP timeout, an OOM, or a simulator crash. No partial decision was executed.

**A complete multi-decision episode has NOT passed. The other seven tasks were not
launched, respecting the requested first-task gate. Pipeline readiness cannot yet be declared.**

## Architecture and reproducible entry point

Qwen3.5-2B/vLLM (reasoning ON, vision + strict JSON) → minimal synchronous controller
→ official XPolicyLab OpenWAM adapter/WebSocket protocol → official RoboDojo EvalEnv.
RPent core was inspected but is not executed in this minimal path.

```bash
cd /root/gpufree-data/robot-agent-exp
./run_baseline.sh --task fill_pen_holder --seed 0
# To stop only project-owned processes:
./stop.sh
```

The original raw config remains unchanged: context 8192, max generated tokens 1536,
planner timeout 90s, actor replan 32, high-level window 64 control steps at 25 Hz,
max 8 planner API calls / 512 env steps. The official task limit remains 1100.
These are bounded interface-baseline settings, not a full-horizon benchmark.
Increasing the token budget or changing prompting has deliberately not been done.

## Actual run and artifacts

Run: `runs/20260921T211840_fill_pen_holder_0/`.

- [Summary](../runs/20260921T211840_fill_pen_holder_0/summary.json)
- [Timeline](../runs/20260921T211840_fill_pen_holder_0/timeline.md)
- [Video](../runs/20260921T211840_fill_pen_holder_0/video.mp4)
- [Exact config](../runs/20260921T211840_fill_pen_holder_0/config.yaml)
- [Versions](../runs/20260921T211840_fill_pen_holder_0/versions.json)
- `source/` contains the code snapshot actually used by the actor and simulator.
- `planner.jsonl`, `actor.jsonl`, `events.jsonl`, `gpu.jsonl`, `stdout.log`, and
  `planner_observation_001.jpg` / `002.jpg` retain the underlying evidence.

Metrics below are from this single raw attempt, not averages across selected reruns.
Earlier launches failed before any policy decision and remain preserved separately.

| Measurement | Actual result |
|---|---|
| Task / seed | fill_pen_holder / 0 |
| Complete episode / task success | No / No |
| Clean simulator exit | Yes; launcher returns failure status |
| Environment control steps | 64 |
| Simulation time after reset | 2.56 s |
| Episode wall clock | 68.57 s |
| End-to-end launch, load, episode, cleanup | 304.09 s |
| Planner HTTP calls / valid decisions | 2 / 1 |
| Planner latencies | 20.288 s; 36.643 s (token-budget failure) |
| Mean planner latency, including failed request | 28.465 s |
| OpenWAM inferences | 2 |
| OpenWAM latencies / mean | 2.283 s; 0.876 s / 1.579 s |
| Total planner / OpenWAM inference time | 56.931 s / 3.158 s |
| Input / output tokens | 1,258 / 2,269 |
| Reasoning tokens / cached tokens | null / null: API did not provide these breakdowns |
| Reasoning actually present | Both responses contain reasoning; not silently disabled |
| Predicted / executed actions | 32/32 in each chunk, 64 total |
| Stale actions dropped | 0; both chunks were fully consumed before the next planner call |
| OOM | No |
| Sampled peak total VRAM | 39,973 MiB = 39.04 GiB, sampled every second |
| All-resident VRAM near reset completion | 39,419 MiB = 38.50 GiB |
| OpenWAM allocator peak during inference | 23.39 GiB (process allocator, not all-driver memory) |
| Video | 1920×480, 65 frames, 2.60 s, fully decoded successfully |
| Disk remaining after tests | Approximately 28 GiB |

Container NVML does not expose per-process memory records: these are **null**, not zero.
Sequential startup device-memory deltas estimate Qwen 6.11 GiB, OpenWAM 23.73 GiB,
Isaac 8.53 GiB. They are estimates, not precise per-process peaks; sampled totals are
the direct measurements. GPU utilization is retained in `gpu.jsonl`. Local tokens,
GPU utilization, and wall time are cost-related measurements; no dollar cost is invented.

## Task gate

| Task | Hierarchical episode | Result |
|---|---|---|
| fill_pen_holder | Started; 64 steps; one valid decision | TIMING: second request token budget exhausted |
| classify_objects | Not run | First-task gate not passed |
| put_bottles_into_dustbin | Not run | First-task gate not passed |
| play_tic_tac_toe | Not run | First-task gate not passed |
| fill_egg_holder | Not run | First-task gate not passed |
| organize_table | Not run | First-task gate not passed |
| make_kong | Not run | First-task gate not passed |
| play_stacking_toy | Not run | First-task gate not passed |

Previously completed environment-only scene smoke tests are not counted as hierarchical episodes.
No statistically meaningful success rate can be estimated from this interrupted attempt.

## Human-readable interpretation and failure taxonomy

- **TIMING — confirmed primary blocker.** At t=20.31s the first valid decision was
  `Hold the pen holder with the left hand`. Two native chunks executed. At t=31.91s,
  updated RGB returned. At t=68.57s the second request exhausted 1536 output tokens
  without a complete JSON decision. The observed reasoning is repetitive, but this
  finite trace alone does not establish an infinite loop.
- **INTERFACE — startup issues fixed, with traces preserved.** Local health requests
  initially inherited proxy settings. The real EvalEnv reset then exposed a CUDA-pose
  to NumPy evaluator mismatch absent from scene-only smoke. A first compatibility
  attempt also exposed the parser's deferred initialization order. These were fixed
  locally; the final run passed reset, RGB, actor RPC, native stepping, and observation
  return. No new upstream source modifications were made.
- **PHYSICAL_EXECUTION — candidate observation, not proven root cause.** Image 002
  shows the left gripper around a white pen while the orange holder remains on the
  table, although the instruction requested holding the holder. The 64-step window
  did not visibly achieve that subgoal. This is not sufficient to claim a general
  actor limitation or attribute every motion error to the actor.
- **HIGH_LEVEL_REASONING — candidate grounding concern.** The first subgoal was
  broadly consistent with the task. The second response's unfinished reasoning
  considered proceeding to pen placement despite the visible mismatch. There was
  no second valid action, so this was not an executed high-level decision.
- **UNKNOWN — remaining causality.** One partial episode cannot separate instruction
  grounding, fixed-window sufficiency, grasp execution, and visual interpretation.
  No prompt, action window, verifier, or planner mechanism was changed to improve it.

The first instruction replacement and empty client queue are verified in real RPC
records. Cross-subgoal stale-action rejection/discard passes unit tests, but a second
real subgoal was never issued; multi-generation integration remains unverified.

## Files and compatibility scope

New research files: `configs/baseline.yaml`; `baseline/{common,planner,actor_server,
simulator,runtime_compat,launch,stop}.py`; `run_baseline.sh`; `stop.sh`;
`tests/test_{planner,runtime_compat,chunks}.py`; `docs/INTERFACES.md`; this report;
`patches/README.md`. Existing project README and .gitignore were extended.

`runtime_compat.py` installs a view only on the run's official NumPy evaluator.
Actual inputs were float32 CUDA tensors of shape [3] and [4]; outputs were equal-value
float32 CPU ndarrays of the same shapes. It does not modify physics, coordinates,
DOF values, task definitions, rewards, success predicates, or planner inputs.
See `patches/README.md` for exact trace and scope. The old authorized RoboDojo
fill_egg_holder compatibility patch is preserved; no additional upstream file changed.

Every launch snapshots research source/config. Summary postprocessing records its own
source hash; startup memory estimates were corrected to use the first sample **after**
each readiness marker, not an in-progress model-loading sample before it.

## Upstream versions

| Component | Commit / revision |
|---|---|
| RoboDojo | 726e9aabfaa642203722eb126f5eaf0f37f3e1ad + pre-existing runtime patch |
| XPolicyLab | bb9a0b5f5136a74503b679af830bfd0a3a837d5c |
| OpenWAM source | Vendored within that XPolicyLab checkout; no independent Git HEAD |
| cuRobo | d17b54ce32cba095c0b000c4c58777075d11de0e |
| RPent | 63c01fbf2ae9a6ab03bbc13eb9abf35853a71a9f; not executed |
| OpenWAM checkpoint | 2c1302294e3ba8319bbdb2c803b7a27de9292d03 |
| Qwen checkpoint | 15852e8c16360a2fea060d615a32b45270f8a8fc |

Isaac Sim 5.1.0.0 / Isaac Lab 2.3.2.post1; runtime Torch 2.7.0+cu128;
actor Torch 2.7.1+cu128; Qwen vLLM 0.20.1 / Torch 2.11.0+cu130.
Full installed-package lists and executable paths are in `versions.json`.

## Verification and remaining warnings

- Six unit tests pass in the OpenWAM environment: strict parsing, single retry,
  no partial execution, privileged-input exclusion, deferred evaluator initialization,
  dtype/value preservation, and chunk generation/discard checks (some tests cover
  multiple assertions). These do not substitute for a complete real episode.
- Python compilation, shell syntax checks, and `git diff --check` pass.
- FFmpeg decoded every recorded video frame without an error.
- Idempotent `stop.sh` was checked after completion. No project model/server remained;
  GPU returned to 136 MiB. No global process-kill command was used.
- Existing material/texture, GPU velocity API, and teardown tensor-view warnings
  remain in stdout; assets were not modified. They did not prevent this scene's
  reset, camera observations, or 64 control steps.

## Remaining blocker

The unchanged raw planner budget did not produce a second valid decision. Continuing
requires an explicitly accepted configuration change or another user-selected course;
the budget/prompt has not been silently changed to obtain a passing run. No eight-task
sweep, optimization method, or additional experiment has been started.
