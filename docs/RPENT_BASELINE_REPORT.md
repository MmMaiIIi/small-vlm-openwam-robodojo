# RPent + local Qwen baseline — blocked, not READY

Date: 2026-09-21. Only **fill_pen_holder / seed 0** was attempted. No sweep,
prompt optimization, reasoning suppression, model change or training.

## Actual architecture and implementation

Local Qwen3.5-2B BF16 CUDA → RPent `ApiAgentLoop` → RPent `Toolkit` →
finite-window Isaac worker / official OpenWAM WebSocket actor → RGB feedback.

Reused RPent modules: planner `solve/_solve`, observer, PydanticAI conversation
history, `_prune_history_images`, tool schema wrappers, `add_tool`, `execute_tool`,
`ToolResult` image feedback, `EnvState`, and finish/max-turn termination.
No custom planner loop/history manager and no upstream RPent patch.

New files: `baseline/local_qwen_planner.py`, `rpent_tools.py`,
`rpent_simulator.py`, `rpent_launch.py`, `configs/rpent_qwen2b_openwam.yaml`,
local/mock/unit diagnostics and this documentation. Default `run_baseline.sh`
now uses RPent; `baseline/stop.py` also recognizes the dedicated planner process.
Historical actor, evaluator compatibility, common logging and cleanup are reused.
Old custom loop is archived at **42fdb1b** and never invoked by the new launcher.

Backend: **Transformers 5.5.4**, direct `Qwen3_5ForConditionalGeneration.generate`.
No HTTP/OpenAI client/API key/model server. The tested vLLM 0.20.1 offline API
created internal TCP listeners, so it could not meet the no-listening-port
constraint; it is only an archived diagnostic, not a selectable second backend.
See [interface audit](RPENT_DESIGN.md) for the source-based decision.

## Validation, including an interface bug found and fixed

- Six new adapter unit tests passed; six historical parser/chunk/runtime tests
  passed in their original environments. Shell syntax and Python compilation passed.
- Local real-model text reasoning and image description passed, with one weight
  load and successive inference calls on the same Python object.
- Initial local tool diagnostic incorrectly inherited model EOS 248044 only.
  The checkpoint has no generation_config.json and declares chat EOS 248046 in
  its tokenizer. This allowed continuation after a completed tool call into fake
  tool-response/user text. The adapter now respects both official EOS IDs.
  This was an **INTERFACE** bug, not a reasoning result, and no prompt was changed.
- With the EOS boundary corrected, actual Qwen + RPent mock completed **3 turns**:
  execute_subgoal → updated image → execute_subgoal → updated image → finish.
  RPent handled history, tool results and termination. No new TCP listeners.
- Mock Qwen finished after two checks although the fixture requested three.
  The three-turn interface gate passed, but following the mock instruction was
  imperfect. No fabricated model responses or task-specific solution was supplied.

Mock evidence: `runs/rpent_local_smoke_v4/`. Its three RPent requests used
**2298 input / 212 output tokens**, latencies **2.740 / 2.275 / 2.410 s**.
The preceding standalone text/image calls are excluded from those numbers.
Model load count was 1 across all five generations. Reasoning tokens are null.
An additional identical local-only audit, `runs/rpent_local_network_audit/`,
also passed with a Python audit hook rejecting AF_INET/AF_INET6 connect/bind.
No model HTTP/API use or new listeners occurred. Diagnostic call IDs include
standalone tests; RPent stats are the authoritative three-turn count. The
committed logger separates RPent turn IDs from standalone generation IDs.

## Real run result

Run: `runs/20260921T222051_rpent_fill_pen_holder_0/`

| Metric | Actual result |
|---|---|
| Multi-decision episode completed | **No** |
| Scene creation / reset / RGB | Passed |
| Three-component coexistence | Passed |
| Planner generations | 1 |
| Accepted RPent turns / tool calls | 0 / 0 |
| Input / output tokens | 952 / 6144 |
| Reasoning tokens / cached tokens | null / null |
| Planner mean / p50 / p95 latency | 178.638 / 178.638 / 178.638 s; n=1 |
| Time to valid decision | No valid decision; null |
| Generation cap hit / 180s timeout hit | Yes / No |
| Episode wall time, including scene shutdown | 180.761 s; excludes component startup |
| Environment steps / simulated time | 0 / 0 s |
| OpenWAM inferences / mean latency | 0 / N/A; weights loaded, no tool dispatched |
| Peak total GPU memory, sampled 1 Hz | 38121 MiB = **37.228 GiB** |
| CUDA peak allocation for Qwen | 4578014208 bytes ≈ 4.264 GiB; not total process residency |
| OOM | No |
| Model weight load count | 1 |
| Process cleanup | All three recorded PIDs exited; GPU returned to 136 MiB |
| Video | Valid 1920×480 MP4, **1 initial frame**; not a rollout video |
| Pipeline ready | **False** |

The model described the RGB and task, then repeatedly reconsidered whether to
pick up the holder or a pen. It exhausted 6144 tokens without `</think>` or any
`<tool_call>`. Final token was 3158, not either configured EOS. This is therefore
distinct from the fixed EOS transport bug.

Taxonomy: **PLANNER_DELIBERATION / TIMING**. The blocking observation is failure
to finalize a decision, not an observed physical execution failure. Physical
execution and multi-turn behavior on the real task remain **unmeasured**.
The simulator waits while planning, so this run does not demonstrate dynamic
physics failures caused by planner latency. Success=false at zero steps is not
a meaningful benchmark success-rate estimate.

Artifacts include exact tokenized input, raw model response, config, source
hashes, packages/commits, events, GPU samples, RPent result, stdout, final
simulator state summary, timeline and initial-frame video. actor/tools JSONL
files are intentionally empty because no action was dispatched. In this run's
raw failure record time_to_valid_decision is absent; interpret it as null, not
zero. Logging now initializes that field explicitly for subsequent runs.

No further task episode was run or configuration optimized after this failure.

## Versions / reproducibility / limitations

| Component | Version / revision |
|---|---|
| RPent | 63c01fbf2ae9a6ab03bbc13eb9abf35853a71a9f |
| RoboDojo | 726e9aabfaa642203722eb126f5eaf0f37f3e1ad + pre-existing runtime patch |
| XPolicyLab / vendored OpenWAM adapter | bb9a0b5f5136a74503b679af830bfd0a3a837d5c |
| Qwen checkpoint | 15852e8c16360a2fea060d615a32b45270f8a8fc |
| OpenWAM checkpoint | 2c1302294e3ba8319bbdb2c803b7a27de9292d03 |
| Planner runtime | Python 3.11.16; Torch 2.11.0 + CUDA 13.0; Transformers 5.5.4 |
| RPent local integration | PydanticAI 2.46.0; Accelerate 1.15.0 |

Config remains reasoning ON, 8192 context, 6144 output maximum with actual-input
margin, 180s guard, 64-step window, native 32-step actor replan, 8 RPent turns,
512 environment steps. Production prompt only substitutes RPent tool protocol
for the historical JSON protocol; no task demonstrations or optimized plan.

```bash
cd /root/gpufree-data/robot-agent-exp
./run_baseline.sh --task fill_pen_holder --seed 0
./stop.sh  # only this project's recorded process groups
```

This reproduces an attempted baseline, not a claim that the blocker is solved.
The launcher now exits nonzero when pipeline readiness is false. The measured
run preceded that exit-status-only logging correction and returned shell 0;
its summary correctly records completed=false and pipeline_ready=false.

Existing Isaac USD/material/teardown warnings remain; assets were not modified.
Transformers uses its CUDA Torch fallback for missing optional fast linear
attention kernels; it does not offload the model to CPU. The lightweight
rpent-qwen overlay has unused upstream CLI/backend dependency metadata warnings
described in RPENT_DESIGN.md; the original installed environments are unchanged.

**Blocker:** Qwen fails to finalize the first real task decision under the fixed
baseline budget. A real multi-turn episode has not been demonstrated. No READY
claim is made, and no 8-task sweep was started.
