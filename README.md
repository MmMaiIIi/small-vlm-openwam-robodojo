# Small VLM + OpenWAM on RoboDojo

Research question:
Can a small reasoning VLM paired with a stronger low-level WAM provide an efficient long-horizon robotic agent?

Current baseline:

```text
Qwen3.5-2B (local Python, reasoning ON)
↓
RPent planner / toolkit / history / tool loop
↓
OpenWAM-Alpha-Sim-RoboDojo
↓
RoboDojo Long-Horizon
```

The default baseline now uses **RPent + local Qwen3.5-2B (Transformers)**.
RPent owns the planner/tool/history loop; Qwen runs directly in the same Python
process, without an HTTP model server or OpenAI client. Runtime results are
recorded separately under `runs/`; source presence alone does not establish readiness.

Goals:

- long-horizon success
- lower planner latency
- lower inference cost
- robust low-level physical execution

Benchmark:

RoboDojo Sim Long-Horizon

## Run

```bash
cd /root/gpufree-data/robot-agent-exp
./run_baseline.sh --task fill_pen_holder --seed 0
```

The launcher uses existing checkpoints, loads local Qwen once, starts the official
OpenWAM actor service and native RoboDojo worker, and cleans up its own
process groups. It never installs packages or downloads models/assets. `./stop.sh`
stops this project's active run; `./stop.sh --run /absolute/path/to/run` targets an
explicit run. A lock prevents overlapping launches from this project.

Config: `configs/rpent_qwen2b_openwam.yaml`. Design: [RPent audit](docs/RPENT_DESIGN.md).
Actual outcome and blocker: [RPent baseline report](docs/RPENT_BASELINE_REPORT.md).
New adapter unit tests use `conda-envs/rpent-qwen/bin/python -m unittest discover
-s tests -p test_rpent_adapter.py -v`. Historical actor/chunk tests use the OpenWAM
environment; the two environments intentionally have different dependencies.

Each run includes its config, source/package versions, raw planner responses, actor
chunk accounting, events, 1 Hz GPU samples, three-camera video, stdout, summary,
and a readable timeline. Planner calls use RGB, the original instruction,
RPent history and non-privileged tool feedback. Success labels remain evaluator-only.

Default budget is 8 RPent turns and 512 control steps. This is a raw bounded
interface baseline, not a full-horizon benchmark or a statistically meaningful
success-rate evaluation. A complete bounded episode can legitimately fail the task.

Historical custom-loop result: [high-budget retry report](docs/BASELINE_HIGH_BUDGET_REPORT.md).
With the authorized 6144-token / 180-second planner budget, two valid decisions
executed 128 steps, then the third call exhausted 6144 tokens without finalizing.
Natural episode completion is not verified; no other task was launched.
The [original 1536-token report](docs/BASELINE_REPORT.md) is preserved.

The old controller is preserved at commit `42fdb1b` and is not used by the default
launcher. No RPent task skills, memory system, learned verifier, adaptive planning,
training, or upstream research-logic changes are included.
