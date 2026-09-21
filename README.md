# Small VLM + OpenWAM on RoboDojo

Research question:
Can a small reasoning VLM paired with a stronger low-level WAM provide an efficient long-horizon robotic agent?

Current baseline:

```text
Qwen3.5-2B
↓
Minimal hierarchical harness
↓
OpenWAM-Alpha-Sim-RoboDojo
↓
RoboDojo Long-Horizon
```

The initial scaffold has now been extended with a minimal, synchronous hierarchical
baseline. Runtime results are recorded separately under `runs/`; source presence alone
does not establish pipeline readiness.

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

The launcher uses existing environments and checkpoints, starts local Qwen and
OpenWAM services, runs one bounded native RoboDojo episode, and cleans up its own
process groups. It never installs packages or downloads models/assets. `./stop.sh`
stops this project's active run; `./stop.sh --run /absolute/path/to/run` targets an
explicit run. A lock prevents overlapping launches from this project.

Config: `configs/baseline.yaml`. Interface rationale: `docs/INTERFACES.md`.
Tests: `/root/gpufree-data/conda-envs/openwam/bin/python -m unittest discover -s tests -v`.
The chunk-contract test imports the actor server, so run the full suite in the actor
environment (Isaac intentionally has an older WebSocket client dependency).

Each run includes its config, source/package versions, raw planner responses, actor
chunk accounting, events, 1 Hz GPU samples, three-camera video, stdout, summary,
and a readable timeline. Planner calls use only current RGB, the original instruction,
and three recent model decisions. Success labels remain evaluator-only.

Default budget is 8 planner API calls and 512 control steps. This is a raw bounded
interface baseline, not a full-horizon benchmark or a statistically meaningful
success-rate evaluation. A complete bounded episode can legitimately fail the task.

Latest result: [high-budget retry report](docs/BASELINE_HIGH_BUDGET_REPORT.md).
With the authorized 6144-token / 180-second planner budget, two valid decisions
executed 128 steps, then the third call exhausted 6144 tokens without finalizing.
Natural episode completion is not verified; no other task was launched.
The [original 1536-token report](docs/BASELINE_REPORT.md) is preserved.

No RPent skills, memory system, learned verifier, adaptive planning, training, or
upstream research-logic patches are included. RPent core remains available but is
not required in this minimal execution path.
