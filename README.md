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

The component environments and checkpoints have been installed and smoke-tested independently. The hierarchical harness, planner integration, robot adapter, and formal evaluation have not appeared on disk yet and are not included in this initial scaffold.

Goals:

- long-horizon success
- lower planner latency
- lower inference cost
- robust low-level physical execution

Benchmark:

RoboDojo Sim Long-Horizon

Work in progress.
