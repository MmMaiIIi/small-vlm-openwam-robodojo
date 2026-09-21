# Reproducibility versions

Recorded from the installed environment on 2026-09-21. Upstream source trees and model weights are referenced only; they are not copied into this repository.

| Component | Version / revision | Reference |
| --- | --- | --- |
| RoboDojo | `726e9aabfaa642203722eb126f5eaf0f37f3e1ad` | `https://github.com/RoboDojo-Benchmark/RoboDojo.git` |
| XPolicyLab | `bb9a0b5f5136a74503b679af830bfd0a3a837d5c` | `https://github.com/XPolicyLab/XPolicyLab.git` |
| OpenWAM | `0.1.0` at XPolicyLab commit `bb9a0b5f5136a74503b679af830bfd0a3a837d5c` | XPolicyLab `policy/OpenWAM/OpenWAM` |
| RPent | `63c01fbf2ae9a6ab03bbc13eb9abf35853a71a9f` (`0.0.0`) | `https://github.com/RLinf/RPent.git` |
| Isaac Sim | `5.1.0.0` | Existing `/opt/conda/envs/isaaclab` installation |
| Isaac Lab | `2.3.2.post1` | Existing `/opt/conda/envs/isaaclab` installation |
| OpenWAM checkpoint | Hugging Face revision `2c1302294e3ba8319bbdb2c803b7a27de9292d03` | `OpenWAM-Alpha-Sim-RoboDojo` |
| Qwen checkpoint | Hugging Face revision `15852e8c16360a2fea060d615a32b45270f8a8fc` | `Qwen/Qwen3.5-2B` |
| vLLM | `0.20.1` | Qwen serving environment |

## Local upstream state

The RoboDojo working tree had a local runtime compatibility modification at capture time in:

- `env/scene_manager/objects/articulation.py`
- `env/scene_manager/scene_manager.py`

It addresses Isaac articulation DOF-limit conversion and teardown ordering. The upstream source tree is not included here. RPent had no local modifications. Baseline implementation has not appeared on disk yet; this repository is a scaffold waiting for baseline code.
