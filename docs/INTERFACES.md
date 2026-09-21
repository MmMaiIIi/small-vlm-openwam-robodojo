# Source-based interface audit

RoboDojo 726e9aabfaa642203722eb126f5eaf0f37f3e1ad and XPolicyLab
bb9a0b5f5136a74503b679af830bfd0a3a837d5c were inspected before implementation.
Each run records exact commits, dirty status, diff hashes, installed packages, and research source hashes.

1. Official `src/eval_client/eval_env.py` owns environment reset, controller interpolation,
   IK, reward updates, and success checks. We instantiate `create_eval_env`, initialize
   `run_reward` / `get_score` / interactive support exactly as `run_eval` does, and use
   `get_obs` / `take_action`. No task definition or robot-coordinate conversion is replaced.
2. Observations contain `vision`, `state`, `instruction`, and metadata. RGB cameras are
   `cam_head`, `cam_left_wrist`, `cam_right_wrist`, normally 480x640x3 uint8.
   The original instruction comes from the official observation description manager.
3. Planner input is an RGB collage (three labelled 384x288 panels), original instruction,
   and last three *attempted* model decisions. No poses, reward, success label or object
   metadata reach Qwen. Progress text does not assert any attempted action succeeded.
4. Official `XPolicyLab/policy/OpenWAM/model.py` reads `obs['instruction']`, formats its
   prompt, encodes all three images and native robot state, and performs official
   world-to-base / EEF20 conversions. Those transformations are reused unchanged.
5. The adapter caches encoded observations in `_batch` / `_order`, not future actions.
   Its `reset()` clears those buffers. The synchronous client owns the returned action
   list. At each EXECUTE we clear observations and start a generation-tagged subgoal;
   all remaining client actions are discarded/accounted before another decision.
6. Official action output has 32 steps; configured actor `replan_steps=32`. Actual
   predicted length is measured on the engine result, not inferred from config.
   Control collection frequency is 25 Hz (physics dt .004, ten physics steps/control).
   Two replans form one 64-step high-level window: 2.56 simulated seconds. The default
   cap is 512 control steps / 8 API calls (20.48 simulated seconds), intentionally a
   bounded interface baseline, not the official full-horizon success-rate benchmark.
   Official per-task limits remain unchanged and can end an episode earlier.
7. Communication uses the official XPolicyLab msgpack WebSocket server/client.
   A research-only subclass adds generation acknowledgements and chunk accounting;
   it does not change model inference or its numeric outputs. A run-local engine
   observer records actual action count. No upstream files are patched.
8. WAIT holds the current observed absolute EE/gripper targets through the native
   controller for the same finite window. FINISH terminates, but evaluator success is
   still measured from official checks, never trusted from the planner declaration.
9. Qwen uses installed vLLM, reasoning enabled, JSON schema output, 8192 context,
   6144 generated-token cap, 180-second request timeout (user-authorized retry;
   the original run used 1536 / 90). Invalid JSON retries once;
   timeout or token exhaustion terminates with `planner_timeout`, no model fallback.
10. RPent planner/toolkit/API sources were inspected. Its general agent loop is not
    needed for this minimal direct local-HTTP baseline; RPent remains installed but
    is not in the executed path. No RPent skill, memory, robot adapter, or agent loop
    is modified or added.

## Metrics caveats

Success is an evaluator-only episode label; N=1 per task is not a success-rate estimate.
Wall clock includes synchronous reasoning/inference waits while simulation is paused.
Simulation time excludes reset/settling and equals executed control steps / 25 Hz.
GPU process and total peaks are sampled every second (not instantaneous hardware maxima).
Reasoning/cache token counts are null when not reported by the API. Output-token totals
include reasoning when vLLM includes it in completion usage. Local GPU-seconds, tokens,
and wall time are cost proxies, not invented dollar prices.

All task failures default to UNKNOWN unless the logs support a more specific attribution.
Subgoal quality and physical execution must be reviewed against the saved images/video;
task failure alone cannot distinguish HIGH_LEVEL_REASONING from PHYSICAL_EXECUTION.
