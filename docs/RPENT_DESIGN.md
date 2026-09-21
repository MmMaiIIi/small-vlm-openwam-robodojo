# RPent local-Qwen baseline interface audit

Only `fill_pen_holder`, seed 0 is in scope. No task examples, plan replay,
memory, verifier, training, or sweep is introduced.

## Reused upstream boundaries

- RPent `63c01fbf2ae9a6ab03bbc13eb9abf35853a71a9f`:
  `ApiAgentLoop.solve/_solve`, `_ApiRunObserver`, PydanticAI history,
  `_prune_history_images`, `Toolkit.add_tool/execute_tool`, `ToolResult`,
  `_make_tool_function`, image feedback, `EnvState`, finish/max-turn handling.
- Only the model transport is replaced by `LocalQwenPlanner`, a PydanticAI
  `Model` subclass. The upstream class happens to be named `ApiAgentLoop`,
  but an API provider/client is not used. The agent factory excludes RPent's
  default filesystem `read_image` tool to expose exactly three robot tools.
- `RobotToolkit` disables default file/memory tools and registers
  `execute_subgoal`, `wait`, `finish`. No `MemoryManager` or robot skill set.
- RPent has no existing RoboDojo/OpenWAM integration in this checkout.
  The isolated Isaac worker accepts finite operations over an anonymous
  AF_UNIX pipe. This is robot IPC, never model transport.
- Actor remains the previously verified `RecordedModel` wrapper over the
  official XPolicyLab model and WebSocket server. No new inference algorithm.
- RoboDojo `726e9aabfaa642203722eb126f5eaf0f37f3e1ad`, XPolicyLab
  `bb9a0b5f5136a74503b679af830bfd0a3a837d5c`. Pre-existing runtime patches
  retained, no upstream files changed by this integration.

## Qwen backend decision

The installed vLLM 0.20.1 offline `LLM.chat` was actually tested, including
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. Text reasoning worked (18 input / 152 output
tokens), but internal distributed initialization introduced listening TCP
endpoints. This violates the requested *no new listening ports* constraint,
even though it is not an HTTP/OpenAI inference server.

Production uses **only Transformers 5.5.4**, `Qwen3_5ForConditionalGeneration`,
`AutoProcessor`, local checkpoint, BF16 CUDA, SDPA, no quantization/offload.
The processor/generation interface follows installed source and
[official Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/v5.5.0/model_doc/qwen3_5).
The one-off vLLM probe is not selectable as a production backend.

The model is loaded once and called directly from RPent's Python process.
No remote image URLs, API credentials, HTTP clients, servers, or LiteLLM.
The checkpoint's own tool template is used. Adapter parsing converts exactly
one complete native XML tool call into RPent's `ToolCallPart`; JSON schema
validates arguments. It never executes partially parsed output.

Reasoning stays enabled. Maximum generation 6144; context 8192, with a
32-token safety margin after actual input tokenization. No history truncation
is added by this adapter; RPent's image pruning remains unchanged. Context
exhaustion, generation exhaustion and a 180-second guard are failures, not
reasons to disable thinking. Unknown reasoning/cache token counts are null.

Transport correction found during validation: this checkpoint lacks
`generation_config.json`; model fallback EOS is endoftext (248044), while
tokenizer chat EOS is im_end (248046). Local generation explicitly respects
both official token IDs. Without chat EOS, the initial tool test continued
past its completed call into a fictitious user/tool response and repeated
text until 6144 tokens. That diagnostic is an **INTERFACE** failure, not proof
of failed reasoning. Recognizing official end-of-turn is not early stopping
reasoning or a prompt/config optimization.

## Robot execution / information boundary

The native actor replan is 32 actions; 25 Hz control and two chunks yield a
64-step (2.56 simulated second) tool window. RPent max turns 8; max steps 512.
Each execute sets the subgoal, resets native cache, validates generation and
instruction acknowledgements, accounts executed/discarded actions and drops
client-owned chunks before returning observation. WAIT holds native absolute
EE targets. Official reset, reward initialization, take_action, and evaluator
are retained. No task/physics/model semantics are changed.

Only task instruction, RGB and explicit execution metadata reach RPent.
Robot state used by OpenWAM, reward and task success stay inside the Isaac
worker; final evaluator labels go to the final report only. Camera panel
layout matches the prior baseline. Prompt changes are only the required
RPent tool names/protocol, not task-specific optimization.

## Environment and historical reference

`conda-envs/rpent-qwen` is a 62 MB (before optional future additions) Python
venv overlay inheriting existing `conda-envs/qwen`. It adds editable RPent,
PydanticAI 2.46.0, imageio and Accelerate 1.15.0, without copying Torch/weights
or modifying the original environments. Only the planner/tool core is used;
unused Claude/Codex/MCP backend dependency metadata is not a claim that those
backends are configured in this overlay.
`pip check` therefore reports absent claude-agent-sdk/openai-codex/scipy and
the inherited MCP 2.2 versus RPent's <2 requirement. These optional execution
paths are not imported/used by this local integration; the original complete
RPent environment is unchanged. This overlay is not advertised as a complete
replacement for the upstream CLI/backend environment.

Old custom loop is archived at commit `42fdb1b`. Its source/reports remain
historical; the new default launcher never calls `planner.py`, `launch.py`, or
`simulator.py`. Only independent actor, runtime compatibility, logging and
process-cleanup helpers are reused. No RPent upstream patch is needed.

## Validation gates

1. `tests/test_rpent_adapter.py`: parsing, schemas, image conversion, exactly
   three tools, inherited RPent loop (CPU).
2. `tests/local_qwen_smoke.py RUN_DIR`: real local text/image/model persistence,
   then real Qwen + RPent + mock tool for three observation checks and finish.
   The generic mock protocol is not a task solution or replayed model answer.
3. Only after gate 2 passes, `./run_baseline.sh --task fill_pen_holder --seed 0`.

Unexecuted code is not evidence of a working robotic pipeline. Actual gate
outcomes and blockers are recorded separately in the result report.
