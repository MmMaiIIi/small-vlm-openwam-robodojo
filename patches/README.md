# Upstream changes

This baseline introduces no upstream source changes. All research code lives here.

The environment already had an authorized reset/cleanup compatibility patch recorded at
`/root/gpufree-data/RoboDojo/.local_patches/fill_egg_holder_reset_compat.patch`.
It predates this baseline and is preserved. Each run records RoboDojo's dirty status and
diff hash alongside its commit so that this prerequisite is not hidden by the commit ID.

## Research-run-local evaluator compatibility view

The first real evaluation reset failed in official
`env/reward_manager/func_parser.py:45`, `np.concatenate([pos, rot])`.
Full traceback: `runs/20260921T210938_fill_pen_holder_0/events.jsonl` / `stdout.log`.
Scene-only deployment smoke did not call the EvalEnv reward initialization.

`LayoutManager.get_instance_pose` returns CUDA tensors for rigid/articulated objects;
Func_Parser uses these in NumPy, transforms3d and Shapely consumers (53 call sites).
Rather than change the layout API for physics/other clients or edit many reward
functions, `baseline/runtime_compat.py` gives **only the current Func_Parser instance**
a delegating view whose pose return values use `detach().cpu().numpy()` when Tensor.
Other returns, arguments, methods, frame conventions, dtype, pose values, reward
predicates, thresholds, and task definitions are unchanged. The first conversion's
actual source/target types, devices and shapes are logged in `events.jsonl`.

This is not a global Torch/NumPy monkey patch and does not expose evaluator state to
the planner. No upstream file is modified; the compatibility view's complete source
is included in every run snapshot. Versions remain Isaac Sim 5.1.0.0, Isaac Lab
2.3.2.post1, runtime Torch 2.7.0+cu128, NumPy 1.26.0, RoboDojo 726e9aa.
