# SIM_TO_REAL Deployment Bug Log

Bugs found and fixed during H100 deployment (March 23-24, 2026).
All fixes applied to local files AND deployed to H100.

---

## Bug S2R-1: TerrainGeneratorCfg duplicate num_cols

**File:** `configs/terrain_cfgs.py`
**Error:** `TypeError: got multiple values for keyword argument 'num_cols'`
**Cause:** `DISTILLATION_TERRAINS_CFG` passed `num_cols=40` alongside `**_COMMON` which already had `num_cols=20`.
**Fix:** Created `_DISTILL_COMMON = {**_COMMON, "num_cols": 40}` to override before unpacking.

---

## Bug S2R-2: RslRlVecEnvWrapper clip_actions incompatibility

**File:** `scripts/train_expert.py`, `scripts/train_distill_s2r.py`
**Error:** `ValueError: Box high must be a np.ndarray, integer, or float, actual type=<class 'bool'>`
**Cause:** H100's gymnasium version rejects `clip_actions=True` in `RslRlVecEnvWrapper` — the `high` bound for the action space Box gets set to a boolean.
**Fix:** Wrapped in try/except: `try: RslRlVecEnvWrapper(env, clip_actions=True) except (TypeError, ValueError): RslRlVecEnvWrapper(env)`

---

## Bug S2R-3: ObservationDelayWrapper TensorDict shape

**File:** `wrappers/observation_delay.py`
**Error:** `IndexError: tuple index out of range` at `obs_tensor.shape[1]`
**Cause:** H100's RSL-RL returns observations as TensorDict `{"policy": tensor(N, 235)}`. The wrapper extracted the tensor correctly but assumed `shape[1]` would always work — it can fail if the tensor is 1D.
**Fix:** Used `obs_tensor.shape[-1]` for obs_dim and `obs_tensor.shape[0] if obs_tensor.dim() > 1` for num_envs. Also changed dict detection to `hasattr(obs_dict, 'keys')` to handle TensorDict.

---

## Bug S2R-4: OnPolicyRunner.set_log_dir not available

**File:** `scripts/train_expert.py`, `scripts/train_distill_s2r.py`
**Error:** `AttributeError: 'OnPolicyRunner' object has no attribute 'set_log_dir'`
**Cause:** H100's RSL-RL version doesn't have the `set_log_dir()` method. The log directory must be passed directly to the constructor.
**Fix:** Changed `OnPolicyRunner(env, cfg, log_dir=None)` + `runner.set_log_dir(path)` to `OnPolicyRunner(env, cfg, log_dir=path)` with `os.makedirs(path, exist_ok=True)` before creation.

---

## Bug S2R-5: Import path mismatch on H100

**File:** `configs/base_s2r_env_cfg.py`, `scripts/train_expert.py`, `scripts/train_distill_s2r.py`
**Error:** Various import failures for `quadruped_locomotion`
**Cause:** H100 has `quadruped_locomotion` pip-installed from `~/multi_robot_training_new/source/quadruped_locomotion/`, not from the relative path `../../multi_robot_training/source/quadruped_locomotion/`.
**Fix:** Added multiple candidate paths tried in order: relative local path, nested H100 path, and `~/multi_robot_training_new/` path. `os.path.isdir()` check skips nonexistent paths.

---

## Bug S2R-6: From-scratch training collapse (standing still exploit)

**Error:** Terrain level declined from 3.2 to 0.07 across all from-scratch runs
**Cause:** S2R wrappers at full intensity (40ms delay + 5% dropout) from step 0 made learning to walk from random initialization impossible. Policy discovered standing still avoids all penalties.
**Fix:** Two changes:
1. Fine-tune from `hybrid_nocoach_19999.pt` (proven walking policy) instead of from scratch
2. Replace static wrappers with `ProgressiveS2RWrapper` — S2R scales from 0% (terrain row 0-2) to 100% (row 6+)

---

## Bug S2R-7: CUDA OOM from zombie processes

**Error:** `torch.OutOfMemoryError: CUDA out of memory` — 93 GB GPU fully consumed
**Cause:** Previous training sessions left zombie Isaac Sim processes holding ~16 GB VRAM each. Killing screen sessions doesn't release GPU memory held by D-state zombie processes.
**Fix:** BMC force restart via Redfish API to fully reboot the server and clear all GPU state.

---

## Bug S2R-8: Aggressive env-level DR causes 100% body_contact termination

**Error:** `body_contact: 1.0000` — every episode ends in a fall, terrain stuck at 0.0
**Cause:** `base_s2r_env_cfg.py` used widened DR (friction 0.15-1.3, mass ±5 kg, pushes ±3 N) that the hybrid_nocoach baseline never trained with. Low friction (0.15) caused instant slipping and body contact on the first step.
**Fix:** Reverted all DR params to Mason's proven values (friction 0.3-1.0, mass ±2.5, pushes disabled). S2R robustness now comes exclusively from the `ProgressiveS2RWrapper` (sensor noise, dropout, delay) which scales safely with terrain curriculum.
**Lesson:** Physics DR and observation-level S2R are separate concerns. Physics DR must match the base checkpoint's training distribution. Observation-level S2R can be added progressively.

---

## Bug S2R-9: Hard body_contact termination kills exploration after actor unfreeze

**Error:** `body_contact: 1.0000` after critic warmup completes (iter 300+), terrain drops to 0.0
**Cause:** Mason's `HybridTerminationsCfg` has `body_contact` as a hard termination — any body touch ends the episode instantly. When the actor unfreezes at iter 300 and starts exploring with new reward weights (foot_clearance 2.0, joint_pos -0.3), the first exploratory motion causes a body touch → episode killed → no learning signal → permanent collapse.
**Fix:** Created `S2RTerminationsCfg` that removes `body_contact` termination entirely. Added `undesired_contacts` reward penalty (weight -1.5) with same sensor config — the robot is discouraged from body contact but episodes continue, allowing it to learn from mistakes.
**Lesson:** Hard termination on body contact is incompatible with reward weight fine-tuning. The actor needs room to explore suboptimal motions and learn from them. Soft penalties > hard kills for training.

---

## Bug S2R-11: lr=3e-5 too aggressive for fine-tuning — gait collapses after unfreeze

**Error:** Terrain level drops from 5.0 to 0.0 immediately after 300-iter critic warmup ends
**Cause:** lr=3e-5 was used for previous Mason training (from scratch with adaptive KL). For actor-only resume fine-tuning, the first gradient updates at 3e-5 are too large — they change the actor weights enough to destroy the learned gait in 1-2 iterations.
**Fix:** Two changes:
1. Reduced lr_max from 3e-5 to **1e-5** (3x more conservative)
2. Added **gradual actor layer unfreeze**: instead of unfreezing all actor layers at iter 300, unfreeze one layer at a time (output layer at 300, middle at 500, all at 700). This gives each layer a stable anchor from the still-frozen layers while it adapts.
**Result:** Terrain held at 5.2 (friction) and **climbed to 5.7** (stairs) through all 3 unfreeze phases. Zero collapse.

---

## Bug S2R-10: Parallel stair training causes PhysX GPU scene corruption

**Error:** `"Scene state is corrupted. Simulation cannot continue!"` + `GPU compressContactStage1 fail to launch kernel`
**Cause:** Stair terrain generates far more contact points per env than flat/friction (feet hitting stair edges at multiple points). Running stairs alongside friction in parallel overwhelmed the PhysX GPU collision stack, corrupting the scene state.
**Impact:** Stairs crashed first with scene corruption. The crash left zombie processes holding ~15 GB VRAM each. Friction then OOM'd because zombies consumed all remaining GPU memory.
**Fix:** BMC restart to clear GPU state. Both experts run in parallel on a CLEAN GPU (no prior zombie processes). The collision stack (2^31) is sufficient when starting from a clean state — the corruption was caused by cascading failures from a dirty GPU, not insufficient stack size.
**Lesson:** Always BMC reboot before launching parallel trainings. Stale GPU state from prior crashes can cause phantom OOM and collision failures even with plenty of nominal VRAM.

---

## Deployment Status (as of 2026-03-24 12:08)

| Expert | Status | Notes |
|--------|--------|-------|
| friction | TRAINING | Started 12:06, solo first then stairs joined, terrain 3.51 warmup, ETA ~15h |
| stairs_up | TRAINING | Started 12:08, parallel with friction on clean GPU, terrain 3.51 warmup, ETA ~21h |
| stairs_down | PENDING | After stairs_up completes |
| boulders | PENDING | |
| slopes | PENDING | |
| mixed_rough | PENDING | |
| distillation | PENDING | Needs all 6 expert checkpoints |

## Key Lessons Learned

1. **Never apply full S2R from step 0** — progressive ramp with terrain curriculum
2. **Never train from scratch with S2R** — fine-tune from a walking policy
3. **Always BMC reboot before launching training** — stale GPU state causes phantom crashes
4. **Use lr_max=1e-5 for fine-tuning** (3e-5 destroyed gait, 1e-3 is from-scratch only)
5a. **Gradual actor layer unfreeze** — output layer first (iter 300), middle (500), all (700)
5. **Soft body contact penalty, not hard termination** — hard kill prevents exploration after reward weight changes
6. **Keep physics DR at Mason's safe values** — wider DR (friction 0.15) causes instant falls for Mason-trained policies
7. **Parallel stair training is OK on a clean GPU** — the PhysX corruption was from dirty state, not inherent to parallel runs

---

*10 bugs found and fixed. Training running on clean GPU with progressive S2R from hybrid_nocoach_19999.*
*TensorBoard: http://172.24.254.24:6007*
