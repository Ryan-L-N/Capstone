# H100 Deployment Bugs — NAV_ALEX Phase C

**Date:** 2026-03-18
**Server:** H100 NVL (ai2ct2), Isaac Lab 0.x, RSL-RL (pip)
**Issue:** NAV_ALEX was developed against a newer local Isaac Lab version. The H100 has an older version with different APIs. These are all API compatibility issues, not logic bugs.

---

## Bug N-1: Gym Registration Not Found

- **Error:** `TypeError: ManagerBasedRLEnv.__init__() missing 1 required positional argument: 'cfg'`
- **Root cause:** `gym.make(env_id, num_envs=N)` doesn't pass `cfg`. Isaac Lab requires `gym.make(env_id, cfg=env_cfg)`.
- **Fix:** Import config class directly and pass it:
  ```python
  from nav_locomotion.tasks.navigation.config.spot.nav_env_cfg import SpotNavExploreCfg
  env_cfg = SpotNavExploreCfg()
  env_cfg.scene.num_envs = args.num_envs
  env = gym.make(env_id, cfg=env_cfg)
  ```
- **File:** `scripts/rsl_rl/train_nav.py`

## Bug N-2: SPOT_CFG Import Path

- **Error:** `ModuleNotFoundError: No module named 'isaaclab_assets.robots.boston_dynamics'`
- **Root cause:** H100 Isaac Lab has `isaaclab_assets.robots.spot`, not `boston_dynamics`.
- **Fix:** Try/except fallback:
  ```python
  try:
      from isaaclab_assets.robots.boston_dynamics import SPOT_CFG
  except (ImportError, ModuleNotFoundError):
      from isaaclab_assets.robots.spot import SPOT_CFG
  ```
- **File:** `tasks/navigation/config/spot/nav_env_cfg.py`

## Bug N-3: MeshWaveTerrainCfg Not Available

- **Error:** `ImportError: cannot import name 'MeshWaveTerrainCfg' from 'isaaclab.terrains.trimesh'`
- **Root cause:** H100 Isaac Lab doesn't have `MeshWaveTerrainCfg`. Wave terrain is `HfWaveTerrainCfg` (height-field based).
- **Fix:** Import from `isaaclab.terrains.height_field` instead:
  ```python
  from isaaclab.terrains.height_field import HfWaveTerrainCfg, HfDiscreteObstaclesTerrainCfg
  ```
  Also update HfDiscreteObstaclesTerrainCfg params (`obs_height_range` → `obstacle_height_range`, etc.)
- **File:** `tasks/navigation/mdp/terrains.py`

## Bug N-4: mdp/__init__.py Premature Import

- **Error:** `ModuleNotFoundError: No module named 'pxr'` (chained from `terrains.py` import)
- **Root cause:** `mdp/__init__.py` imported terrains at module load time, which chains into Isaac Lab → pxr (only available after AppLauncher).
- **Fix:** Removed eager imports from `mdp/__init__.py`. Terrains are imported lazily via `nav_env_cfg.py` (which runs after AppLauncher).
- **File:** `tasks/navigation/mdp/__init__.py`

## Bug N-5: terrain_out_of_bounds Not Available

- **Error:** `AttributeError: module 'isaaclab.envs.mdp' has no attribute 'terrain_out_of_bounds'`
- **Root cause:** This termination function doesn't exist in the H100's Isaac Lab version.
- **Fix:** Replaced with `bad_orientation` termination (limit_angle=1.5 radians).
- **File:** `tasks/navigation/config/spot/nav_env_cfg.py`

## Bug N-6: terrain_levels_vel Curriculum Not Available

- **Error:** Would crash at `mdp.terrain_levels_vel` reference.
- **Root cause:** No terrain-level curriculum functions in this Isaac Lab's `mdp` module.
- **Fix:** Replaced `NavCurriculumCfg` with empty `pass` body. Terrain curriculum promotion is handled internally by `TerrainGeneratorCfg(curriculum=True)` based on episode survival.
- **File:** `tasks/navigation/config/spot/nav_env_cfg.py`

## Bug N-7: ManagerBasedRLEnvCfg.SimCfg Not Available

- **Error:** `AttributeError: type object 'ManagerBasedRLEnvCfg' has no attribute 'SimCfg'`
- **Root cause:** H100 uses `SimulationCfg` from `isaaclab.sim`, not a nested `SimCfg` class.
- **Fix:** `from isaaclab.sim import SimulationCfg` and use `sim = SimulationCfg(dt=0.002, render_interval=50)`.
- **File:** `tasks/navigation/config/spot/nav_env_cfg.py`

## Bug N-8: RSL-RL OnPolicyRunnerCfg Import

- **Error:** `ImportError: cannot import name 'OnPolicyRunnerCfg' from 'rsl_rl.runners'`
- **Root cause:** H100 uses Isaac Lab's wrapper classes in `isaaclab_rl.rsl_rl`, not raw RSL-RL config classes.
- **Fix:** Import from `isaaclab_rl.rsl_rl`:
  ```python
  from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
  ```
  With try/except fallback for newer versions.
- **File:** `tasks/navigation/config/spot/agents/rsl_rl_ppo_cfg.py`

## Bug N-9: MeshRepeatedBoxesTerrainCfg.ObjectCfg API

- **Error:** `TypeError: ObjectCfg.__init__() got an unexpected keyword argument 'height_range'`
- **Root cause:** H100 `ObjectCfg` uses `height: float` and `size: tuple[float, float]`, not `height_range` and `size_range`.
- **Fix:** Changed to scalar `height` + tuple `size`:
  ```python
  ObjectCfg(num_objects=10, height=0.05, size=(0.5, 0.5))
  ```
- **File:** `tasks/navigation/mdp/terrains.py`

## Bug N-10: Terrain Patch Must Be Square

- **Error:** `ValueError: The terrain must be square. Received size: (50.0, 20.0).`
- **Root cause:** `MeshRandomGridTerrainCfg` requires square terrain patches.
- **Fix:** Changed from `(50, 20)` to `(20, 20)`.
- **File:** `tasks/navigation/mdp/terrains.py`

## Bug N-11: Zero Border Width in Random Grid Terrain

- **Error:** `RuntimeError: Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'.`
- **Root cause:** `border_width = size - int(size/grid_width) * grid_width`. With size=20 and grid_width=0.4: `int(20/0.4)=50`, `50*0.4=20.0`, border=0.
- **Fix:** Changed boulders `grid_width` from 0.4 to 0.45. `int(20/0.45)=44`, `44*0.45=19.8`, border=0.2 > 0.
- **File:** `tasks/navigation/mdp/terrains.py`

## Bug N-12: Joint Default Positions Out of Limits

- **Error:** `ValueError: The following joints have default positions out of the limits: 'fl_kn': 0.000 not in [-2.793, -0.247]`
- **Root cause:** `joint_pos={".*": 0.0}` sets all joints to 0, but knee joints have range [-2.793, -0.247]. Zero is outside this range.
- **Fix:** Set per-joint defaults matching training:
  ```python
  joint_pos={".*_hx": 0.0, "f.*_hy": 0.9, "h.*_hy": 1.1, ".*_kn": -1.5}
  ```
- **File:** `tasks/navigation/config/spot/nav_env_cfg.py`

## Bug N-13: Depth Image Not Flattened + Wrong Action Dim

- **Error:** `RuntimeError: Unable to concatenate observation terms in group 'policy'. The shapes of the terms are: [(64, 64, 1), (3,), (3,), (3,), (12,)]`
- **Root cause:** Two issues:
  1. `depth_image_obs` returns (64, 64, 1) — needs flattening for 4D tensors (was only handling 3D).
  2. `mdp.last_action` returns (12,) (joint positions from the underlying env), not (3,) (nav velocity commands).
- **Fix:**
  1. Changed `if depth_normalized.dim() == 3` to `if depth_normalized.dim() >= 3` for 4D case.
  2. Created `nav_prev_action()` returning zeros(N, 3) as placeholder — proper tracking deferred to NavEnvWrapper.
- **Files:** `tasks/navigation/mdp/observations.py`, `tasks/navigation/config/spot/nav_env_cfg.py`

## Bug N-14: Gym OrderEnforcing Wrapper Blocks Isaac Lab Access

- **Error:** `AttributeError: 'OrderEnforcing' object has no attribute 'device'` (and `num_envs`, `observation_manager`, `scene`, `episode_length_buf`, etc.)
- **Root cause:** `gym.make()` wraps the env in Gymnasium wrappers (`OrderEnforcing`, etc.) that don't expose Isaac Lab attributes.
- **Fix:** Added `self._unwrapped = env.unwrapped` in NavEnvWrapper and replaced ALL `self.env.scene`, `self.env.observation_manager`, etc. with `self._unwrapped.scene`, `self._unwrapped.observation_manager`, etc. Also added `episode_length_buf` setter for RSL-RL compatibility.
- **Files:** `modules/nav_env_wrapper.py`, `scripts/rsl_rl/train_nav.py`

## Bug N-15: RSL-RL Config Must Be Dict, Not Dataclass

- **Error:** `TypeError: 'SpotNavPPORunnerCfg' object is not subscriptable`
- **Root cause:** H100's RSL-RL `OnPolicyRunner` expects a dict config, not the `@configclass` dataclass.
- **Fix:** Used `from isaaclab.utils import class_to_dict` to convert before passing to runner.
- **File:** `scripts/rsl_rl/train_nav.py`

## Bug N-16: ActorCriticCNN Not in RSL-RL Namespace

- **Error:** `NameError: name 'ActorCriticCNN' is not defined`
- **Root cause:** RSL-RL uses `eval(class_name)` to instantiate the policy class. `ActorCriticCNN` isn't in its module scope.
- **Fix:** Injected into the runner module: `rsl_rl.runners.on_policy_runner.ActorCriticCNN = ActorCriticCNN`
- **File:** `scripts/rsl_rl/train_nav.py`

## Bug N-17: RSL-RL ActorCritic Constructor Signature Mismatch

- **Error:** `RuntimeError: Calculated padded input size per channel: (3 x 3). Kernel size: (5 x 5)`
- **Root cause:** RSL-RL passes `(obs_tensordict, obs_groups, num_actions, **policy_cfg)` positionally. Our CNN had `(num_obs, num_actions, depth_res, ...)`, so `num_actions=3` went to `depth_res=3` → 3x3 depth image.
- **Fix:** Changed constructor signature to match RSL-RL: `(obs, obs_groups=None, num_actions=3, ...)`. Extract `num_obs` from the obs TensorDict. Added `int()` casts for all dim parameters.
- **File:** `modules/depth_cnn.py`

## Bug N-18: RSL-RL Interface Methods Missing

- **Error:** `AttributeError: 'ActorCriticCNN' object has no attribute 'is_recurrent'`
- **Root cause:** H100's RSL-RL PPO calls `policy.is_recurrent`, `policy.action_mean`, `policy.action_std`, `policy.get_actions_log_prob()`, `policy.update_normalization()` — our CNN didn't have these.
- **Fix:** Added all required properties and methods: `is_recurrent=False`, `action_mean/std` from distribution, `get_actions_log_prob()`, `update_normalization()` (no-op), `get_hidden_states()` (None). Also changed `act()` to return just actions (RSL-RL calls separate methods for log_prob/value).
- **File:** `modules/depth_cnn.py`

## Bug N-19: get_observations() Must Return TensorDict-like Object

- **Error:** `AttributeError: 'dict' object has no attribute 'to'`
- **Root cause:** RSL-RL calls `obs.to(device)` on the result. A plain dict doesn't support `.to()`.
- **Fix:** Created `_ObsDict(dict)` subclass with a `.to()` method that moves all tensor values.
- **File:** `modules/nav_env_wrapper.py`

## Bug N-20: step() Must Return Same Format as get_observations()

- **Error:** `ValueError: dictionary update sequence element #0 has length 4108; 2 is required`
- **Root cause:** `step()` returned a raw tensor, but RSL-RL's storage expects TensorDict format matching `get_observations()`.
- **Fix:** Wrapped step() return in `_ObsDict({"policy": nav_obs})`.
- **File:** `modules/nav_env_wrapper.py`

## Bug N-21: projected_gravity → projected_gravity_b

- **Error:** `AttributeError: 'ArticulationData' object has no attribute 'projected_gravity'`
- **Root cause:** H100 Isaac Lab uses `projected_gravity_b` (body-frame), not `projected_gravity`.
- **Fix:** Changed to `robot.data.projected_gravity_b`.
- **File:** `modules/nav_env_wrapper.py`

## Bug N-22: Gym Requires reset() Before step()

- **Error:** `gymnasium.error.ResetNeeded: Cannot call env.step() before calling env.reset()`
- **Root cause:** RSL-RL's `learn()` calls `step()` without calling `reset()` first.
- **Fix:** Added `nav_env.reset()` before `runner.learn()`.
- **File:** `scripts/rsl_rl/train_nav.py`

---

## Deployment Status (2026-03-18)

**DEPLOYED AND TRAINING** on H100 in screen `nav_train`.

- **Config:** 2048 envs, 30K iters, save every 100, AI coach every 250
- **GPU:** 38% utilization, 50.5 GB / 96 GB VRAM
- **TensorBoard:** http://172.24.254.24:6006
- **Panels:** `Curriculum/terrain_level`, `Nav/*`, `AI_Coach/*`, `Reward_Weights/*`, `Weight_Changes/*`
- **Coach:** Working — first consultation at iter 250, logs to `spot_nav_explore_ppo/coach_decisions.jsonl`
- **Total bugs fixed:** 22 (N-1 through N-22)

---

## Lesson Learned

**Always test against the target Isaac Lab version before deployment.** The NAV_ALEX codebase was developed against a newer local Isaac Lab (pip-installed with Isaac Sim 5.1). The H100 has a source-installed Isaac Lab with different:
- Module paths (`isaaclab_assets.robots.spot` vs `boston_dynamics`)
- Config classes (`RslRlOnPolicyRunnerCfg` vs `OnPolicyRunnerCfg`)
- Terrain APIs (`HfWaveTerrainCfg` vs `MeshWaveTerrainCfg`, different ObjectCfg fields)
- Available MDP functions (no `terrain_out_of_bounds`, no `terrain_levels_vel`)
- Config patterns (`SimulationCfg` vs `SimCfg`)

**Prevention:** Run a `--max_iterations 1` smoke test on H100 immediately after SCP, before writing any training scripts. Fix all import/config errors before attempting full training.
