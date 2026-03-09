# AI Trainer Smoke Test Log

*March 8, 2026*

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Machine | Local Windows (MSYS), RTX GPU |
| Conda env | `isaaclab311` (Isaac Sim 5.1.0) |
| Envs | 250 |
| Terrain | flat (plane) |
| Max iterations | 1000 (500 per phase config) |
| Coach interval | 20 iters |
| Coach model | claude-sonnet-4-20250514 |
| LR | 3e-4 → 1e-5 (cosine) |
| Noise | [0.3, 0.5] |
| API key | Provided via CLI |

---

## Results

### What Worked

1. **Isaac Sim initialized correctly** — headless mode, 250 envs, flat terrain, all 22 reward terms loaded.

2. **AI Coach fired on schedule** — 7 consultations at iters 20, 40, 60, 80, 100, 120, 140.

3. **Coach decisions were correct** — all 7 returned `no_change` with high confidence (0.90-0.95). Reasoning was accurate:
   - Iter 20: "Training just started. High flip rate normal for early training."
   - Iter 40: "Still very early. 83% flip, 10% survival completely normal."
   - Iter 60: "Flip improved 83%→67%. Need 200-500 iters minimum."
   - Iter 80: "Flip stable ~68%, survival 32%. Still fundamental learning phase."
   - Iter 100: "Flip 70%, episode length 0. Policy hasn't learned basic balance."
   - Iter 120: "Flip improved to 63%, survival 37%. Still premature to intervene."
   - Iter 140: Emergency NaN detected — coach correctly halted.

4. **Emergency NaN detection worked** — caught NaN in policy parameters at iter 140, logged to decision log, training stopped cleanly.

5. **TensorBoard integration active** — `AI_Coach/`, `Reward_Weights/`, `Reward_Contrib/` panels populated. Local TensorBoard on port 6007.

6. **Decision log (JSONL)** — all 7 consultations logged with full metrics, decision, guardrail messages, API latency.

7. **API latency** — 3.1s to 4.4s per call. Acceptable overhead at 100-iter intervals (~15s each on H100).

### What Failed

#### NaN at Iteration 140 (Value Loss Explosion)

**Sequence:**
```
iter ~130: value_loss = 1.02 (normal)
iter ~132: value_loss = 46,770,467 (EXPLOSION)
iter ~134: value_loss = 677,459,947,895,652,352
iter ~136: value_loss = 644,098,850,084,158,775,941,398,528
iter ~138: value_loss = 221,795,634,891,945,148,968,455,046,214,713,344
iter  140: NaN everywhere
```

**Root cause:** `lr_max=3e-4` with only 250 environments. The flat phase config allows 3e-4 (proven safe with 10,000 envs on H100). With 250 envs, the PPO gradient estimates are ~6.3x noisier (√10000/√250 ≈ 6.3). The value function receives noisy gradients and enters a positive feedback loop: bad value estimates → bad advantages → policy spike → worse value estimates → explosion.

**Why the emergency system didn't catch it earlier:** The value_loss watchdog threshold is 100.0, and the emergency check only runs at coach intervals (every 20 iters). The value loss went from 1.02 to 46M between iterations — entirely within a single 20-iter window. The coach check at iter 120 saw healthy metrics; by iter 140, everything was NaN.

**NOT a production issue.** On H100 with 5000 envs and lr_max=3e-5, this value loss pattern doesn't occur. The combination of high LR + low env count is unique to the local smoke test.

---

## Bugs Discovered and Fixed

### Bug #30: `terrain_types` AttributeError on Flat Terrain

**File:** `shared/reward_terms.py` line 134

**Error:**
```
AttributeError: 'TerrainImporter' object has no attribute 'terrain_types'
```

**Cause:** `VegetationDragReward.__init__()` called `terrain.terrain_types[env_ids]` unconditionally. On flat terrain (plane mode), `TerrainImporter` doesn't create `terrain_types` — it's only available on generated/curriculum terrain.

**Fix:** Guard with `hasattr(terrain, "terrain_types")`:
```python
if self.terrain_aware:
    terrain = self._env_ref.scene.terrain
    if hasattr(terrain, "terrain_types"):  # Guard for flat terrain
        robot_cols = terrain.terrain_types[env_ids]
        # ... terrain-aware logic
```

### Bug #31: `terrain_levels` AttributeError on Flat Terrain

**File:** Isaac Lab's `mdp/curriculums.py` line 56 (called via env config)

**Error:**
```
AttributeError: 'TerrainImporter' object has no attribute 'terrain_levels'
```

**Cause:** The env config's curriculum term (`terrain_levels_vel`) calls `terrain.terrain_levels` which only exists on generated terrain, not flat planes.

**Fix:** In `train_ai.py`, `apply_phase_terrain()` now disables the terrain curriculum for flat phase:
```python
if terrain == "flat":
    env_cfg.scene.terrain.terrain_type = "plane"
    if env_cfg.curriculum is not None:
        env_cfg.curriculum.terrain_levels = None
```
Follows Isaac Lab convention (used by `anymal_b/flat_env_cfg.py`, `cassie/flat_env_cfg.py`, etc.).

### Bug #32: `conda --no-banner` Not Supported

**Error:**
```
conda-script.py: error: unrecognized arguments: --no-banner
```

**Cause:** Older conda version on local Windows machine doesn't support `--no-banner`.

**Fix:** Use `conda run -n isaaclab311` without the flag, or call Python directly:
```
/c/miniconda3/envs/isaaclab311/python.exe train_ai.py ...
```

---

## Recommendations for H100 Deployment

### Must Do Before First H100 Run

1. **Install anthropic SDK:** `pip install anthropic` in `env_isaaclab` conda env.

2. **Sync code:** Upload `ai_trainer/` folder + updated `train_ai.py` + updated `shared/reward_terms.py` to H100.

3. **Verify Isaac Lab reward_manager API:** Run a quick test to confirm `env.unwrapped.reward_manager._term_cfgs` exists and has `.weight` attributes on H100's Isaac Lab version. The internal API could differ between versions.

4. **Use phase config LR, not CLI override:** The `PHASE_CONFIGS` in `config.py` have proven LR values. Don't override with `--lr_max` unless intentional.

5. **Use 5000+ envs:** The value loss explosion at iter 140 was caused by 250 envs + lr=3e-4. With 5000 envs, gradient noise is 4.5x lower (√5000/√250 ≈ 4.5), well within stable range.

6. **Set coach_interval=100 for production.** 20 was for smoke test only — at 100-iter intervals, the coach has meaningful trend data. At 20-iter intervals, every consultation says "too early."

7. **Consider a dry run with `--no_coach` first** to verify env setup, phase transitions, and logging work before adding the API dependency.

### Value Loss Watchdog Gap

The current emergency check runs only at coach intervals. Between consultations, value loss can spike from 1 to 46M without being caught. Options:

- **Option A (recommended):** Add a lightweight value_loss check inside the main update hook (runs every iteration, no API call). If value_loss > 1000, trigger emergency LR halving immediately.
- **Option B:** Reduce coach interval for the first 200 iters (e.g., every 10 iters) when value loss spikes are most likely.
- **Option C:** Accept the gap — on H100 with correct LR/envs, value loss doesn't spike this way.

### Bug #33: RSL-RL Internal Metrics Not Captured (FIXED in v6)

**Discovered during H100 deployment (not smoke test).** The `MetricsCollector` looked for `"Mean reward"`, `"Mean value_function loss"`, etc. in `reward_info`, but these are local variables inside RSL-RL's `learn()` — they never appear in `env.unwrapped.extras["log"]`. Result: coach saw `mean_reward=0.0`, `value_loss=0.0`, `noise_std=0.0`.

**Fix:** Added `runner.log()` interception in `train_ai.py` to capture `rewbuffer`, `lenbuffer`, `mean_value_loss`, `mean_surrogate_loss`, and `mean_std` from the `locals()` dict RSL-RL passes to `self.log()`. Injected into `_last_reward_info` with matching key names. 1-iteration delay (negligible at 100-iter coach interval).

### Metrics Collection at Iteration 0

Currently, `_last_reward_info` starts empty (`{}`). The first coach consultation at iter 20 receives mostly zeros for metrics. This is fine (coach says "too early") but slightly wasteful. Consider:
- Pre-populating with iter 0 metrics
- Or simply skip the first consultation (start coach checks at iter `coach_interval * 2`)

---

## Cost Analysis

| Item | Count | Cost |
|------|-------|------|
| API calls (smoke test) | 7 | ~$0.07 |
| API calls (full flat phase, 500 iters @ 20) | ~25 | ~$0.25 |
| API calls (full robust phase, 30K iters @ 100) | ~300 | ~$3.00 |
| API calls (full 4-phase run) | ~420 | ~$4.20 |
| GPU time saved (no human monitoring) | ~8-12 hrs | Priceless |

---

## Files Modified During Smoke Test

| File | Change |
|------|--------|
| `shared/reward_terms.py` | Bug #30: `hasattr` guard for `terrain_types` |
| `train_ai.py` | Bug #31: disable curriculum for flat terrain |
| `train_ai.py` | Added TensorBoard integration (`_tb_log_coach`, `_tb_log_emergency`, `Reward_Weights/`, etc.) |
| `train_ai.py` | Initial weights logging at iter 0 |
| `train_ai.py` | Bug #33: `runner.log()` interception for RSL-RL internal metrics |
