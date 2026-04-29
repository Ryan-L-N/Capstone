# Pedipulation Bug Log

## Bug P-1: RewardManager rejects `**kwargs` in ManagerTermBase.__call__

**Date:** 2026-03-19
**Severity:** Blocking (training won't start)
**Status:** FIXED

**Symptom:**
```
ValueError: The term 'foot_tracking' expects mandatory parameters: ['kwargs']
and optional parameters: [], but received: ['sigma', 'fl_cfg', 'fr_cfg'].
```

**Root Cause:**
Isaac Lab's `RewardManager._resolve_common_term_cfg()` inspects the `__call__` signature
of reward term classes to validate that config params match expected parameters. When
`__call__` uses `**kwargs`, the introspection sees `kwargs` as a mandatory VAR_KEYWORD
parameter and fails to match the actual param names from the config.

**Bad Code:**
```python
class FootTrackingReward(ManagerTermBase):
    def __call__(self, env, **kwargs):  # RewardManager can't match params
        ...
```

**Fix:**
Explicitly list all config params in the `__call__` signature, even though we already
extracted them in `__init__`:
```python
class FootTrackingReward(ManagerTermBase):
    def __call__(self, env, sigma=0.1, fl_cfg=None, fr_cfg=None):
        ...
```

Same fix applied to `StandingStabilityReward.__call__` (added `sensor_cfg=None`).

**Lesson:**
For Isaac Lab ManagerTermBase reward classes, **never use `**kwargs`** in `__call__`.
Always explicitly declare every parameter that appears in the RewardTermCfg.params dict,
even if the values are only used in `__init__`. The RewardManager validates the signature
at environment construction time.

---

## Bug P-2: Deprecation warning for `quat_rotate_inverse`

**Date:** 2026-03-19
**Severity:** Warning (non-blocking, training runs fine)
**Status:** Known, will fix later

**Symptom:**
```
WARNING: The function 'quat_rotate_inverse' will be deprecated in favor of
the faster method 'quat_apply_inverse'. Please use 'quat_apply_inverse' instead....
```

**Root Cause:**
`mdp/rewards.py` uses `isaaclab.utils.math.quat_rotate_inverse` which is deprecated
in newer Isaac Lab versions. Should migrate to `quat_apply_inverse`.

**Location:** `mdp/rewards.py:_foot_pos_body_frame()` function.

**Fix (deferred):** Replace `quat_rotate_inverse` with `quat_apply_inverse` when
confirmed that the API is identical. Not urgent — both produce correct results.

---

## Bug P-3: Value loss spike at iter 0 (expected)

**Date:** 2026-03-19
**Severity:** Informational (expected behavior, not a bug)
**Status:** N/A — by design

**Symptom:**
```
[GUARD] Value loss spike: 195.6 > 100.0 at iter 0 (spike #1). Halving LR for 50 iters.
```

**Explanation:**
After weight surgery, the critic has zero-initialized columns for the 5 new pedipulation
dims. The reward landscape is completely different (12 new terms vs 19 locomotion terms),
so the critic's value estimates are garbage at the start. The value loss watchdog correctly
detects this and halves the LR for 50 iters while the critic calibrates.

Additionally, the critic warmup (300 iters) keeps the actor frozen, preventing the large
value loss gradients from corrupting the walking policy. This is working as intended.
