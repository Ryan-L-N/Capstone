# Final Capstone Policy — Plain-Language Story

*A plain-English walkthrough of the Final Capstone Policy training arc: what we were trying
to do, the bugs that nearly killed it, and how we ended up with a walking policy.*

---

## What we were actually trying to do

Teach a Spot robot one **single policy** that can handle four very different
arenas — low-friction floor, grass (drag), boulder field, stairs — and *also*
pass the Cole navigation arena (obstacle-clutter waypointing). Prior work had
trained a separate policy per terrain type (V14, V16, V18, V19 for stairs).
That approach hit a wall: the stairs policy topped out at 21m and couldn't get
any better no matter how we tuned it. And shipping four different policies that
each only knew one arena isn't a real product.

**The goal:** one policy, all four arenas, plus Cole. One brain, many terrains.

---

## Why it was called "Final Capstone Policy"

The original design for this unified policy was a 3-week build. We had 5.5
calendar days. Team7 was sharing the H100 GPU server and took it over Tuesday
5PM through Thursday 8AM — 39 hours of the budget blocked right at the start.
That left about 72 hours of actual training time to ship something good.

So: ambitious design, compressed into a window so tight that any single big bug
would blow the whole project. Hence "Final Capstone Policy."

---

## The starting point: V18, V19, and the 21m wall

Before Final Capstone Policy, we had a stairs policy called V18. It could walk up to stair
zone 3 (about 21m), then flip. V19 tried to fix it by adding an "altitude
reward" — give the policy credit for climbing higher. V19 learned to climb…
but it also destabilized on flat ground. Fine-tune pressure on a new reward
term piled onto the existing optimizer momentum and broke the gait.

**Lesson banked:** adding new reward terms to a partially-trained policy is
risky. It doesn't just teach the new skill — it warps the old ones too.

That failure drove the decision to start the unified policy from scratch with a
clean reward stack rather than trying to fine-tune V18 into a parkour policy.

---

## The design (what "full original" means)

Five pieces, all at once:

1. **Teacher-student architecture.** Train a "teacher" that gets privileged
   information (true friction coefficient, true robot mass, a 2D map of
   terrain heights). Then distill into a "student" that only sees what a real
   robot could see (proprioception, IMU, height scan) by copying the teacher.
2. **Asymmetric critic.** During teacher training, the *actor* (the thing
   that outputs actions) only sees normal observations, but the *critic* (the
   thing that evaluates how good a state is) sees the privileged info. This
   lets the critic give better value estimates without leaking privileged data
   into the deployable actor.
3. **Obstacle scatter on every terrain patch.** Don't have a "boulder patch"
   and separately a "flat-clutter patch" — drop obstacles INTO the stairs and
   slopes and boulders too. Forces the policy to navigate *and* walk at once.
4. **10-step observation history.** Policy sees not just the current frame but
   the last 10. Helps with occluded obstacles and sensor noise.
5. **Parkour-paper domain randomization ranges.** Huge friction range (0.6–2.0),
   wide mass range (0–3kg added), aggressive pushes (±0.6 m/s every 6–10s).
   Forces robustness.

---

## Phase 1: wiring it up locally (Tue→Thu)

Team7 had the H100, so Phase 1 was pure code. Build the environment config,
wire the asymmetric critic into RSL-RL (the RL framework), write the privileged
observation functions, build the obstacle scatter module.

One real gotcha here — **Python import shadowing.** We named a new config
directory `configs/`. Python treated it as a "namespace package" hidden by a
*real* package called `configs/` in a sibling project (SIM_TO_REAL). That meant
our new configs were silently being shadowed by old ones — the policy was
training with the wrong environment and we didn't notice until the smoke run
showed wrong observation shapes. Fix: rename to `pn_cfg/` and add an explicit
`__init__.py`.

**Lesson banked:** never give a config directory a generic name that another
project in the same Python path might also use.

The 32-env × 10-iter smoke test passed Wednesday noon. Clean exit, right
architecture (actor 235→12, critic 485→1), checkpoints wrote. Green light to
go big.

---

## Phase 2: six days of NaN

This is where it got painful. Gabriel gave the "GO" signal Tuesday night when
he noticed Team7 wasn't actually using the H100 (0% GPU util, no active
sessions). We launched the full teacher training: 4096 environments, 8000
iterations, resume the actor from a previously-trained baseline
(`hybrid_nocoach_19999.pt`) but start with a fresh critic.

**Option 1** — standard launch. Iter 0–1 produced normal rewards
(−17.6, −13.9). Iter 2: every metric turned to NaN. Action noise NaN, value
loss NaN, surrogate loss NaN. Training kept running with NaN-clipped actions
→ robot just stood there → episode length pinned at 1000 steps. 19 minutes
wasted.

*What happened:* actor-only resume means loading a pretrained actor paired
with a *randomly initialized* critic with a new 485-input shape. On PPO update
step 0, the critic's value estimates are garbage, so the value loss is huge,
so the gradient norms blow through the clipping threshold, NaN propagates,
game over.

**Option 2** — lowered learning rate and noise ceiling so the first PPO step
would be gentler. Went NaN at iter 5. Same failure, slightly delayed.

**Option 3** — freeze the actor completely for the first 100 iterations. Let
the fresh critic fit the value function before it gets to influence the actor.
Went NaN at iter 2. Puzzling — we explicitly froze
`actor.parameters()`.

*Root cause:* RSL-RL's actor has a `policy.std` that's a separate
`nn.Parameter` *outside* `actor.parameters()`. Freezing the actor didn't freeze
the std. Fresh critic's exploding value loss → gradient NaN → std NaN →
actions NaN.

**Option 4** — freeze actor AND std AND zero-initialize the critic's output
layer so initial V_pred = 0 (bounds value loss to something finite). Lasted
three iters before NaN. The critic's zero init was correct; something else
was corrupting forward passes.

**Option 5** — add `nan_to_num` forward **pre-hooks** on actor and critic.
The `HOW_TO_TRAIN_YOUR_RAWDOG.md` notes from prior projects warned that
privileged observations (especially the height-scan raycast) can carry `inf`
or `nan` and pollute the critic forward. Hooks clean inputs before they hit
the network.

Option 5 cleared the warmup period cleanly. Reward climbed 7.3 → 20. At iter
100 the actor unfroze. At iter 106 (six iters later): NaN. Again. By iter
693 the policy was silently writing NaN checkpoints.

*Root cause (the real one, this time):* `torch.nn.utils.clip_grad_norm_` has
a nasty failure mode. If any single parameter gradient is `inf`, the total
norm becomes `inf`, then the clip coefficient is `inf/inf = nan`, then
**every parameter** gets scaled by NaN on the next line (`grad.mul_(clip_coef)`).
One bad gradient anywhere in the model turns ALL weights to NaN in one step.
Pre-hooks on inputs can't stop this, because the bad gradient came from the
loss computation on the output side.

**Option 6** — add `nan_to_num` forward **post-hooks** on actor and critic
outputs too. Sanitize both directions. Warmup cleared (reward +135 with
frozen actor). At iter 100 actor unfroze — there was a massive value-loss
spike (vf_loss went 3e11 → 6e22 → inf), but the post-hooks caught it.
Recovery iter 104: vf_loss back to 0.55. Training stable.

That was the fix. All six options stayed stacked (actor frozen 100 iters,
std frozen, critic head zeroed, pre-hooks, post-hooks). Five protections,
one real fix.

**Lesson banked:** for PPO + gradient clipping, you need `nan_to_num` on
*both sides* of every module. One bad output + gradient clip = instant
global NaN.

---

## The reward-hacking detour

Option 6 trained for ~1400 iterations. Reward climbed from +15 to +228 and
*plateaued*. Looked great on paper. But:

- `error_vel_xy` was **3.05** (should be ~0.5). Policy ignoring velocity
  commands.
- `terrain_levels` stuck at **0.0**. Curriculum never promoted a single env.
- `body_flip_over` had dropped from 82% to 0.7% — policy learned not to
  fall, good.
- `time_out` was **99.3%** of episode terminations.

Translation: the policy stood in place doing tiny up-and-down jiggles that
satisfied the gait/air_time/foot_clearance reward bonuses. It never actually
moved forward. The curriculum couldn't promote anyone because nobody was
covering distance.

We called it "stand still and jiggle." Classic reward hacking — the policy
found a zero-effort way to farm positive reward and the negative terms
weren't strong enough to override it.

**The real root cause** (identified after the memory diff):
`action_scale` was 0.3 in Loco_Policy_5_Final_Capstone_Policy but the resumed actor was trained with
0.2. The same actor output commanded joint positions 1.5× larger, so every
step overshot. PPO gradients at the 0.3-scale context pulled the actor into
a *worse* local minimum (jiggle) because the original learned manifold was
shaped for 0.2.

---

## The fresh start that actually worked (Thu morning)

Killed Option 6 at iter 5555. Accepted that the pretrained actor resume was
the problem, not just a convergence delay.

Three targeted edits before the relaunch:

1. **Reward rebalance.** Bump `base_linear_velocity` (forward speed) from
   5 → 10. Drop `gait` from 10 → 3 and `air_time` from 5 → 2. Make forward
   motion dominate the positive reward stack so "stand still and jiggle"
   can't win.
2. **Tight command curriculum.** Shrink the commanded velocity ranges from
   `x: (-1, 1.5)` to `x: (0.3, 0.8)`. Drop the promote-distance threshold
   from ~15m/ep to ~6m/ep so a baby policy can actually pass and advance.
3. **Softer DR temporarily.** Friction (0.6, 2.0) → (0.8, 1.3), push
   interval 6–10s → 10–14s. Don't overwhelm a newborn policy with
   parkour-paper-level randomization. Widen later.

Launched from scratch, no actor resume, no critic warmup. Iteration time 9s
(half of Option 6's 18s because 4096 envs behaved better on less-stressed
memory). Reward was negative for the first few iters — normal for random
init. At iter 2500 the policy was already at 2/4 COMPLETE on the 4-env eval
and walking forward with an intact gait.

**Lesson banked:** if you have a reward-hacking failure, the fix is almost
always **rebalance the reward stack** or **constrain the action space** —
not add guardrails on top.

---

## Evaluating the policies

As training progressed we evaluated checkpoints every ~500 iterations on the
4-env arena (friction / grass / boulder / stairs). Each env gives one of:

- **COMPLETE**: robot reached the 49m goal.
- **FLIP**: robot's body rotated past the fall threshold (faceplant).
- **FELL**: robot's base height dropped below fall threshold but no flip.
- **TIMEOUT**: 10 minutes of sim time without reaching goal or falling.

| Checkpoint | Friction | Grass | Boulder | Stairs | COMPLETE |
|---|---|---|---|---|---|
| iter 2500 | ✅ 49.5 / 128s | ✅ 49.5 / 222s | FLIP 20.9 z3 | FLIP 24.2 z3 | **2/4** |
| iter 3000 | ✅ 49.5 / 115s | ✅ 49.5 / 237s | FLIP 12.1 z2 | FLIP 16.3 z2 | **2/4** |
| iter 4000 | ✅ 49.5 / 193s | TIMEOUT 46.2 | TIMEOUT 12.7 | FLIP 24.1 z3 | **1/4** |
| iter 6000 | FLIP 32.1 z4 | ✅ 49.5 / 216s | FLIP 22.6 z3 | FLIP 24.2 z3 | **1/4** |

Iter 2500 topped the COMPLETE count but walked with a visible *hop* on grass.
Iter 4000 showed the "freeze instead of fall" pattern — body_flip_over had
dropped so low that the policy learned to stand still when uncertain rather
than risk a fall. That wins on the penalty but loses on progress.

Iter 6000 pushed hard enough to flip on friction, but with a much cleaner gait
than 2500. Which is the better ship candidate? Pure COMPLETE count says 2500.
Visual inspection says 6000's gait is more real. *This is why you never
evaluate a locomotion policy on metrics alone.*

---

## The low-friction gap (and why Phase 3 is running)

Critical insight from iter 6000's friction FLIP: the 4-env friction arena
tests a **lower friction coefficient than our training floor**. When we
softened DR to 0.8–1.3 during the fresh restart, we made the policy
blind to anything slipperier than 0.8. The eval arena is around 0.4 or
lower. The policy has literally never seen that surface during training.

**It's a distribution gap, not a skill deficit.** No amount of extra
training at friction 0.8–1.3 teaches low-friction skills.

**Phase 3 plan** (currently running on H100):
- Resume from `model_6100.pt` (the first ckpt after iter 6000 under wider
  DR — cleanest adaptation baseline).
- Widen domain randomization back toward parkour-paper spec:
  static friction (0.8, 1.3) → (0.6, 2.0), dynamic (0.6, 1.2) → (0.4, 1.8),
  mass 0-1.5kg → 0-3kg, pushes from 10-14s @ ±0.3 → 6-10s @ ±0.6.
- Harden the terrain curriculum: stair ceiling 23cm → 30cm, boulder 50cm →
  60cm + more density, slope angles up to 29°, wider stepping stones,
  denser flat-clutter for Cole.
- Run 2000 more iters (iter 6100 → 8100).

First mid-train eval at iter 6500: visually the best gait we've seen —
doesn't hop like 2500, walks properly, pushes further than any previous
ckpt on boulder/stairs. COMPLETE count still 1/4 at 400 Phase-3 iters (only
20% of the budget), but the trajectory is the right one.

---

## Lessons we're shipping with

1. **PPO + gradient clipping needs `nan_to_num` on both pre AND post forward
   hooks.** One bad output turns all weights NaN in one step without them.
2. **Positive reward bonuses get reward-hacked.** Negative penalties on
   specific failure modes (no-fly, undesired-contacts) are safer.
3. **Action-scale mismatch silently breaks pretrained actor resume.** Same
   actor, same observations, different scale = worse local minimum.
4. **Soft DR during early training, wide DR during fine-tune.** Exploration
   is fragile when the policy can't walk; generalization is fragile when it
   can.
5. **Visual inspection beats COMPLETE count when they disagree.** Metrics
   optimize what you measure. The gait you see in rendered eval is the
   policy you're actually shipping.
6. **Never rename a config directory to `configs/`.** Python namespace
   package shadowing will silently load the wrong file and you will spend
   hours debugging ghosts.
7. **Iteration count reported by RSL-RL is additive to resume.** If you
   resume from iter 5999 and pass `--max_iterations 8000`, you get 5999 +
   8000 = 13999, not 8000. Pass the INCREMENT you want, not the target.
8. **A "GO" from the user overrides pre-committed gates when the evidence
   is clear.** We launched Phase 2 early because Team7 was idle — gate
   said "wait until Thursday," evidence said "H100 is free now." Evidence
   won.

---

## Where we are right now

Phase-3 is running. Iter 6500 evaluated with mixed metric results but the
best visual gait of the series. Training continues to iter 8100 (ETA
~14:00 EDT). Next evals at 7000, 7500, 8100. If Phase-3 holds the clean
gait and either improves friction or matches 2500's COMPLETE count with a
better-looking walk, it becomes the ship candidate over 2500.

The hop was a local minimum. The fresh gait is the real policy.
