# Model Training Evaluation: Locomotion & Navigation Policies

*AI2C Tech Capstone — Carnegie Mellon University | May 2026*
*Team: Alex, Ryan, Colby, Cole | Platform: Boston Dynamics Spot*

---

## 1. Purpose & Scope

> *"The purpose of Model Training Evaluation documents is to help to assess, compare, and communicate the performance of trained models. They contribute to the development of reliable, transparent, and accountable systems."*

This document evaluates the trained locomotion and navigation policies produced over the capstone, focusing on three questions:

1. **Performance growth** — how did each successive generation of policy improve over the last, and which design changes drove those gains?
2. **Training environment fidelity** — how well do the curricula we trained on transfer to the canonical 4-arena evaluation bench, and where do they break down?
3. **Comparative ranking** — across our locomotion and navigation entries, which policies are deployment-ready, which are still gated, and what is the empirical evidence?

We rely on a single canonical evaluation harness — the 4-environment graduated-difficulty arena — for every locomotion claim, and the in-team Cole circular waypoint arena + Skill-Nav Lite battery for every navigation claim. Statistical comparisons use Welch's *t*-test on progress and the two-proportion *z*-test on completion / fall rates, with Cohen's *d* reported as effect size. Sample sizes are 100 episodes per policy / environment for the canonical loco bench and 25 episodes for the navigation arenas (per the deployment-readiness criterion).

**Headline.** The shipped locomotion policy (`parkour_phasefwplus_22100.pt`, "22100") significantly outperforms the flat baseline on every environment in the 4-arena bench (Welch's *t*, *p* < 1e-17, Cohen's *d* between +1.39 and +3.75), lifting friction completion from 0% to 96% and grass completion from 0% to 75%, and converting catastrophic falls into upright stalls on rough terrain (boulder fall rate 62% → 0%, stairs 99% → 51%). On the navigation side, the FM V3 + 7 m raycast + A\* replan stack is the only approach to deliver 25 / 25 waypoint capture with zero falls on the Cole rich arena (1095 m / 654 s / 0 falls); two of the three other approaches are blocked or unfinished.

---

## 2. Evaluation Methodology

### 2.1 Canonical 4-Environment Arena (Locomotion)

Four physically distinct 50 m arenas, each subdivided into five 10 m zones of monotonically increasing difficulty:

| Environment | Mechanism | Zone 1 | Zone 5 |
|---|---|---|---|
| Friction | Surface μ scaling | 60-grit (μ=0.90) | Oil on steel (μ=0.05) |
| Grass | Velocity-dependent drag | Light fluid | Dense brush |
| Boulder | Polyhedra obstacle field | Gravel (3-5 cm) | Large boulders (80-120 cm) |
| Stairs | Continuous ascending steps | Access ramp (3 cm) | Max challenge (23 cm) |

The robot spawns at (0, 15, 0.6) m and is commanded forward at the policy's natural velocity (waypoint follower with proportional yaw correction). The course is judged complete on reaching x ≥ 49.0 m. A fall is registered when base height drops below 0.15 m. Episode timeout is 600 s of simulation time.

### 2.2 Per-Episode Metrics

Seventeen metrics are recorded per episode (full schema in `4_env_test/episode_schema.json`). The headline subset:

- **Forward progress (m)** — max x-displacement from spawn
- **Completion (bool)** — robot reached x ≥ 49 m
- **Fall detected (bool) + Fall zone (1-5)** — body below 0.15 m, and where
- **Stability score** — composite of mean roll, pitch, height variance, angular velocity (lower = more stable)
- **Mean velocity (m/s)** — average forward speed

A *stall* is a derived state: the episode timed out (600 s) without falling and without completing. Stalls indicate the policy stayed upright but could not advance, which is the dominant failure mode of 22100 on rough terrain.

### 2.3 Statistical Methods

- **Progress comparisons:** Welch's unequal-variances *t*-test, *α* = 0.05.
- **Completion / fall rates:** two-proportion *z*-test against pooled standard error.
- **Effect size:** Cohen's *d* (positive = 22100 > flat).
- **Significance markers:** `***` *p* < 0.001, `**` *p* < 0.01, `*` *p* < 0.05, `ns` otherwise.

### 2.4 Navigation Bench

Skill-Nav Lite uses Cole's circular 50 m waypoint arena (25 waypoints A-Y, 0.5 m capture radius). For each policy we report waypoints captured (out of 25), arc length traveled (m), wall-clock time (s), and falls. The "rich" arena adds 46 obstacles + 7 obstacle shapes at 0.25 / 0.25 / 0.0 density. We additionally report the FM V3 max-density 30% boulder battery (4 policies × 25 seeds) for stress-testing.

---

## 3. Locomotion Policy Performance Growth

### 3.1 Policy Lineage

The locomotion policy line evolved through five generations. Each generation was fully evaluated on the 4-arena bench before the next was started.

| Gen | Policy | Checkpoint | Key Design Change |
|---|---|---|---|
| 0 | Custom 22-term | (deprecated) | 22 reward terms, [1024,512,256] MLP, 1.2M params |
| 1 | ARL Baseline | `mason_baseline_final_19999.pt` | Adopted ARL's 11-term reward, [512,256,128] (286K params) |
| 2 | ARL Hybrid | `mason_hybrid_best_33200.pt` | Added 3 safety fixes (terrain-rel-height, DOF limits, clamped action smoothness) |
| 3 | Obstacle Expert | `obstacle_best_44400.pt` | Retrained Hybrid on 60% boulder/stair mix; foot-clearance 0.5 → 2.0 |
| 4 | Distilled Master | `distilled_6899.pt` | Multi-expert distillation (Hybrid + Obstacle) with sigmoid terrain router |
| **5** | **22100 Final** | **`parkour_phasefwplus_22100.pt`** | **H100 PARKOUR\_NAV "Hail Mary": teacher-student + asymmetric critic + obstacle scatter on all terrain** |

Generation 0 was abandoned after 11 trials and ~30 sub-iterations. Subsequent generations all share the same [512, 256, 128] / 286K-parameter architecture, making them directly substitutable in the evaluation harness.

### 3.2 Training Environments by Generation

| Gen | Curriculum | Episode | Envs | Total steps |
|---|---|---|---|---|
| 1 | 12-terrain (40% geometric / 35% surface / 25% compound) | 20 s | 4,096 | ~0.8 B |
| 2 | Same 12-terrain + safety fixes | 20 s | 4,096 | ~2.0 B |
| 3 | 60% boulder/stair, 40% mixed | 20 s | 4,096 | ~2.4 B |
| 4 | 12-terrain (no specialization) | 20 s | 4,096 | ~0.4 B |
| 5 | 12-terrain + obstacle scatter on all rows + privileged-info teacher | 20 s | 4,096 | ~3.5 B |

The 12-terrain curriculum uses an automatic difficulty controller: robots are promoted to harder rows on consistent survival and demoted on falls. Generation 5 added **obstacle scatter on every difficulty row** (not just compound rows), so the student saw rocks even on "flat" terrain — this was the central design change behind 22100's robustness on the friction arena.

### 3.3 Stage-by-Stage Performance Growth

100-episode mean progress per generation (m):

| Generation | Friction | Grass | Boulder | Stairs |
|---|---|---|---|---|
| 1. ARL Baseline | 36.9 | 29.6 | 14.4 | 10.9 |
| 2. ARL Hybrid | **48.9** | 27.2 | 20.3 | 11.2 |
| 3. Obstacle Expert\* | 42.2 | 31.7 | **30.4** | 15.7 |
| 5. **22100 Final** | **47.8** | **47.9** | **23.3** | **21.2** |

\* *Obstacle Expert results from a single-episode evaluation; all other rows are 100-episode means. Generation 4 (Distilled Master) was a development-only checkpoint and is not included in the canonical bench.*

100-episode fall rate per generation:

| Generation | Friction | Grass | Boulder | Stairs |
|---|---|---|---|---|
| 1. ARL Baseline | 37% | 23% | 13% | 20% |
| 2. ARL Hybrid | 2% | 15% | 3% | 36% |
| 5. **22100 Final** | **3%** | **0%** | **0%** | **51%** |

> **Two distinct phase shifts.** Gen 1 → Gen 2 was a *stability* phase shift: friction falls collapsed from 37% to 2% via the three safety additions. Gen 2 → Gen 5 was a *capability* phase shift: progress on every environment closed within 2 m of the goal on smooth terrain and roughly doubled on rough terrain.

### 3.4 Headline: 22100 vs Flat Baseline (n = 100 / env)

The "Flat Baseline" referenced here is the original `flat_baseline.pt` policy from the Feb-19 Rough-vs-Flat study — the same baseline used in every prior phase deliverable, re-evaluated on 100 episodes per environment for parity with the 22100 sample.

| Environment | Flat completion | 22100 completion | Δ progress (m) | *t* | *p* | Cohen's *d* |
|---|---|---|---|---|---|---|
| Friction | 0% | **96%** | +9.25 | +9.80 | < 1e-17 | **+1.39** |
| Grass | 0% | **75%** | +21.27 | +26.50 | < 1e-56 | **+3.75** |
| Boulder | 0% | 0% | +12.52 | +25.63 | < 1e-63 | **+3.62** |
| Stairs | 0% | 0% | +13.91 | +22.55 | < 1e-53 | **+3.19** |

Every environment is `***` significant on the Welch test. Cohen's *d* is "large" (> 0.8) on every environment and exceeds 3.0 on three of four — historically large effects for locomotion-policy comparisons.

> **Smooth terrain → completion lift, rough terrain → fall-to-stall conversion.** On friction and grass the bottleneck for the flat policy was *survival on degrading surface conditions*, which 22100's curriculum directly addresses. On boulder and stairs the bottleneck is *geometry* — neither policy can finish, but 22100 cuts boulder fall rate from 62% to **0%** and stairs from 99% to 51%, transforming the failure mode from catastrophic into recoverable.

### 3.5 Failure-Mode Shift

The fall and stall heatmaps (in `Experiments/Ryan/Final_Capstone_Policy_22100/eval_100ep/plots_22100_vs_flat/`) show the per-zone distribution. Three diagnostic patterns:

- **Friction:** Flat falls concentrate in zone 4 (wet ice, μ = 0.15). 22100 pushes through zone 5 with only 3 falls remaining.
- **Boulder:** Flat fails 62 / 100 in zone 2 (river rocks, ≈0.10 m). 22100 has zero falls; 82 episodes wedge in zone 3 (large rocks, ≈0.30 m), 18 reach zone 4. The failure mode is now timeout, not collapse.
- **Stairs:** Flat fails 99 / 100 in zones 1-2 (3-8 cm steps). 22100 has 51 falls (zone 2 → 3 transition where step height jumps 8 → 13 cm); the other 49 episodes timeout upright on the zone 2 plateau.

Stairs zone 3 remains the open problem for the next training cycle.

---

## 4. Navigation Policy Performance Growth

### 4.1 Approaches

Three navigation approaches were pursued in parallel, plus a deployment-grade hand-tuned baseline ("Skill-Nav Lite"):

| Approach | Sensing | Net | Loco backbone | Status |
|---|---|---|---|---|
| Alex — NAV\_ALEX | 32×32 depth (1,024 pix) | CNN + MLP, 489K params | Boulder V6 (4500) | Active training |
| Colby — CombinedPolicy | 64×64 depth (4,096 pix) | CNN + MLP | Mason Hybrid (33200) | **Blocked** (RSL-RL 5.0.1 API) |
| Cole — VS3 | 48 raycasts (3 layers) | MLP [256,256,128] | SpotFlatTerrainPolicy | Active development |
| **Skill-Nav Lite** | **FM V3 + 7 m raycast** | **A\* with cadence-gated replan** | **22100 Final** | **Deployed (best in-team result)** |

All four share the same hierarchical principle: a high-level navigator outputs `[vx, vy, ωz]` velocity commands at 10-20 Hz; a frozen low-level loco policy converts those to 12 joint targets at 50 Hz. The differences are in the *sensing*, *training curriculum*, and *whether learning is end-to-end* (Alex / Colby / Cole) or *symbolic + reactive* (Skill-Nav Lite).

### 4.2 Training Environments

The three end-to-end approaches use Isaac Lab terrain curricula similar to Generation 5 of the locomotion line. The hand-tuned Skill-Nav Lite uses a deployment-time symbolic planner and is "trained" only in the sense that its hyperparameters were tuned against held-out arena instances.

| Approach | Curriculum type | Stages / levels | Density |
|---|---|---|---|
| Alex (NAV\_ALEX) | Terrain difficulty | 6 levels × 10 types | Boulders 15% (highest weight) |
| Colby | Same as Alex | 6 levels × 10 types | (training blocked — no signal yet) |
| Cole (VS3) | Task complexity | 7 stages: stability → push → 3 nav-range stages → expert | Obstacle 5-25% by stage |
| Skill-Nav Lite | Hyperparameter tuning | 11 manual iterations | 0.25 / 0.25 / 0.0 (46 obstacles, 7 shapes) |

Cole's seven-stage task curriculum is the most distinctive: stability training before any navigation goal, plus an explicit object-pushing phase (Stage 2) where the robot learns to displace 5 lightweight obstacles by 1 m+ each.

### 4.3 Stage-by-Stage Performance Growth

For Cole's circular 50 m / 25-waypoint arena ("Cole rich", quarter density):

| Approach | Waypoints | Distance (m) | Time (s) | Falls | Notes |
|---|---|---|---|---|---|
| Alex / NAV\_ALEX | (no eval yet) | — | — | — | Active training, ~5/5,000 iters |
| Colby / CombinedPolicy | 0 / 25 | — | — | — | Training blocked at iter 99 |
| Cole / VS3 (mason\_hybrid) | 3 / 25 | — | — | — | 30% density, n = 1 |
| Cole / VS3 (baseline) | 2 / 25 | — | — | — | 30% density, n = 1 |
| Cole / VS3 (V6) | 0 / 25 | — | — | — | 30% density, FELL at 3.7 s |
| **FM V3 + 7 m + A\*** | **25 / 25** | **1095** | **654** | **0** | **Quarter density, deployment-grade** |

For the deployment-grade Skill-Nav Lite, additional 3-seed batteries on harder densities:

| Density | Mean waypoints | Falls | Notes |
|---|---|---|---|
| Quarter (deployment) | 19.7 / 25 | 0 / 3 | Honest split: works on sparse unknown |
| Max (stress test) | 3.7 / 25 | 1 / 3 | Seed 42 was lucky; not deployment-ready |

### 4.4 Comparative Analysis

Within the end-to-end approaches, the central trade-off is *sensing horizon vs compute cost*:

| Trade-off | Depth camera (Alex / Colby) | Raycasts (Cole) |
|---|---|---|
| Range | 30 m (∼10 s lookahead at 3 m/s) | ~5 m |
| Compute | Heavy (raycaster bottleneck — 125 s/iter at 64×64) | Light |
| Resolution | 1,024–4,096 pixels | 48 rays |
| Limitation | Static meshes only (no dynamic obstacles) | Reactive only — no route planning |

Skill-Nav Lite sits outside this trade-off: it pairs a 7 m raycast (cheap) with a global A\* replan against a self-built occupancy map. This is what made it the only approach to clear the rich arena with zero falls — the global plan absorbs the short-horizon penalty that pure-reactive Cole VS3 cannot.

> **A trained navigation policy has not yet beaten our hand-tuned planner on the 25-waypoint bench.** The end-to-end approaches are still in active development; Skill-Nav Lite is the current production answer.

---

## 5. Cross-Policy Composition

The shipped robot is a *stack* — 22100 (loco) below Skill-Nav Lite (nav). The 4-arena bench evaluates 22100 in isolation; the navigation bench evaluates Skill-Nav Lite with 22100 as the frozen backbone. The composed performance is roughly the *intersection* of each layer's capability envelope:

- On flat / sparse terrain (the deployment regime), both layers pass: Skill-Nav Lite captures 19.7 / 25 waypoints with zero falls.
- On dense unknown terrain (max-density rich arena), Skill-Nav Lite's planning quality degrades (3.7 / 25, 1 fall over 3 seeds). The bottleneck is the planner, not 22100 — when we replay the same arena under teleop with 22100, the loco layer holds.
- On stairs / boulder beyond zone 2, 22100 is the bottleneck. Even a perfect planner cannot push the stack past 21 m on stairs because the loco layer cannot survive 13 cm steps.

This separation is useful: it tells us the next training cycle should target stairs zone 3 (loco) and dense-arena planning quality (nav) independently.

---

## 6. Limitations & Open Issues

1. **Sample size on completion-event tails.** All loco evaluations use n = 100 / env. For *p* < 1e-50 effects this is more than enough. For rare-event analysis (e.g. zone-5 falls) the per-cell counts are small and conclusions are correspondingly soft.
2. **Single-seed arena instances.** The 4-arena bench uses a fixed seed (42) for reproducibility. Boulder placement variability is therefore not directly captured in the standard deviation; we estimate it separately on the FM V3 30%-density 4-policy battery.
3. **Sim-to-sim only.** Every result here is in NVIDIA Isaac Sim 5.1.0. Real-world transfer introduces sensor noise, actuator latency, and terrain modeling errors that are not captured. Sim-to-real validation on the physical Spot is gated on ARL release of their bench.
4. **Cole VS3 sample size.** Cole's three-policy 30%-density battery is n = 1 per cell, single seed. The "V6 falls" / "mason 3/25" finding warrants a 100-seed re-run before being treated as a hard ranking — this is the in-flight Ryan handoff (`Experiments/Ryan/Nav_evals/`).
5. **Obstacle Expert single-episode evaluation.** The Generation 3 Obstacle Expert was evaluated on a single episode per environment in §3.3 of the locomotion deliverable; the 100-episode confirmation was deprioritized once Generation 5 (22100) was shipped and superseded it.
6. **No paired hardware repro yet.** The biggest open gap is that no policy in this evaluation has been validated on the physical Spot. The full evaluation framework is reproducible end-to-end in simulation; real hardware is the next milestone.

---

## 7. Lessons Learned

1. **Run the same harness on every generation.** The single biggest evaluation-quality decision was committing early to the 4-arena bench (Feb 19). Every locomotion claim in this document compares against that bench, which is why we can quote effect sizes across generations a year apart.
2. **A zero-fall result is not the same as a high-completion result.** 22100 has zero falls on boulders but 0% completion. Reporting only completion would have hidden the failure-mode shift that is the most useful story for the next training cycle.
3. **Hand-tuned baselines are evaluation infrastructure, not embarrassments.** Skill-Nav Lite is the navigation team's best result by a wide margin. Treating it as the production baseline (not a "not-really-a-policy" oddity) is what made the trained-policy gaps quantifiable.
4. **Small sample sizes warn loudly.** Cole's n = 1 density-flip result was correctly flagged at handoff; the 100-seed Ryan re-run will tell us whether the "30% mason > baseline" claim survives. Single-seed claims must come with a re-run plan.
5. **Statistical effect sizes carry the story when *p*-values bottom out.** Cohen's *d* between +1.39 and +3.75 is the more useful number once *p* < 1e-17 — the effect is real and large, but how large matters for prioritization. Reporting only *p* < 0.001 would have flattened a 3× variance in effect size across environments.
6. **Failure modes are policies too.** The fall heatmap (where falls happen, by zone) was added to the bench mid-project after we noticed that "fall rate" alone was hiding the difference between "policy collapses early" and "policy collapses late". Per-zone tracking is now mandatory in any new evaluation we add.
7. **Lock the canonical evaluation harness early; iterate on policies, not benches.** The 4-arena bench has been frozen since February. Every subsequent training cycle has been forced to defend against that bench, which is the only reason we have a clean 5-generation growth curve. Changing the bench mid-project would have erased the comparison.

---

*Generated for the AI2C Tech Capstone Phase Deliverables. Source data, scripts, and per-episode JSONLs are in:*
- *`Locomotion_Codebases/4_env_test/results/parallel_2026-02-21_08-24-21/`* (flat baseline)
- *`Experiments/Ryan/Final_Capstone_Policy_22100/eval_100ep/`* (22100 + comparison plots)
- *`Experiments/Alex/NAV_ALEX/`* (Alex navigation)
- *`Experiments/Colby/CombinedPolicyTraining/`* (Colby navigation)
- *`Experiments/Cole/RL_FOLDER_VS3/`* (Cole navigation)
- *`Experiments/Alex/skill_nav_lite/`* (Skill-Nav Lite hand-tuned baseline)
