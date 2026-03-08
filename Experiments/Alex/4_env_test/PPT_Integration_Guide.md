# PPT Integration Guide — Phase 2 Deliverables Update

> **For Claude PPT:** This document provides slide-by-slide content to integrate into the existing presentation **"Rough-Terrain Autonomous Walking (RAW) Quadruped - Mid Phase Presentation 3.pptx"** (23 slides, widescreen 16:9). Match the existing visual style: dark/professional theme, TITLE_AND_TWO_COLUMNS and TITLE_ONLY layouts, slide numbers in bottom-right, concise bullet points, tables where appropriate.

---

## EXISTING SLIDE STRUCTURE (23 slides)

| # | Title | Action |
|---|---|---|
| 1 | Title Slide — "Immersive Modeling and Simulation for Autonomy" | **UPDATE** date and subtitle |
| 2 | "Are you ready? – IsaacYes" | KEEP |
| 3 | Team Members + Mentors | KEEP |
| 4 | Agenda | **UPDATE** agenda items |
| 5 | Refined Project Scope | KEEP (scope unchanged) |
| 6 | Getting Baseline Data | KEEP |
| 7 | Data Dictionary Summary | **UPDATE** — add obstacle env, update variable count |
| 8 | Five Baseline Testing Environments | **UPDATE** — mention all 5 environments |
| 9 | Deep Dive Friction Environment | KEEP |
| 10 | Deep Dive Friction Environment (cont.) | KEEP |
| 11 | Deep Dive Cluttered Space Environment | KEEP |
| 12 | Clutter Space Environment | KEEP |
| 13 | Failure Reasons Distribution | KEEP |
| 14 | Distribution of Waypoints | KEEP |
| 15 | Waypoints Reached Per Episode | KEEP |
| 16 | Simulation Compute Infrastructure (specs) | KEEP |
| 17 | Simulation Compute Infrastructure (pros/cons) | KEEP |
| 18 | Project Timeline | **UPDATE** statuses |
| 19 | Phase II Detailed Schedule | KEEP |
| 20 | What's Next: RL Pipeline | **REPLACE** — this is no longer "what's next", we did it |
| 21 | Risks to Project Scope | **UPDATE** risk levels |
| 22 | Questions | MOVE to end |
| 23 | Deep Dive Environment total test | KEEP or REMOVE |

**NEW SLIDES TO INSERT** after slide 19 (before current "What's Next"):

---

## SLIDES TO UPDATE

### Slide 1 — Title (UPDATE)

**Title:** Immersive Modeling and Simulation for Autonomy
**Subtitle:** End of Phase II / Mid-Phase III Presentation
**Date:** March 2026

---

### Slide 4 — Agenda (UPDATE)

**Title:** Presentation Agenda

**Left column:**
- Recap: Project Scope & Environments
- Phase 2 Deliverables: Data Pipeline
- Model Documentation Update
- 4-Phase Training Curriculum
- Training Results So Far

**Right column:**
- 5th Environment: Obstacle Navigation
- Safety Mechanisms & Lessons Learned
- Updated Timeline & Risks
- What's Next: Final Evaluation
- Questions

---

### Slide 7 — Data Dictionary Summary (UPDATE)

**Title:** Data Dictionary Summary

Update the table and callout boxes:

**Callout box:** "17 variables per episode record stored as JSONL"

**Table:**

| Variable | Type | Description |
|---|---|---|
| episode_id | string | Unique episode identifier |
| policy | string | "flat" (NVIDIA baseline) or "rough" (curriculum-trained) |
| environment | string | friction, grass, boulder, stairs, **obstacle** |
| completion | bool | Reached end of arena / goal |
| progress | float | Max forward distance (meters) |
| stability_score | float | Composite stability metric (lower = better) |
| fall_detected | bool | Fall occurred during episode |

**Key Metrics callout:** Progress (m), fall_detected, stability_score, zone_reached (1-5)

**Update note:** "5 environments x 2 policies = **10 combinations** x 100 episodes = **1,000 total runs**"

---

### Slide 8 — Five Testing Environments (UPDATE)

**Title:** Five Evaluation Environments

**List (with brief descriptors):**
1. **Friction** — Sandpaper to ice (static friction 0.90 down to 0.05)
2. **Grass** — Light brush to dense vegetation (drag 0.5 to 20.0)
3. **Boulders** — Gravel to large rocks (3cm to 120cm polyhedra)
4. **Stairs** — Shallow ramp to steep stairs (3cm to 23cm risers)
5. **Obstacle** — 100m x 100m field, 360 random objects (furniture, cars, trucks) — NEW

**Note:** "Obstacle environment built by Cole (Testing_Environment_1.py)"

---

### Slide 18 — Project Timeline (UPDATE)

**Title:** Project Timeline

| Phase | Week | Description | Status |
|---|---|---|---|
| Phase 1 | 1-4 | Project Scoping / NVIDIA Omniverse Exploration / Infrastructure | **Complete** |
| Phase 2 | 5-8 | Develop training/testing environments and data exploration | **Complete** |
| Phase 3 | 9-12 | Train RL Model (4-phase curriculum) | **In Progress** |
| Phase 4 | 13-14 | Final evaluation across 5 environments | Future |
| Phase 4.1 | 15-16 | Paper / Presentation / Flexible | Future |

---

### Slide 21 — Risks (UPDATE)

**Title:** Risks to Project Scope

**Risk 1 — Knowledge Gaps: ~~HIGH~~ LOW**
- Resolved through 11+ training trials and 5 critical bug fixes
- Training pipeline is now stable with automatic safety mechanisms
- 4-phase curriculum successfully producing terrain-capable policies

**Risk 2 — Insufficient Infrastructure: ~~MEDIUM~~ LOW**
- H100 stable after BMC recovery procedures established
- 120+ hours of successful GPU training completed
- Server recovery via Redfish API when CUDA deadlocks occur

**Risk 3 — Training Instability: MEDIUM (NEW)**
- NaN divergence and value loss oscillation can crash multi-day runs
- Mitigated by 4 automatic safety mechanisms (NaN sanitizer, watchdog, noise clamp, pre-forward guard)
- Still requires monitoring — not fully autonomous

---

## NEW SLIDES TO INSERT

Insert these slides after current slide 19 (Phase II schedule) and before the updated "What's Next" slide.

---

### NEW SLIDE A — "How Our AI Learns" (Overview)

**Layout:** TITLE_AND_TWO_COLUMNS

**Title:** How Our AI Learns: PPO + Curriculum

**Left column — "The Algorithm":**
- Proximal Policy Optimization (PPO)
- Up to 40,960 simulated robots training simultaneously
- Each robot walks, gets scored, AI updates its "brain"
- Cosine annealing learning rate with warmup
- Custom safety mechanisms prevent training crashes

**Right column — "The Brain":**

```
235 inputs → [1024] → [512] → [256] → 12 joint commands
                ELU       ELU      ELU
```

- 235 inputs: body feel (48) + terrain vision (187 height scan points)
- 12 outputs: target angle for each leg joint
- ~1.8 million trainable parameters
- Separate "actor" (decides) and "critic" (evaluates) networks

---

### NEW SLIDE B — "4-Phase Training Curriculum"

**Layout:** TITLE_ONLY (full-width table)

**Title:** 4-Phase Training Curriculum

**Table:**

| Phase | What It Trains On | Robots | Iterations | Key Result |
|---|---|---|---|---|
| **A — Flat** | 100% flat ground | 20,480 | 500 | 99.3% survival — learned to walk |
| **A.5 — Transition** | 50% flat + gentle rough | 20,480 | 1,000 | 92.9% survival — adapted to bumps |
| **B-easy — Easy Rough** | 11 terrain types, low difficulty | 40,960 | 5,002 | Handles varied terrain |
| **B — Full Rough** | 11 terrain types, max difficulty | 5,000 | Ongoing | Climbing hard stairs, obstacles |

**Bottom callout:** "Each phase builds on the previous — like training wheels before a bike"

**Bottom-right:** "Total: 120+ hours of H100 GPU training across 11+ trial runs"

---

### NEW SLIDE C — "11 Training Terrains"

**Layout:** TITLE_AND_TWO_COLUMNS

**Title:** 11 Custom Training Terrain Types

**Left column — visual grouping (could be icons or a grid image):**

**Geometric (40%):**
- Pyramid stairs (up/down)
- Random boxes
- Stepping stones

**Surface (40%):**
- Random rough ground
- Slopes (up/down)
- Waves, friction planes, vegetation

**Compound (20%):**
- Heightfield stairs
- Discrete obstacles
- Repeated box patterns

**Right column — key parameters:**
- 10 difficulty rows x 40 terrain columns = 400 patches
- Each patch: 8m x 8m
- Curriculum auto-promotes surviving robots to harder rows
- Difficulty scales within each type (e.g., stair height 5cm to 25cm)
- Domain randomization: friction [0.3, 1.5], mass +/-5kg, random pushes every 10-15s

---

### NEW SLIDE D — "Reward Function: Teaching Good Walking"

**Layout:** TITLE_AND_TWO_COLUMNS

**Title:** 19 Reward Terms: What "Good Walking" Means

**Left column — "Earn Points For:"**
- Trot gait synchronization (+10.0)
- Walking at commanded speed (+5.0)
- Turning at commanded rate (+5.0)
- Proper foot timing (+5.0)
- Lifting feet to clear obstacles (+2.0)
- Velocity modulation (+2.0)

**Right column — "Lose Points For:"**
- Hitting joint limits (-5.0)
- Tilting/falling over (-3.0)
- Body scraping ground (-2.0)
- Bouncing/swaying (-2.0)
- Body-ground contact (-1.5)
- Jerky movements (-1.0)
- Asymmetric gait (-1.0)
- Foot slipping (-0.5)
- Plus 5 more small penalties for energy, torque, acceleration

**Bottom callout:** "Weights tuned across 11+ training trials — small changes can crash training"

---

### NEW SLIDE E — "Training Safety Mechanisms"

**Layout:** TITLE_AND_TWO_COLUMNS

**Title:** 4 Safety Mechanisms (Learned the Hard Way)

**Left column — "The Bugs":**

| Bug | What Went Wrong |
|---|---|
| #22 | Height penalty exploded on rough terrain |
| #23 | Learning rate too high → instant crash |
| #24 | NaN corruption in neural network — clamp() doesn't fix NaN |
| #25 | Value loss oscillation → cascading divergence |
| #26 | Too much randomness → robot can't pass curriculum |

**Right column — "The Fixes":**

1. **NaN Sanitizer** — Explicitly detect & replace NaN/Inf before every forward pass
2. **Value Loss Watchdog** — Auto-halve learning rate when loss > 100
3. **Noise Clamping [0.3, 0.7]** — Cap exploration randomness to allow terrain promotion
4. **Disabled height penalty on rough terrain** — World-frame Z doesn't make sense on elevated surfaces

**Bottom callout:** "These bugs each cost 12-48 hours of lost training time. Now they're automatic safeguards."

---

### NEW SLIDE F — "Training Progress: Where We Are"

**Layout:** TITLE_AND_TWO_COLUMNS

**Title:** Training Progress — Phase B (Final Phase)

**Left column — "Current Status":**
- Phase B (Trial 11) running on H100
- 11 terrain types at full difficulty (10 rows)
- 5,000 parallel environments
- Learning rate: 3e-5 (conservative — higher values crash)
- Save interval: every 100 iterations (~65M steps)

**Right column — "Key Metrics":**
- Terrain level: 3.77 and climbing (out of 10)
- Flip rate: ~8.9% (robot flips < 1 in 10 episodes)
- Reward: 10.5 (early — still improving)
- Iteration time: ~12.8 seconds

**Bottom:** "Previous best (Trial 10k) flatlined at terrain 0.83 — curriculum approach is working"

**Note to Claude PPT:** If possible, add a simple line chart placeholder labeled "TensorBoard: Reward vs. Iteration" showing an upward trend.

---

### NEW SLIDE G — "5th Environment: Obstacle Navigation"

**Layout:** TITLE_AND_TWO_COLUMNS

**Title:** New: Obstacle Navigation Arena (Cole)

**Left column — description:**
- 100m x 100m enclosed field
- 360 randomly placed obstacles:
  - 100 large furniture (couch, chair, table, shelf, ottoman, bed, cabinet)
  - 250 small clutter items
  - 5 cars + 5 trucks
- Fixed start: (-45, 0)
- Random goal: 75m+ away
- 5 zones by distance from start (20m bands)

**Right column — why it matters:**
- Tests spatial navigation, not just terrain traversal
- Robot was never explicitly trained on obstacle avoidance
- Tests whether height-scan perception transfers to new challenges
- Qualitatively different from 4 linear courses
- "Can the robot navigate a furniture store?"

**Bottom:** "Source: Cole/Testing_Environments/Testing_Environment_1.py"

**Note to Claude PPT:** If possible, add a top-down schematic of the 100m x 100m field with scattered rectangles representing obstacles, a green dot at (-45, 0) for start, and a red dot far away for goal.

---

### NEW SLIDE H — "Evaluation Pipeline"

**Layout:** TITLE_AND_TWO_COLUMNS

**Title:** Evaluation Pipeline: 1,000 Test Episodes

**Left column — "The Process":**
1. Run each (environment, policy) combination for 100 episodes
2. 5 environments x 2 policies = **10 combinations**
3. Total: **1,000 episodes** (~11 hours on H100)
4. Each episode: robot walks, we record 17 metrics
5. Output: JSONL files → summary stats → plots

**Right column — "Statistical Tests":**
- **Welch's t-test:** Is mean progress different? (flat vs rough per env)
- **Cohen's d:** How big is the effect?
- **Two-proportion z-test:** Is completion rate different?
- **Alpha = 0.05** (95% confidence)

**Visualizations:**
- Completion rate bar charts
- Progress box plots
- Fall zone heatmaps
- Stability-by-zone line plots

---

### UPDATED SLIDE 20 — "What's Next" (REPLACE existing)

**Layout:** TITLE_AND_TWO_COLUMNS

**Title:** What's Next: Phase 3 → 4

**Left column — "Remaining Work":**
- Finish Phase B training (Trial 11 ongoing)
- Select best checkpoint for production evaluation
- Run full 1,000-episode evaluation (~11 hours)
- Generate statistical report and visualizations
- Compare flat vs rough across all 5 environments

**Right column — "Timeline":**

| Week | Task |
|---|---|
| 9-10 | Complete Phase B training, select final model |
| 11-12 | Run production evaluation (1,000 episodes) |
| 13 | Statistical analysis and visualization |
| 14 | Final report and presentation |
| 15-16 | Buffer / refinement |

**Bottom callout:** "The hard part (training) is nearly done. Evaluation is straightforward."

---

## FINAL SLIDE ORDER (after integration)

| # | Slide | Status |
|---|---|---|
| 1 | Title (updated date) | UPDATED |
| 2 | "Are you ready?" | KEPT |
| 3 | Team Members | KEPT |
| 4 | Agenda (updated) | UPDATED |
| 5 | Refined Project Scope | KEPT |
| 6 | Getting Baseline Data | KEPT |
| 7 | Data Dictionary Summary (updated) | UPDATED |
| 8 | Five Evaluation Environments (updated) | UPDATED |
| 9 | Deep Dive Friction (1) | KEPT |
| 10 | Deep Dive Friction (2) | KEPT |
| 11 | Deep Dive Cluttered Space | KEPT |
| 12 | Clutter Space Environment | KEPT |
| 13 | Failure Reasons Distribution | KEPT |
| 14 | Distribution of Waypoints | KEPT |
| 15 | Waypoints Reached Per Episode | KEPT |
| 16 | Compute Infrastructure (specs) | KEPT |
| 17 | Compute Infrastructure (pros/cons) | KEPT |
| 18 | Project Timeline (updated) | UPDATED |
| 19 | Phase II Detailed Schedule | KEPT |
| 20 | **NEW: How Our AI Learns (PPO + Curriculum)** | NEW |
| 21 | **NEW: 4-Phase Training Curriculum** | NEW |
| 22 | **NEW: 11 Training Terrains** | NEW |
| 23 | **NEW: 19 Reward Terms** | NEW |
| 24 | **NEW: Training Safety Mechanisms** | NEW |
| 25 | **NEW: Training Progress (Phase B)** | NEW |
| 26 | **NEW: 5th Env — Obstacle Navigation** | NEW |
| 27 | **NEW: Evaluation Pipeline** | NEW |
| 28 | What's Next (replaced) | REPLACED |
| 29 | Risks (updated) | UPDATED |
| 30 | Questions | MOVED to end |

**Total: 30 slides** (was 23, added 8 new, removed 1 unused)

---

## STYLE NOTES FOR CLAUDE PPT

- **Layouts used in existing deck:** TITLE, TITLE_ONLY, TITLE_AND_TWO_COLUMNS, TITLE_AND_BODY
- **Slide numbers** appear in bottom-right of every slide
- **Tables** use a clean style with header row highlighted
- **Bullet points** are concise (5-8 words each, not sentences)
- **Key numbers** are called out in larger font or colored boxes
- **Tone:** Professional but accessible — this is a capstone presentation for professors and TAs
- **No emojis** in slide content (except slide 2 which is a joke)
- Existing slides reference "clutter"/"cluttered" for what we now call "obstacle" — these are the same environment (Cole's). Update terminology to be consistent.
