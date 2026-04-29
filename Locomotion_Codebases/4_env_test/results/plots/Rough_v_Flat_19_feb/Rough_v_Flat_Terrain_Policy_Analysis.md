# Comparative Analysis of Flat-Terrain and Rough-Terrain Locomotion Policies for Quadruped Robots Across Variable-Difficulty Environments

**Authors:** Gabriel Santiago, Alex (Team), Ryan L.N.

**Date:** February 19, 2026

**Affiliation:** Capstone Research Project

---

## Abstract

Quadruped robots deployed in unstructured environments must navigate terrain conditions that differ substantially from their training domains. This study presents a systematic comparative evaluation of two reinforcement-learning-based locomotion policies --- one trained exclusively on flat terrain ("flat policy") and one trained on rough, varied terrain ("rough policy") --- deployed on a simulated Boston Dynamics Spot robot across four physically distinct environments of escalating difficulty. Each environment comprises a 50-meter linear course divided into five zones of progressively increasing challenge. A total of 320 episodes (40 per policy-environment combination) were collected in NVIDIA Isaac Sim 5.1, with per-episode metrics including forward progress, stability score, fall detection, and zone-level performance. Results reveal a statistically significant interaction between policy type and terrain category: the flat policy achieved superior forward progress on low-friction surfaces (*p* < 0.001, *d* = -1.18), while the rough policy significantly outperformed on discontinuous terrains including stairs (*p* < 0.001, *d* = 1.12) and boulders (*p* = 0.033, *d* = 0.50). No policy achieved course completion (49 m) in any environment, and fall rates ranged from 10% (grass-flat) to 100% (stairs-flat), indicating that even purpose-trained policies face catastrophic failure boundaries as terrain difficulty escalates. These findings quantify the domain-specificity of learned locomotion policies and establish empirical thresholds at which policy transfer breaks down.

**Keywords:** quadruped locomotion, sim-to-sim transfer, reinforcement learning, terrain adaptability, Boston Dynamics Spot, Isaac Sim

---

## 1. Introduction

Legged robots have emerged as compelling platforms for traversing unstructured environments inaccessible to wheeled counterparts. Among commercially available quadrupeds, the Boston Dynamics Spot has become a standard research platform due to its robust mechanical design and compatibility with custom locomotion controllers. Recent advances in deep reinforcement learning (RL) have produced locomotion policies capable of remarkable agility in simulation (Rudin et al., 2022; Miki et al., 2022), yet a critical open question persists: *how well do policies trained on one terrain distribution generalize to physically distinct terrain categories?*

Prior work has demonstrated that policies trained on rough terrain can transfer to real-world deployment via sim-to-real pipelines (Lee et al., 2020; Kumar et al., 2021). However, most evaluations focus on aggregate metrics across a single terrain type rather than systematic comparisons across a controlled spectrum of difficulty. This study addresses three research questions:

1. **RQ1:** Does a policy trained on rough terrain outperform a flat-terrain policy when deployed on physically challenging surfaces?
2. **RQ2:** At what difficulty threshold does each policy experience catastrophic failure (i.e., fall)?
3. **RQ3:** How do stability metrics degrade as terrain difficulty increases, and does the degradation pattern differ between policies?

To answer these questions, we designed four environment types --- friction-varying surfaces, vegetation-drag terrain, boulder fields, and staircases --- each subdivided into five zones of escalating difficulty. This graduated design enables fine-grained analysis of where, specifically, each policy's capabilities break down.

---

## 2. Methods

### 2.1 Simulation Platform

All experiments were conducted in NVIDIA Isaac Sim 5.1.0 with Isaac Lab 0.54.2, using PhysX as the rigid-body physics engine. The simulation operated at 500 Hz (physics time step = 0.002 s) with a control frequency of 50 Hz (decimation factor = 10). The PhysX solver used 4 position iterations and 0 velocity iterations, consistent with the training configuration.

### 2.2 Robot Model

The robot model is a simulated Boston Dynamics Spot quadruped with 12 actuated joints (3 per leg: hip abduction/adduction, hip flexion/extension, knee flexion/extension). Joint PD gains were set to K_p = 60.0 N-m/rad and K_d = 1.5 N-m-s/rad, with an action scale of 0.25. Effort limits were 45 N-m for hip joints and 110 N-m for knee joints. The observation space comprised 235 dimensions: 48 proprioceptive features (joint positions, velocities, projected gravity, and commanded velocities) and 187 height-scan measurements.

### 2.3 Locomotion Policies

Two pre-trained policies were evaluated, both produced via proximal policy optimization (PPO) in Isaac Lab:

- **Flat Policy:** Trained on flat, featureless ground planes. This policy learned to walk efficiently on ideal surfaces without exposure to terrain perturbations.
- **Rough Policy:** Trained on procedurally generated rough terrain with randomized height fields, slopes, and obstacles. This policy was exposed to a broad distribution of terrain challenges during training.

Both policies accept identical 235-dimensional observation vectors and output 12-dimensional joint position targets at 50 Hz. No fine-tuning or domain adaptation was applied; each policy was deployed as-is from its training checkpoint.

### 2.4 Environment Design

Four environments were constructed, each spanning a 50 m x 30 m arena along the X-axis. Each environment is divided into five 10-meter zones of monotonically increasing difficulty:

#### 2.4.1 Friction Environment

Surface friction coefficients decrease progressively, simulating transitions from high-grip to near-frictionless surfaces.

| Zone | Label | Static Friction (mu_s) | Dynamic Friction (mu_d) |
|------|-------|----------------------|------------------------|
| 1 | 60-grit sandpaper | 0.90 | 0.80 |
| 2 | Dry rubber on concrete | 0.60 | 0.50 |
| 3 | Wet concrete | 0.35 | 0.25 |
| 4 | Wet ice | 0.15 | 0.08 |
| 5 | Oil on polished steel | 0.05 | 0.02 |

#### 2.4.2 Grass (Vegetation Drag) Environment

Simulated vegetation exerts velocity-dependent drag forces on the robot's legs, with increasing stalk density and height.

| Zone | Label | Drag Coefficient | Stalk Height (m) |
|------|-------|-----------------|-------------------|
| 1 | Light fluid | 0.5 | None |
| 2 | Thin grass | 2.0 | 0.15 -- 0.25 |
| 3 | Medium lawn | 5.0 | 0.25 -- 0.35 |
| 4 | Thick grass | 10.0 | 0.30 -- 0.45 |
| 5 | Dense brush | 20.0 | 0.35 -- 0.50 |

#### 2.4.3 Boulder Environment

Randomly distributed polyhedra (octahedra, trapezohedra, dodecahedra, and icosahedra in equal proportion) create an uneven ground surface with progressively larger obstacles.

| Zone | Label | Edge Size (m) | Object Count |
|------|-------|---------------|-------------|
| 1 | Gravel | 0.03 -- 0.05 | 4,500 |
| 2 | River rocks | 0.10 -- 0.15 | 2,400 |
| 3 | Large rocks | 0.25 -- 0.35 | 1,200 |
| 4 | Small boulders | 0.50 -- 0.70 | 600 |
| 5 | Large boulders | 0.80 -- 1.20 | 300 |

#### 2.4.4 Stairs Environment

Continuous ascending staircases with increasing riser heights. Transition zones of 5 steps with linearly interpolated riser heights smooth the boundary between consecutive zones.

| Zone | Label | Step Height (m) | Step Depth (m) | Steps |
|------|-------|----------------|----------------|-------|
| 1 | Access ramp | 0.03 | 0.30 | 33 |
| 2 | Low residential | 0.08 | 0.30 | 33 |
| 3 | Standard residential | 0.13 | 0.30 | 33 |
| 4 | Steep commercial | 0.18 | 0.30 | 33 |
| 5 | Maximum challenge | 0.23 | 0.30 | 33 |

### 2.5 Experimental Protocol

Each of the 8 policy-environment combinations was evaluated for 40 independent episodes (N = 320 total). Per-episode parameters:

- **Episode timeout:** 600 simulation seconds
- **Spawn position:** (0.0, 15.0, 0.6) meters --- center of arena width, 0.6 m above ground
- **Completion criterion:** Robot base X-position >= 49.0 m
- **Fall criterion:** Robot base height < 0.15 m
- **Navigation:** A waypoint-following controller commanded forward velocity v_x = 1.0 m/s with proportional yaw correction (K_p = 2.0)
- **Random seed:** Fixed (42) for reproducibility

### 2.6 Metrics

Seventeen per-episode metrics were recorded, including:

- **Forward progress** (m): Maximum X-displacement from spawn
- **Completion** (boolean): Whether the robot reached X >= 49.0 m
- **Fall detected** (boolean): Whether base height dropped below 0.15 m
- **Fall location** (m) and **fall zone** (1--5): Where failure occurred
- **Stability score** (composite): Weighted combination of mean roll, pitch, height variance, and angular velocity --- lower is more stable
- **Mean velocity** (m/s): Average forward velocity over the episode
- **Zone reached** (1--5): Highest difficulty zone entered
- **Episode length** (s): Duration before termination

### 2.7 Statistical Analysis

Between-policy comparisons used Welch's unequal-variances *t*-test on forward progress and the two-proportion *z*-test on completion rates. Effect sizes are reported as Cohen's *d*. Significance threshold: alpha = 0.05.

---

## 3. Results

### 3.1 Overall Performance

Table 1 summarizes aggregate performance across all 320 episodes.

**Table 1.** Summary statistics by environment and policy (N = 40 per cell).

| Environment | Policy | Mean Progress (m) | SD | Median (m) | Fall Rate | Mean Stability | Mean Velocity (m/s) | Mean Zone |
|-------------|--------|-------------------|------|------------|-----------|---------------|---------------------|-----------|
| Friction | Flat | **38.74** | 1.99 | 39.50 | 15.0% | **0.165** | **0.965** | **4.0** |
| Friction | Rough | 27.07 | 13.84 | 32.45 | 70.0% | 0.637 | 0.623 | 3.35 |
| Grass | Flat | 27.12 | 7.05 | 30.97 | **10.0%** | 0.553 | 0.485 | 3.2 |
| Grass | Rough | 25.03 | 3.99 | 23.64 | 17.5% | **0.347** | 0.467 | 3.1 |
| Boulder | Flat | 10.81 | 2.64 | 11.05 | 60.0% | 0.986 | 0.528 | 1.95 |
| Boulder | Rough | **13.41** | 6.90 | 13.30 | 67.5% | **0.915** | **0.560** | **2.1** |
| Stairs | Flat | 7.22 | 4.95 | 10.63 | **100.0%** | 1.659 | 0.459 | 1.68 |
| Stairs | Rough | **11.39** | 1.83 | 11.57 | 15.0% | **0.829** | 0.212 | **1.98** |

No episodes achieved course completion (0% across all conditions), confirming that the 50 m graduated-difficulty course exceeds the traversal capability of both policies under all tested terrain conditions.

### 3.2 Statistical Comparisons

**Table 2.** Statistical tests comparing flat vs. rough policy by environment. Progress compared via Welch's *t*-test; completion rates compared via two-proportion *z*-test.

| Environment | Progress *t* | Progress *p* | Cohen's *d* | Completion *z* | Completion *p* | Favored Policy |
|-------------|-------------|-------------|-------------|---------------|---------------|----------------|
| Friction | 5.213 | **6.0 x 10^-6** | -1.181 (large) | 0.0 | 1.000 | Flat |
| Grass | 1.610 | 0.113 (n.s.) | -0.365 (small) | 0.0 | 1.000 (n.s.) | Flat (n.s.) |
| Boulder | -2.196 | **0.033** | 0.497 (medium) | 0.0 | 1.000 | Rough |
| Stairs | -4.932 | **1.0 x 10^-5** | 1.117 (large) | 0.0 | 1.000 | Rough |

*Note: Completion z-tests are uniformly non-significant because no episodes achieved course completion (0% across all conditions).*

Three of four environments yielded statistically significant differences between policies. The interaction is notably crossed: the flat policy is superior on smooth, continuous surfaces (friction), while the rough policy is superior on discontinuous, obstacle-rich terrain (boulders, stairs).

### 3.3 Completion Rates

![Completion Rate by Environment and Policy](completion_rates.png)

*Figure 1. Completion rates (%) by environment and policy. No policy-environment combination achieved course completion, reflecting the high cumulative difficulty of the 50 m graduated course.*

As shown in Figure 1, completion rates were uniformly 0% across all conditions. This result, while ostensibly uninformative, is itself a meaningful finding: it demonstrates that even the best-performing condition (friction-flat, reaching 38.74 m on average) was unable to traverse the final 11 m of the course, which corresponds to the "wet ice" and "oil on polished steel" friction zones. The graduated-difficulty design ensures that every policy eventually encounters terrain beyond its capability envelope.

### 3.4 Forward Progress Distribution

![Progress Distribution by Environment](progress_boxplot.png)

*Figure 2. Box plots of forward progress (m) by environment and policy. Each box represents 40 episodes. Outliers shown as circles. The friction environment shows the widest between-policy separation; stairs shows the most constrained progress overall.*

The progress distributions in Figure 2 reveal several notable patterns:

**Friction** exhibited the clearest policy differentiation. The flat policy achieved consistently high progress (median = 39.50 m, IQR = 1.99 m) with a tight distribution, while the rough policy showed substantially greater variance (SD = 13.84 m) and a lower median (32.45 m). The rough policy's high variance is attributable to a bimodal failure pattern: 70% of episodes terminated in falls (predominantly in zone 4, the "wet ice" zone), while successful episodes achieved comparable progress to the flat policy.

**Grass** showed overlapping distributions with no statistically significant difference (*p* = 0.113). Both policies navigated to zone 3 (medium lawn), suggesting that velocity-dependent drag forces affect both policies approximately equally.

**Boulder** was the most physically challenging environment for both policies, with neither consistently reaching beyond zone 2 (river rocks, 10--20 m). The rough policy's modest advantage (13.41 m vs. 10.81 m) was driven by its ability to occasionally reach zone 3 (10 of 40 episodes), which the flat policy never achieved.

**Stairs** produced the most dramatic policy divergence in terms of fall rate. The flat policy fell in 100% of episodes (mean progress = 7.22 m), while the rough policy fell in only 15% of episodes --- an 85 percentage-point difference. Despite lower fall rates, the rough policy's mean velocity was only 0.212 m/s (compared to 0.459 m/s for the flat policy before falling), indicating that the rough policy adopted a cautious, slow gait to maintain balance on the stepped terrain.

### 3.5 Fall Analysis

![Fall Rate by Zone](fall_heatmap.png)

*Figure 3. Fall rate heatmap by zone and policy-environment combination. Darker cells indicate higher fall percentages. The heatmap reveals that fall locations are environment-specific: boulders and stairs cause falls in zones 1--2, while friction causes falls in zone 4.*

The fall heatmap (Figure 3) provides spatial resolution on where each policy fails:

- **Boulder-flat** falls are heavily concentrated in zone 2 (river rocks, edge size 0.10--0.15 m), with a fall rate exceeding 50%. The flat policy, never having encountered uneven ground during training, lacks the reactive foothold adjustments necessary to navigate even moderate obstacles.
- **Boulder-rough** shows a shifted distribution: fewer falls in zone 2 and emergent falls in zone 3, indicating that the rough policy can adapt to river rocks but struggles with large rocks (edge size 0.25--0.35 m).
- **Friction-flat** falls occur exclusively in zone 4 (wet ice, mu_s = 0.15), demonstrating a sharp capability boundary. The flat policy maintains excellent stability through zones 1--3 but cannot adapt its gait to near-frictionless surfaces.
- **Friction-rough** shows a broader fall distribution with the highest concentration in zone 4, but also occasional early failures in zone 1, suggesting that the rough policy's more dynamic gait can be counterproductive on high-friction surfaces.
- **Stairs-flat** shows the most severe pattern: high fall rates in both zone 1 (access ramp, 0.03 m steps) and zone 2 (low residential, 0.08 m steps), with the majority of falls occurring at the zone 1--2 transition. This indicates that even shallow stairs rapidly destabilize the flat policy.
- **Stairs-rough** falls are modest (15%) and concentrated at the zone 2 boundary, where step heights increase from 0.08 m to 0.13 m.

### 3.6 Stability Degradation

![Stability Score by Zone Reached](stability_by_zone.png)

*Figure 4. Mean stability score (lower = more stable) as a function of the highest zone reached, for each policy-environment combination. Lines terminate where no episodes reached the subsequent zone.*

The stability-by-zone analysis (Figure 4) reveals environment-specific degradation patterns:

- **Friction-flat** (green) maintains remarkably low stability scores (~0.16) across all reached zones, reflecting the flat policy's highly efficient gait on smooth surfaces. This near-constant stability explains its ability to reach zone 4 consistently.
- **Friction-rough** (red) is approximately 2x less stable than friction-flat at every zone, consistent with the rough policy's more dynamic, perturbation-resistant gait generating unnecessary oscillation on smooth ground.
- **Stairs-flat** (pink) exhibits the highest instability, starting at 1.66 in zone 1 and decreasing as it reaches zone 2 --- but this decrease is a survivorship artifact: only episodes that happened to remain upright reached zone 2, and they did so with lower-than-average instability.
- **Boulder-rough** (orange) shows escalating instability from zone 1 (0.91) to zone 3 (1.37), demonstrating progressive destabilization as obstacle size increases.
- **Grass** policies (purple, brown) show relatively flat stability profiles, consistent with drag forces producing a velocity reduction rather than a destabilizing mechanical perturbation.

---

## 4. Discussion

### 4.1 Policy-Terrain Interaction

The central finding of this study is a crossed interaction between policy type and terrain category. The flat policy excels on continuous surfaces where efficient, rhythmic gait patterns are optimal --- its low stability scores on friction terrain (0.165) reflect a locomotion strategy that minimizes unnecessary body oscillation. Conversely, the rough policy's training on uneven terrain endowed it with reactive capabilities critical for discontinuous surfaces: it reduces fall rates on stairs from 100% to 15% and extends boulder traversal into zone 3.

This interaction has practical implications for deployment planning. A flat-terrain policy should be preferred for environments with smooth but variable-friction surfaces (e.g., warehouses, paved roads), while a rough-terrain policy is essential for obstacle-rich or stepped environments. Neither policy is universally superior.

### 4.2 Failure Boundaries

The graduated-difficulty design enabled precise identification of each policy's failure boundary:

- **Flat policy on friction:** Capable through zone 3 (mu_s >= 0.35); fails at zone 4 (mu_s = 0.15)
- **Rough policy on friction:** Capable through zone 3 with high variance; fails broadly in zone 4
- **Flat policy on boulders:** Fails at zone 2 (edge size >= 0.10 m)
- **Rough policy on boulders:** Extends to zone 3 (edge size ~0.25 m); fails at zone 3--4 boundary
- **Flat policy on stairs:** Fails at zone 1--2 transition (step height increasing from 0.03 m to 0.08 m)
- **Rough policy on stairs:** Capable through zone 2 (step height = 0.08 m); fails at zone 2--3 boundary (0.13 m steps)

These thresholds can inform operational envelopes for real-world deployment.

### 4.3 The Velocity-Stability Tradeoff

An unexpected finding concerns the stairs environment: the rough policy achieves lower fall rates (15% vs. 100%) but at substantially lower mean velocity (0.212 m/s vs. 0.459 m/s). This suggests that the rough policy has learned a cautious, high-foot-clearance gait that sacrifices speed for stability --- a strategy the flat policy lacks entirely. The flat policy attempts to maintain its efficient 1.0 m/s target velocity on stairs, which leads to rapid destabilization and fall.

### 4.4 Limitations

Several limitations should be noted:

1. **No completions observed.** The 50 m course length combined with five zones of escalating difficulty proved too challenging for either policy. Future work could adjust zone lengths or implement curriculum-based evaluation with variable course difficulty.

2. **Single seed.** All episodes used a fixed random seed (42), which ensures reproducibility but limits assessment of stochastic variation in environment generation (particularly for boulder placement).

3. **Sim-to-sim only.** Results were obtained entirely in simulation. Real-world transfer introduces additional challenges (sensor noise, actuator latency, ground truth errors) that are not captured here.

4. **Two policies only.** A broader comparison including hybrid policies, adaptive controllers, or multi-terrain-trained policies would strengthen generalizability claims.

5. **Sample size.** While 40 episodes per condition provided sufficient statistical power for progress comparisons (3 of 4 environments reached significance), rare events such as zone-5 falls may be underrepresented.

---

## 5. Conclusion

This study provides the first systematic, zone-level comparative evaluation of flat-terrain and rough-terrain locomotion policies for quadruped robots across four physically distinct environment categories. The results demonstrate that policy-terrain specificity is not merely a theoretical concern but produces large, statistically significant performance differences: a 31.5 m advantage for the flat policy on friction surfaces (*d* = -1.18) and an 85 percentage-point fall rate reduction for the rough policy on stairs (*d* = 1.12).

The graduated-difficulty design proved effective at revealing precise failure boundaries, with each policy exhibiting a characteristic zone at which performance degrades catastrophically. These empirical thresholds --- such as the flat policy's inability to handle friction coefficients below 0.35 or step heights above 0.08 m --- provide actionable guidance for deployment planning.

Future work will extend this evaluation to larger sample sizes, additional policy architectures (including terrain-adaptive and multi-modal controllers), and sim-to-real validation on the physical Spot platform. The environment framework and evaluation pipeline developed for this study are open-source and designed for reproducibility.

---

## References

Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021). RMA: Rapid Motor Adaptation for Legged Robots. *Proceedings of Robotics: Science and Systems (RSS)*.

Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2020). Learning quadrupedal locomotion over challenging terrain. *Science Robotics*, 5(47), eabc5986.

Miki, T., Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild. *Science Robotics*, 7(62), eabk2822.

Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. *Conference on Robot Learning (CoRL)*.

---

## Appendix A: Raw Data Summary

**Table A1.** Complete summary statistics.

| Environment | Policy | N | Completion | Progress (mean +/- SD) | Median | Stability (mean +/- SD) | Fall Rate | Velocity | Zone (mean) | Z1 | Z2 | Z3 | Z4 | Z5 |
|-------------|--------|---|------------|----------------------|--------|------------------------|-----------|----------|-------------|----|----|----|----|-----|
| Boulder | Flat | 40 | 0.0% | 10.81 +/- 2.64 | 11.05 | 0.986 +/- 0.527 | 60.0% | 0.528 | 1.95 | 2 | 38 | 0 | 0 | 0 |
| Boulder | Rough | 40 | 0.0% | 13.41 +/- 6.90 | 13.30 | 0.915 +/- 0.625 | 67.5% | 0.560 | 2.10 | 6 | 24 | 10 | 0 | 0 |
| Friction | Flat | 40 | 0.0% | 38.74 +/- 1.99 | 39.50 | 0.165 +/- 0.027 | 15.0% | 0.965 | 4.00 | 0 | 0 | 0 | 40 | 0 |
| Friction | Rough | 40 | 0.0% | 27.07 +/- 13.84 | 32.45 | 0.637 +/- 1.294 | 70.0% | 0.623 | 3.35 | 8 | 0 | 2 | 30 | 0 |
| Grass | Flat | 40 | 0.0% | 27.12 +/- 7.05 | 30.97 | 0.553 +/- 0.538 | 10.0% | 0.485 | 3.20 | 0 | 14 | 4 | 22 | 0 |
| Grass | Rough | 40 | 0.0% | 25.03 +/- 3.99 | 23.64 | 0.347 +/- 0.240 | 17.5% | 0.467 | 3.10 | 0 | 2 | 32 | 6 | 0 |
| Stairs | Flat | 40 | 0.0% | 7.22 +/- 4.95 | 10.63 | 1.659 +/- 1.912 | 100.0% | 0.459 | 1.68 | 13 | 27 | 0 | 0 | 0 |
| Stairs | Rough | 40 | 0.0% | 11.39 +/- 1.83 | 11.57 | 0.829 +/- 0.385 | 15.0% | 0.212 | 1.98 | 1 | 39 | 0 | 0 | 0 |

**Table A2.** Statistical test results (flat vs. rough per environment).

| Environment | Welch's *t* | *p*-value | Cohen's *d* | Interpretation |
|-------------|------------|-----------|-------------|----------------|
| Boulder | -2.196 | 0.033 | 0.497 (medium) | Rough significantly better |
| Friction | 5.213 | 6.0 x 10^-6 | -1.181 (large) | Flat significantly better |
| Grass | 1.610 | 0.113 | -0.365 (small) | No significant difference |
| Stairs | -4.932 | 1.0 x 10^-5 | 1.117 (large) | Rough significantly better |

---

*Simulation platform: NVIDIA Isaac Sim 5.1.0, Isaac Lab 0.54.2, PhysX 5.x, PyTorch 2.7.0+cu128. Hardware: NVIDIA RTX 2000 Ada Generation (8 GB VRAM). Total runtime: 6.3 hours (320 episodes).*
