# Final Capstone Policy (22100) vs Flat Policy — Poster Analysis

**Setup:** 100 episodes per policy × 4 environments (Friction / Grass / Boulder / Stairs)
in NVIDIA Isaac Sim 5.1.0. Same 50 m graduated-difficulty arenas as the
Feb-19 Rough-vs-Flat baseline. Headless on H100, Apr 29–30 2026.

**Policies:**
- **Flat** — `flat_baseline.pt` (baseline from `parallel_2026-02-21_08-24-21`)
- **22100 (Final Capstone)** — `parkour_phasefwplus_22100.pt` with `--mason` obs ordering, action_scale 0.3, per-env target_vx + zone_slowdown_cap

## 1. Headline (poster centerpiece)

`headline_panel.png` — completion rate + mean progress side-by-side.

| Env       | Flat completion | 22100 completion | Δ          |
|-----------|----------------:|-----------------:|-----------:|
| Friction  | **0%**          | **96%**          | **+96 pp** |
| Grass     | 0%              | **75%**          | +75 pp     |
| Boulder   | 0%              | 0%               | 0 pp       |
| Stairs    | 0%              | 0%               | 0 pp       |

| Env       | Flat progress (m) | 22100 progress (m) | Δ          |
|-----------|------------------:|-------------------:|-----------:|
| Friction  | 38.6 ± 4.2        | **47.8 ± 8.5**     | +9.3 m     |
| Grass     | 26.6 ± 7.2        | **47.9 ± 3.5**     | +21.3 m    |
| Boulder   | 10.8 ± 3.2        | **23.3 ± 3.7**     | +12.5 m    |
| Stairs    |  7.3 ± 5.0        | **21.2 ± 3.7**     | +13.9 m    |

## 2. Statistical results

Welch's *t* on progress (n=100 each); two-proportion *z* on completion + fall.

| Env       | t      | p (progress) | Cohen's *d* | Δcompletion | p (compl) | Δfall    | p (fall)   | sig   |
|-----------|-------:|-------------:|------------:|------------:|----------:|---------:|-----------:|:-----:|
| Friction  |  +9.80 | < 1e-17      | **+1.39**   | +96 pp      | < 1e-40   | −8 pp    | 2.66e-2    | ***   |
| Grass     | +26.50 | < 1e-56      | **+3.75**   | +75 pp      | < 1e-27   | −15 pp   | 5.65e-5    | ***   |
| Boulder   | +25.63 | < 1e-63      | **+3.62**   | 0 pp        | n.s.      | −62 pp   | < 1e-20    | ***   |
| Stairs    | +22.55 | < 1e-53      | **+3.19**   | 0 pp        | n.s.      | −48 pp   | 4.66e-15   | ***   |

22100 is significantly better on progress in **all four environments** with
*large* Cohen's *d* (≥ 1.39 everywhere; > 3 on grass/boulder/stairs).
Completion advantage is the headline on smooth terrain (friction + grass);
fall-rate reduction is the headline on rough terrain (boulder 62 pp drop
to **0%**, stairs 48 pp drop).

## 3. Failure-mode shift (fall + stall heatmaps)

`fall_heatmap.png`, `stall_heatmap.png` — both 2-panel (Flat | 22100):

- **Friction:** flat falls in zone 4 (wet ice, μ=0.15) — 11 falls. 22100
  pushes through zone 5 (oil/μ=0.05) and only 3 falls remain.
- **Grass:** flat falls 15× across z2-z4. 22100 has **zero falls**;
  remaining 25 stalls are in zone 5 (dense brush) — robot upright, slowed
  by drag.
- **Boulder:** flat falls 62× concentrated in zone 2 (river rocks,
  edge≈0.1 m). 22100 has **zero falls** — 82 episodes wedge in zone 3 (large
  rocks, ≈0.3 m), 18 break through to zone 4. The failure mode flips from
  *catastrophic* (fall) to *quasi-static* (timeout).
- **Stairs:** flat falls 99× in z1-z2 (3-8 cm steps). 22100 cuts that to
  51 falls (still concentrated z2 → z3 transition, 8→13 cm), and 49
  episodes timeout upright on the z2 plateau.

## 4. Velocity & zone-reach lift

`mean_velocity.png`, `zone_distribution.png`:

| Env      | Flat vel (m/s) | 22100 vel (m/s) | Flat mean zone | 22100 mean zone |
|----------|---------------:|----------------:|---------------:|----------------:|
| Friction | 0.95           | **1.32** (+39%) | 4.0            | **4.86**        |
| Grass    | 0.49           | **1.36** (+178%)| 3.1            | **4.93**        |
| Boulder  | 0.52           | 0.56            | 1.9            | **3.18**        |
| Stairs   | 0.43           | 0.57 (+33%)     | 1.7            | **2.72**        |

## 5. Stability behaviour

`stability_by_zone.png` — mean stability score vs. highest zone reached
(lower = more stable). 22100 traces are solid lines, flat are dashed.

Key observations:
- 22100 maintains lower stability score than flat on grass and boulder
  across every shared zone.
- On friction, 22100's stability rises sharply at zone 5 — the price of
  pushing into terrain flat never reached.
- Stairs: both policies show high stability scores, but 22100's distribution
  is centered on zone 2-3 vs flat's zone 1-2.

## 6. File index

```
plots_22100_vs_flat/
├── headline_panel.png         # POSTER — completion + progress
├── composite_4panel.png       # POSTER — PPT widescreen 4-panel
├── outcome_pies_4x2.png       # 4 envs × 2 policies, falls/stalls/complete
├── completion_rates.png       # grouped bar with sig markers
├── fall_rates.png             # grouped bar with sig markers
├── mean_progress.png          # grouped bar w/ ±SD + sig markers
├── progress_boxplot.png       # full distribution
├── mean_velocity.png          # grouped bar
├── zone_distribution.png      # stacked-bar pair, max zone reached
├── fall_heatmap.png           # 2-panel zone × env, both policies
├── stall_heatmap.png          # 2-panel zone × env, both policies
├── stability_by_zone.png      # stability vs zone, all envs both policies
├── summary.csv                # 8 rows (4 env × 2 policy) full stats
├── statistical_tests.csv      # 4 rows, t/p/d + completion-z + fall-z
└── Poster_Analysis.md         # this file
```

Generated by `../visualize_22100_vs_flat.py`.
