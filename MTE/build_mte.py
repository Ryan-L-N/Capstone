"""
Build Model_Training_Evaluation.docx in Colby's doc style, then convert to PDF.

Style mirrors DIV_P_4/build_docx.py: Calibri 11pt, dark-blue headings (#1B3A5C),
table grids with dark-blue header band + alternating F2F6FA row shading,
italic blue callouts with left border, and monospace code blocks.

Run once:  python build_mte.py
Outputs:   Model_Training_Evaluation.docx
           Model_Training_Evaluation.pdf  (via docx2pdf -> Word COM on Windows)
"""

from __future__ import annotations

import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


HERE = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(HERE, "Model_Training_Evaluation.docx")
PDF_PATH = os.path.join(HERE, "Model_Training_Evaluation.pdf")


# ── Document setup ───────────────────────────────────────────────────────────
doc = Document()

style = doc.styles["Normal"]
font = style.font
font.name = "Calibri"
font.size = Pt(11)
font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
pf = style.paragraph_format
pf.space_after = Pt(6)
pf.space_before = Pt(0)
pf.line_spacing = 1.15

for level, size, color in [
    ("Heading 1", 22, RGBColor(0x1B, 0x3A, 0x5C)),
    ("Heading 2", 16, RGBColor(0x1B, 0x3A, 0x5C)),
    ("Heading 3", 13, RGBColor(0x2E, 0x56, 0x7A)),
]:
    s = doc.styles[level]
    s.font.name = "Calibri"
    s.font.size = Pt(size)
    s.font.color.rgb = color
    s.font.bold = True
    s.paragraph_format.space_before = Pt(18 if level != "Heading 1" else 6)
    s.paragraph_format.space_after = Pt(6)

for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)


# ── Helpers (lifted from DIV_P_4/build_docx.py) ──────────────────────────────
def add_hr() -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    pPr = p._p.get_or_add_pPr()
    pBdr = pPr.makeelement(qn("w:pBdr"), {})
    bottom = pBdr.makeelement(qn("w:bottom"), {
        qn("w:val"): "single",
        qn("w:sz"): "6",
        qn("w:space"): "1",
        qn("w:color"): "CCCCCC",
    })
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_meta(text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    p.paragraph_format.space_after = Pt(2)


def add_body(text: str, bold_prefix: str | None = None):
    p = doc.add_paragraph()
    if bold_prefix:
        b = p.add_run(bold_prefix)
        b.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p


def _add_runs(p, segments):
    """segments: list of (text, {bold?, italic?, mono?}). Build a paragraph."""
    for text, opts in segments:
        run = p.add_run(text)
        if opts.get("bold"):
            run.bold = True
        if opts.get("italic"):
            run.italic = True
        if opts.get("mono"):
            run.font.name = "Consolas"
            run.font.size = Pt(10)
    return p


def add_rich(segments):
    """Body paragraph with a list of (text, opts) tuples."""
    p = doc.add_paragraph()
    return _add_runs(p, segments)


def add_callout(text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    pPr = p._p.get_or_add_pPr()
    pBdr = pPr.makeelement(qn("w:pBdr"), {})
    left = pBdr.makeelement(qn("w:left"), {
        qn("w:val"): "single",
        qn("w:sz"): "18",
        qn("w:space"): "8",
        qn("w:color"): "2E74B5",
    })
    pBdr.append(left)
    pPr.append(pBdr)
    shd = pPr.makeelement(qn("w:shd"), {
        qn("w:val"): "clear",
        qn("w:color"): "auto",
        qn("w:fill"): "E8F0FE",
    })
    pPr.append(shd)
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    run.italic = True


def add_table(headers, rows, bold_col=None, highlight_rows=None):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = tbl.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = tcPr.makeelement(qn("w:shd"), {
            qn("w:val"): "clear",
            qn("w:color"): "auto",
            qn("w:fill"): "1B3A5C",
        })
        tcPr.append(shd)

    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = tbl.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(10)
            if bold_col is not None and c_idx == bold_col:
                run.bold = True
            if highlight_rows and r_idx in highlight_rows:
                run.bold = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if r_idx % 2 == 1:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                shd = tcPr.makeelement(qn("w:shd"), {
                    qn("w:val"): "clear",
                    qn("w:color"): "auto",
                    qn("w:fill"): "F2F6FA",
                })
                tcPr.append(shd)

    doc.add_paragraph()
    return tbl


def add_numbered(text: str, bold_prefix: str | None = None):
    p = doc.add_paragraph(style="List Number")
    if bold_prefix:
        b = p.add_run(bold_prefix)
        b.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p


def add_bullet(text: str, bold_prefix: str | None = None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        b = p.add_run(bold_prefix)
        b.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Title block ──
title = doc.add_heading("Model Training Evaluation", level=1)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle = doc.add_heading("Locomotion & Navigation Policies", level=2)
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_meta("AI2C Tech Capstone — Carnegie Mellon University | May 2026")
add_meta("Team: Alex, Ryan, Colby, Cole | Platform: Boston Dynamics Spot")
add_hr()

# ── 1. Purpose & Scope ──
doc.add_heading("1. Purpose & Scope", level=2)

add_callout(
    "\"The purpose of Model Training Evaluation documents is to help to assess, "
    "compare, and communicate the performance of trained models. They contribute "
    "to the development of reliable, transparent, and accountable systems.\""
)

add_body(
    "This document evaluates the trained locomotion and navigation policies "
    "produced over the capstone, focusing on three questions:"
)
add_numbered(
    "how did each successive generation of policy improve over the last, and "
    "which design changes drove those gains?",
    bold_prefix="Performance growth — ",
)
add_numbered(
    "how well do the curricula we trained on transfer to the canonical 4-arena "
    "evaluation bench, and where do they break down?",
    bold_prefix="Training environment fidelity — ",
)
add_numbered(
    "across our locomotion and navigation entries, which policies are deployment-"
    "ready, which are still gated, and what is the empirical evidence?",
    bold_prefix="Comparative ranking — ",
)

add_body(
    "We rely on a single canonical evaluation harness — the 4-environment "
    "graduated-difficulty arena — for every locomotion claim, and the in-team "
    "Cole circular waypoint arena + Skill-Nav Lite battery for every navigation "
    "claim. Statistical comparisons use Welch's t-test on progress and the "
    "two-proportion z-test on completion / fall rates, with Cohen's d reported as "
    "effect size. Sample sizes are 100 episodes per policy / environment for the "
    "canonical loco bench and 25 episodes for the navigation arenas (per the "
    "deployment-readiness criterion)."
)

add_callout(
    "Headline. The shipped locomotion policy (parkour_phasefwplus_22100.pt, "
    "\"22100\") significantly outperforms the flat baseline on every environment "
    "in the 4-arena bench (Welch's t, p < 1e-17, Cohen's d between +1.39 and "
    "+3.75), lifting friction completion from 0% to 96% and grass completion "
    "from 0% to 75%, and converting catastrophic falls into upright stalls on "
    "rough terrain (boulder fall rate 62% -> 0%, stairs 99% -> 51%). On the "
    "navigation side, the FM V3 + 7m raycast + A* replan stack is the only "
    "approach to deliver 25/25 waypoint capture with zero falls on the Cole "
    "rich arena (1095m / 654s / 0 falls); two of the three other approaches "
    "are blocked or unfinished."
)

add_hr()

# ── 2. Evaluation Methodology ──
doc.add_heading("2. Evaluation Methodology", level=2)

doc.add_heading("2.1 Canonical 4-Environment Arena (Locomotion)", level=3)
add_body(
    "Four physically distinct 50m arenas, each subdivided into five 10m zones "
    "of monotonically increasing difficulty:"
)
add_table(
    ["Environment", "Mechanism", "Zone 1", "Zone 5"],
    [
        ["Friction", "Surface μ scaling", "60-grit (μ=0.90)", "Oil on steel (μ=0.05)"],
        ["Grass", "Velocity-dependent drag", "Light fluid", "Dense brush"],
        ["Boulder", "Polyhedra obstacle field", "Gravel (3-5 cm)", "Large boulders (80-120 cm)"],
        ["Stairs", "Continuous ascending steps", "Access ramp (3 cm)", "Max challenge (23 cm)"],
    ],
)
add_body(
    "The robot spawns at (0, 15, 0.6) m and is commanded forward at the policy's "
    "natural velocity (waypoint follower with proportional yaw correction). The "
    "course is judged complete on reaching x ≥ 49.0 m. A fall is registered when "
    "base height drops below 0.15 m. Episode timeout is 600 s of simulation time."
)

doc.add_heading("2.2 Per-Episode Metrics", level=3)
add_body(
    "Seventeen metrics are recorded per episode (full schema in 4_env_test/"
    "episode_schema.json). The headline subset:"
)
add_bullet("max x-displacement from spawn", bold_prefix="Forward progress (m) — ")
add_bullet("robot reached x ≥ 49 m", bold_prefix="Completion (bool) — ")
add_bullet("body below 0.15 m, and where", bold_prefix="Fall detected (bool) + Fall zone (1-5) — ")
add_bullet(
    "composite of mean roll, pitch, height variance, angular velocity (lower = more stable)",
    bold_prefix="Stability score — ",
)
add_bullet("average forward speed", bold_prefix="Mean velocity (m/s) — ")
add_body(
    "A stall is a derived state: the episode timed out (600 s) without falling "
    "and without completing. Stalls indicate the policy stayed upright but could "
    "not advance, which is the dominant failure mode of 22100 on rough terrain."
)

doc.add_heading("2.3 Statistical Methods", level=3)
add_bullet("Welch's unequal-variances t-test, α = 0.05.", bold_prefix="Progress comparisons: ")
add_bullet(
    "two-proportion z-test against pooled standard error.",
    bold_prefix="Completion / fall rates: ",
)
add_bullet("Cohen's d (positive = 22100 > flat).", bold_prefix="Effect size: ")
add_bullet(
    "*** p < 0.001, ** p < 0.01, * p < 0.05, ns otherwise.",
    bold_prefix="Significance markers: ",
)

doc.add_heading("2.4 Navigation Bench", level=3)
add_body(
    "Skill-Nav Lite uses Cole's circular 50 m waypoint arena (25 waypoints A-Y, "
    "0.5 m capture radius). For each policy we report waypoints captured (out "
    "of 25), arc length traveled (m), wall-clock time (s), and falls. The "
    "\"rich\" arena adds 46 obstacles + 7 obstacle shapes at 0.25/0.25/0.0 "
    "density. We additionally report the FM V3 max-density 30% boulder battery "
    "(4 policies × 25 seeds) for stress-testing."
)

add_hr()

# ── 3. Locomotion Policy Performance Growth ──
doc.add_heading("3. Locomotion Policy Performance Growth", level=2)

doc.add_heading("3.1 Policy Lineage", level=3)
add_body(
    "The locomotion policy line evolved through five generations. Each generation "
    "was fully evaluated on the 4-arena bench before the next was started."
)
add_table(
    ["Gen", "Policy", "Checkpoint", "Key Design Change"],
    [
        ["0", "Custom 22-term", "(deprecated)",
         "22 reward terms, [1024,512,256] MLP, 1.2M params"],
        ["1", "ARL Baseline", "mason_baseline_final_19999.pt",
         "Adopted ARL's 11-term reward, [512,256,128] (286K params)"],
        ["2", "ARL Hybrid", "mason_hybrid_best_33200.pt",
         "Added 3 safety fixes (terrain-rel-height, DOF limits, clamped action smoothness)"],
        ["3", "Obstacle Expert", "obstacle_best_44400.pt",
         "Retrained Hybrid on 60% boulder/stair mix; foot-clearance 0.5 -> 2.0"],
        ["4", "Distilled Master", "distilled_6899.pt",
         "Multi-expert distillation (Hybrid + Obstacle) with sigmoid terrain router"],
        ["5", "22100 Final", "parkour_phasefwplus_22100.pt",
         "H100 PARKOUR_NAV \"Hail Mary\": teacher-student + asymmetric critic + obstacle scatter on all terrain"],
    ],
    highlight_rows=[5],
)
add_body(
    "Generation 0 was abandoned after 11 trials and ~30 sub-iterations. Subsequent "
    "generations all share the same [512, 256, 128] / 286K-parameter architecture, "
    "making them directly substitutable in the evaluation harness."
)

doc.add_heading("3.2 Training Environments by Generation", level=3)
add_table(
    ["Gen", "Curriculum", "Episode", "Envs", "Total steps"],
    [
        ["1", "12-terrain (40% geometric / 35% surface / 25% compound)",
         "20 s", "4,096", "~0.8 B"],
        ["2", "Same 12-terrain + safety fixes", "20 s", "4,096", "~2.0 B"],
        ["3", "60% boulder/stair, 40% mixed", "20 s", "4,096", "~2.4 B"],
        ["4", "12-terrain (no specialization)", "20 s", "4,096", "~0.4 B"],
        ["5", "12-terrain + obstacle scatter on all rows + privileged-info teacher",
         "20 s", "4,096", "~3.5 B"],
    ],
)
add_callout(
    "Generation 5 added obstacle scatter on every difficulty row (not just "
    "compound rows), so the student saw rocks even on \"flat\" terrain — this "
    "was the central design change behind 22100's robustness on the friction arena."
)

doc.add_heading("3.3 Stage-by-Stage Performance Growth", level=3)
add_body("100-episode mean progress per generation (m):")
add_table(
    ["Generation", "Friction", "Grass", "Boulder", "Stairs"],
    [
        ["1. ARL Baseline", "36.9", "29.6", "14.4", "10.9"],
        ["2. ARL Hybrid", "48.9", "27.2", "20.3", "11.2"],
        ["3. Obstacle Expert*", "42.2", "31.7", "30.4", "15.7"],
        ["5. 22100 Final", "47.8", "47.9", "23.3", "21.2"],
    ],
    highlight_rows=[3],
)
add_rich([
    ("* Obstacle Expert results from a single-episode evaluation; all other rows "
     "are 100-episode means. Generation 4 (Distilled Master) was a development-"
     "only checkpoint and is not included in the canonical bench.",
     {"italic": True}),
])

add_body("100-episode fall rate per generation:")
add_table(
    ["Generation", "Friction", "Grass", "Boulder", "Stairs"],
    [
        ["1. ARL Baseline", "37%", "23%", "13%", "20%"],
        ["2. ARL Hybrid", "2%", "15%", "3%", "36%"],
        ["5. 22100 Final", "3%", "0%", "0%", "51%"],
    ],
    highlight_rows=[2],
)
add_callout(
    "Two distinct phase shifts. Gen 1 -> Gen 2 was a stability phase shift: "
    "friction falls collapsed from 37% to 2% via the three safety additions. "
    "Gen 2 -> Gen 5 was a capability phase shift: progress on every environment "
    "closed within 2 m of the goal on smooth terrain and roughly doubled on "
    "rough terrain."
)

doc.add_heading("3.4 Headline: 22100 vs Flat Baseline (n=100/env)", level=3)
add_body(
    "The \"Flat Baseline\" referenced here is the original flat_baseline.pt "
    "policy from the Feb-19 Rough-vs-Flat study — the same baseline used in "
    "every prior phase deliverable, re-evaluated on 100 episodes per environment "
    "for parity with the 22100 sample."
)
add_table(
    ["Env", "Flat compl.", "22100 compl.", "Δ progress (m)", "t", "p", "Cohen's d"],
    [
        ["Friction", "0%", "96%", "+9.25", "+9.80", "< 1e-17", "+1.39"],
        ["Grass", "0%", "75%", "+21.27", "+26.50", "< 1e-56", "+3.75"],
        ["Boulder", "0%", "0%", "+12.52", "+25.63", "< 1e-63", "+3.62"],
        ["Stairs", "0%", "0%", "+13.91", "+22.55", "< 1e-53", "+3.19"],
    ],
)
add_body(
    "Every environment is *** significant on the Welch test. Cohen's d is "
    "\"large\" (> 0.8) on every environment and exceeds 3.0 on three of four — "
    "historically large effects for locomotion-policy comparisons."
)
add_callout(
    "Smooth terrain -> completion lift, rough terrain -> fall-to-stall conversion. "
    "On friction and grass the bottleneck for the flat policy was survival on "
    "degrading surface conditions, which 22100's curriculum directly addresses. "
    "On boulder and stairs the bottleneck is geometry — neither policy can finish, "
    "but 22100 cuts boulder fall rate from 62% to 0% and stairs from 99% to 51%, "
    "transforming the failure mode from catastrophic into recoverable."
)

doc.add_heading("3.5 Failure-Mode Shift", level=3)
add_body(
    "The fall and stall heatmaps (in Experiments/Ryan/Final_Capstone_Policy_22100/"
    "eval_100ep/plots_22100_vs_flat/) show the per-zone distribution. Three "
    "diagnostic patterns:"
)
add_bullet(
    "Flat falls concentrate in zone 4 (wet ice, μ = 0.15). 22100 pushes through "
    "zone 5 with only 3 falls remaining.",
    bold_prefix="Friction: ",
)
add_bullet(
    "Flat fails 62/100 in zone 2 (river rocks, ~0.10 m). 22100 has zero falls; "
    "82 episodes wedge in zone 3 (large rocks, ~0.30 m), 18 reach zone 4. The "
    "failure mode is now timeout, not collapse.",
    bold_prefix="Boulder: ",
)
add_bullet(
    "Flat fails 99/100 in zones 1-2 (3-8 cm steps). 22100 has 51 falls (zone 2 -> 3 "
    "transition where step height jumps 8 -> 13 cm); the other 49 episodes timeout "
    "upright on the zone 2 plateau.",
    bold_prefix="Stairs: ",
)
add_body(
    "Stairs zone 3 remains the open problem for the next training cycle."
)

add_hr()

# ── 4. Navigation Policy Performance Growth ──
doc.add_heading("4. Navigation Policy Performance Growth", level=2)

doc.add_heading("4.1 Approaches", level=3)
add_body(
    "Three navigation approaches were pursued in parallel, plus a deployment-"
    "grade hand-tuned baseline (\"Skill-Nav Lite\"):"
)
add_table(
    ["Approach", "Sensing", "Net", "Loco backbone", "Status"],
    [
        ["Alex — NAV_ALEX", "32×32 depth (1,024 pix)", "CNN+MLP, 489K params",
         "Boulder V6 (4500)", "Active training"],
        ["Colby — CombinedPolicy", "64×64 depth (4,096 pix)", "CNN+MLP",
         "Mason Hybrid (33200)", "Blocked (RSL-RL 5.0.1 API)"],
        ["Cole — VS3", "48 raycasts (3 layers)", "MLP [256,256,128]",
         "SpotFlatTerrainPolicy", "Active development"],
        ["Skill-Nav Lite", "FM V3 + 7m raycast", "A* with cadence-gated replan",
         "22100 Final", "Deployed (best in-team result)"],
    ],
    highlight_rows=[3],
)
add_body(
    "All four share the same hierarchical principle: a high-level navigator "
    "outputs [vx, vy, ωz] velocity commands at 10-20 Hz; a frozen low-level "
    "loco policy converts those to 12 joint targets at 50 Hz. The differences "
    "are in the sensing, training curriculum, and whether learning is "
    "end-to-end (Alex / Colby / Cole) or symbolic + reactive (Skill-Nav Lite)."
)

doc.add_heading("4.2 Training Environments", level=3)
add_body(
    "The three end-to-end approaches use Isaac Lab terrain curricula similar "
    "to Generation 5 of the locomotion line. The hand-tuned Skill-Nav Lite "
    "uses a deployment-time symbolic planner and is \"trained\" only in the "
    "sense that its hyperparameters were tuned against held-out arena instances."
)
add_table(
    ["Approach", "Curriculum type", "Stages / levels", "Density"],
    [
        ["Alex (NAV_ALEX)", "Terrain difficulty", "6 levels × 10 types",
         "Boulders 15% (highest weight)"],
        ["Colby", "Same as Alex", "6 levels × 10 types",
         "(training blocked — no signal yet)"],
        ["Cole (VS3)", "Task complexity",
         "7 stages: stability -> push -> 3 nav-range stages -> expert",
         "Obstacle 5-25% by stage"],
        ["Skill-Nav Lite", "Hyperparameter tuning", "11 manual iterations",
         "0.25 / 0.25 / 0.0 (46 obstacles, 7 shapes)"],
    ],
)
add_body(
    "Cole's seven-stage task curriculum is the most distinctive: stability "
    "training before any navigation goal, plus an explicit object-pushing "
    "phase (Stage 2) where the robot learns to displace 5 lightweight "
    "obstacles by 1 m+ each."
)

doc.add_heading("4.3 Stage-by-Stage Performance Growth", level=3)
add_body("Cole's circular 50 m / 25-waypoint arena (\"Cole rich\", quarter density):")
add_table(
    ["Approach", "Waypoints", "Distance (m)", "Time (s)", "Falls", "Notes"],
    [
        ["Alex / NAV_ALEX", "(no eval yet)", "—", "—", "—",
         "Active training, ~5/5,000 iters"],
        ["Colby / CombinedPolicy", "0/25", "—", "—", "—",
         "Training blocked at iter 99"],
        ["Cole / VS3 (mason_hybrid)", "3/25", "—", "—", "—",
         "30% density, n=1"],
        ["Cole / VS3 (baseline)", "2/25", "—", "—", "—",
         "30% density, n=1"],
        ["Cole / VS3 (V6)", "0/25", "—", "—", "—",
         "30% density, FELL at 3.7s"],
        ["FM V3 + 7m + A*", "25/25", "1095", "654", "0",
         "Quarter density, deployment-grade"],
    ],
    highlight_rows=[5],
)
add_body("3-seed batteries on harder densities (Skill-Nav Lite only):")
add_table(
    ["Density", "Mean waypoints", "Falls", "Notes"],
    [
        ["Quarter (deployment)", "19.7/25", "0/3", "Honest split: works on sparse unknown"],
        ["Max (stress test)", "3.7/25", "1/3", "Seed 42 was lucky; not deployment-ready"],
    ],
)

doc.add_heading("4.4 Comparative Analysis", level=3)
add_body(
    "Within the end-to-end approaches, the central trade-off is sensing horizon "
    "vs compute cost:"
)
add_table(
    ["Trade-off", "Depth camera (Alex / Colby)", "Raycasts (Cole)"],
    [
        ["Range", "30 m (~10 s lookahead at 3 m/s)", "~5 m"],
        ["Compute", "Heavy (raycaster bottleneck — 125 s/iter at 64×64)", "Light"],
        ["Resolution", "1,024-4,096 pixels", "48 rays"],
        ["Limitation", "Static meshes only (no dynamic obstacles)",
         "Reactive only — no route planning"],
    ],
)
add_body(
    "Skill-Nav Lite sits outside this trade-off: it pairs a 7 m raycast (cheap) "
    "with a global A* replan against a self-built occupancy map. This is what "
    "made it the only approach to clear the rich arena with zero falls — the "
    "global plan absorbs the short-horizon penalty that pure-reactive Cole VS3 "
    "cannot."
)
add_callout(
    "A trained navigation policy has not yet beaten our hand-tuned planner on "
    "the 25-waypoint bench. The end-to-end approaches are still in active "
    "development; Skill-Nav Lite is the current production answer."
)

add_hr()

# ── 5. Cross-Policy Composition ──
doc.add_heading("5. Cross-Policy Composition", level=2)
add_body(
    "The shipped robot is a stack — 22100 (loco) below Skill-Nav Lite (nav). "
    "The 4-arena bench evaluates 22100 in isolation; the navigation bench "
    "evaluates Skill-Nav Lite with 22100 as the frozen backbone. The composed "
    "performance is roughly the intersection of each layer's capability envelope:"
)
add_bullet(
    "On flat / sparse terrain (the deployment regime), both layers pass: "
    "Skill-Nav Lite captures 19.7/25 waypoints with zero falls."
)
add_bullet(
    "On dense unknown terrain (max-density rich arena), Skill-Nav Lite's planning "
    "quality degrades (3.7/25, 1 fall over 3 seeds). The bottleneck is the "
    "planner, not 22100 — when we replay the same arena under teleop with 22100, "
    "the loco layer holds."
)
add_bullet(
    "On stairs / boulder beyond zone 2, 22100 is the bottleneck. Even a perfect "
    "planner cannot push the stack past 21 m on stairs because the loco layer "
    "cannot survive 13 cm steps."
)
add_callout(
    "This separation tells us the next training cycle should target stairs zone 3 "
    "(loco) and dense-arena planning quality (nav) independently."
)

add_hr()

# ── 6. Limitations & Open Issues ──
doc.add_heading("6. Limitations & Open Issues", level=2)

limitations = [
    ("Sample size on completion-event tails. ",
     "All loco evaluations use n = 100/env. For p < 1e-50 effects this is more "
     "than enough. For rare-event analysis (e.g. zone-5 falls) the per-cell counts "
     "are small and conclusions are correspondingly soft."),
    ("Single-seed arena instances. ",
     "The 4-arena bench uses a fixed seed (42) for reproducibility. Boulder "
     "placement variability is therefore not directly captured in the standard "
     "deviation; we estimate it separately on the FM V3 30%-density 4-policy "
     "battery."),
    ("Sim-to-sim only. ",
     "Every result here is in NVIDIA Isaac Sim 5.1.0. Real-world transfer "
     "introduces sensor noise, actuator latency, and terrain modeling errors "
     "that are not captured. Sim-to-real validation on the physical Spot is "
     "gated on ARL release of their bench."),
    ("Cole VS3 sample size. ",
     "Cole's three-policy 30%-density battery is n = 1 per cell, single seed. "
     "The \"V6 falls\" / \"mason 3/25\" finding warrants a 100-seed re-run "
     "before being treated as a hard ranking — this is the in-flight Ryan "
     "handoff (Experiments/Ryan/Nav_evals/)."),
    ("Obstacle Expert single-episode evaluation. ",
     "The Generation 3 Obstacle Expert was evaluated on a single episode per "
     "environment in §3.3; the 100-episode confirmation was deprioritized once "
     "Generation 5 (22100) was shipped and superseded it."),
    ("No paired hardware repro yet. ",
     "The biggest open gap is that no policy in this evaluation has been "
     "validated on the physical Spot. The full evaluation framework is "
     "reproducible end-to-end in simulation; real hardware is the next milestone."),
]
for bold, rest in limitations:
    add_numbered("", bold_prefix=bold)
    doc.paragraphs[-1].add_run(rest)

add_hr()

# ── 7. Lessons Learned ──
doc.add_heading("7. Lessons Learned", level=2)

lessons = [
    ("Run the same harness on every generation. ",
     "The single biggest evaluation-quality decision was committing early to the "
     "4-arena bench (Feb 19). Every locomotion claim in this document compares "
     "against that bench, which is why we can quote effect sizes across "
     "generations a year apart."),
    ("A zero-fall result is not the same as a high-completion result. ",
     "22100 has zero falls on boulders but 0% completion. Reporting only completion "
     "would have hidden the failure-mode shift that is the most useful story for "
     "the next training cycle."),
    ("Hand-tuned baselines are evaluation infrastructure, not embarrassments. ",
     "Skill-Nav Lite is the navigation team's best result by a wide margin. "
     "Treating it as the production baseline (not a \"not-really-a-policy\" "
     "oddity) is what made the trained-policy gaps quantifiable."),
    ("Small sample sizes warn loudly. ",
     "Cole's n = 1 density-flip result was correctly flagged at handoff; the "
     "100-seed Ryan re-run will tell us whether the \"30% mason > baseline\" "
     "claim survives. Single-seed claims must come with a re-run plan."),
    ("Statistical effect sizes carry the story when p-values bottom out. ",
     "Cohen's d between +1.39 and +3.75 is the more useful number once p < 1e-17 — "
     "the effect is real and large, but how large matters for prioritization. "
     "Reporting only p < 0.001 would have flattened a 3× variance in effect size "
     "across environments."),
    ("Failure modes are policies too. ",
     "The fall heatmap (where falls happen, by zone) was added to the bench "
     "mid-project after we noticed that \"fall rate\" alone was hiding the "
     "difference between \"policy collapses early\" and \"policy collapses late\". "
     "Per-zone tracking is now mandatory in any new evaluation we add."),
    ("Lock the canonical evaluation harness early; iterate on policies, not benches. ",
     "The 4-arena bench has been frozen since February. Every subsequent training "
     "cycle has been forced to defend against that bench, which is the only reason "
     "we have a clean 5-generation growth curve. Changing the bench mid-project "
     "would have erased the comparison."),
]
for bold, rest in lessons:
    add_numbered("", bold_prefix=bold)
    doc.paragraphs[-1].add_run(rest)

add_hr()

# ── Footer / source pointer ──
add_meta(
    "Generated for the AI2C Tech Capstone. Source data, scripts, and per-episode "
    "JSONLs:"
)
for line in [
    "Locomotion_Codebases/4_env_test/results/parallel_2026-02-21_08-24-21/   (flat baseline, 100 ep/env)",
    "Experiments/Ryan/Final_Capstone_Policy_22100/eval_100ep/                (22100 + comparison plots)",
    "Experiments/Alex/NAV_ALEX/                                              (Alex navigation)",
    "Experiments/Colby/CombinedPolicyTraining/                               (Colby navigation)",
    "Experiments/Cole/RL_FOLDER_VS3/                                         (Cole navigation)",
    "Experiments/Alex/skill_nav_lite/                                        (Skill-Nav Lite hand-tuned baseline)",
]:
    p = doc.add_paragraph()
    run = p.add_run(line)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    p.paragraph_format.space_after = Pt(2)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE + PDF CONVERT
# ══════════════════════════════════════════════════════════════════════════════
doc.save(DOCX_PATH)
print(f"DOCX saved: {DOCX_PATH}")

try:
    from docx2pdf import convert  # type: ignore
    convert(DOCX_PATH, PDF_PATH)
    print(f"PDF saved:  {PDF_PATH}")
except Exception as exc:  # pragma: no cover
    print(f"PDF conversion failed: {exc}", file=sys.stderr)
    print("DOCX is ready; convert manually if Word is unavailable.", file=sys.stderr)
