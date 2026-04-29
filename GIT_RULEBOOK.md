# Git & AI Collaboration Rulebook

## Immersive Modeling and Simulation for Autonomy — Capstone Team

---

> **Who is this for?** Every team member (Alex, Colby, Cole, Dylan, Ryan, Gabriel, and any AI assistants helping with development). Bookmark this file. Reference it often. If you're an AI model assisting this team, **you must read and follow this document before suggesting any Git operations or code changes.**

---

## Table of Contents

1. [Why This Matters](#1-why-this-matters)
2. [Repository Overview & Current Observations](#2-repository-overview--current-observations)
3. [Initial Cleanup Recommendations](#3-initial-cleanup-recommendations)
4. [Branching Strategy](#4-branching-strategy)
5. [Day-to-Day Git Workflow](#5-day-to-day-git-workflow)
6. [Commit Messages](#6-commit-messages)
7. [Pull Requests & Code Reviews](#7-pull-requests--code-reviews)
8. [Merge Conflicts](#8-merge-conflicts)
9. [Production & Testing Environments](#9-production--testing-environments)
10. [GitHub Actions — CI/CD](#10-github-actions--cicd)
11. [Working with AI Assistants](#11-working-with-ai-assistants)
12. [Managing Large Files (URDF, USD, Meshes, Policies)](#12-managing-large-files-urdf-usd-meshes-policies)
13. [The .gitignore — What Should Never Be Committed](#13-the-gitignore--what-should-never-be-committed)
14. [Tools & Resources](#14-tools--resources)
15. [Troubleshooting](#15-troubleshooting)
16. [Team Etiquette](#16-team-etiquette)
17. [Quick-Reference Cheat Sheet](#17-quick-reference-cheat-sheet)

---

## 1. Why This Matters

You are building reinforcement learning policies for quadrupedal robots (Spot, Vision 60) using NVIDIA Isaac Sim. This involves:

- **Python training scripts** that change frequently.
- **URDF/USD robot models** that are large binary files.
- **Trained policy weights** (`.npz`, `.pt`, `.onnx`) that need versioning.
- **Multiple team members** experimenting simultaneously on an H100 cluster.
- **AI coding assistants** (like GitHub Copilot, ChatGPT, Claude) helping write and debug code.

Without clear Git practices, you will inevitably:

- Overwrite each other's work.
- Lose hours of training results.
- Ship broken code to `main` that blocks the entire team.
- Confuse AI assistants, who will suggest conflicting workflows.

Git is your single source of truth. This rulebook keeps everyone — humans *and* AIs — aligned.

---

## 2. Repository Overview & Current Observations

Your current repo structure looks roughly like this:

```
Immersive-Main-Fresh/            <-- Primary working repo
├── .git/
├── .gitignore                   <-- Currently only ignores isaacSim_env/
├── README.md
├── requirements.txt             <-- 165 pinned Python dependencies
├── Environments/
│   ├── Testing/                 <-- Empty, with .gitkeep
│   └── Training/                <-- Empty, with .gitkeep
└── Results/
```

### Observations & Concerns

| Concern | Detail |
|---------|--------|
| **Minimal `.gitignore`** | Only `isaacSim_env/` is ignored. `__pycache__/`, `.pyc`, `.vscode/`, `*.usd`, `*.npz` checkpoints, and training logs are not excluded. |
| **No branching strategy** | All work appears to land directly on `main`. This is fragile. |
| **No CI/CD** | No `.github/workflows/` directory exists. Nothing prevents broken code from being merged. |
| **Per-person Experiment folders** | The old repo (`Immersive-Modeling-and-Simulation-for-Autonomy`) used `Experiments/Alex/`, `Experiments/Ryan/`, etc. This is fine for sandbox experimentation, but polished code must eventually live in shared directories. |
| **Duplicate repos** | `Immersive-Main-Fresh` and `Immersive-Modeling-and-Simulation-for-Autonomy` appear to be near-identical. Consolidate to one canonical source of truth. |
| **Large binary files in tree** | URDF (`.urdf`), USD (`.usd`), meshes, and policy weights (`.npz`) are checked directly into Git. These should use Git LFS or be stored externally. |
| **`__pycache__/` committed** | Build artifacts and bytecode are tracked. These must be removed and ignored. |

---

## 3. Initial Cleanup Recommendations

Before adopting the workflow below, do a one-time cleanup. **One person** (ideally the repo admin) should perform these steps on a dedicated branch:

### 3.1 Expand the `.gitignore`

Replace the current `.gitignore` with a comprehensive version:

```gitignore
# ──────────────────── Python ────────────────────
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
*.egg
dist/
build/
*.whl

# ──────────────────── Virtual Environments ────────────────────
isaacSim_env/
venv/
.venv/
env/

# ──────────────────── IDE / Editor ────────────────────
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# ──────────────────── Isaac Sim / NVIDIA ────────────────────
# Large local caches produced by Isaac Sim
_isaac_sim/
*.log
*.bak

# ──────────────────── Training Artifacts ────────────────────
# (Track these with Git LFS instead — see Section 12)
# Uncomment if NOT using Git LFS:
# *.npz
# *.pt
# *.onnx
# *.usd

# ──────────────────── OS / System ────────────────────
*.tmp
*.bak
*.swp
desktop.ini

# ──────────────────── Secrets ────────────────────
.env
*.pem
*.key
```

### 3.2 Remove already-tracked artifacts

After updating `.gitignore`, remove files that should no longer be tracked:

```bash
# Remove cached Python bytecode from tracking
git rm -r --cached **/__pycache__
git commit -m "chore: remove __pycache__ from tracking"
```

### 3.3 Set up Git LFS (see Section 12)

---

## 4. Branching Strategy

We use a simplified **Git Flow** model with three long-lived branches and short-lived feature branches.

```
main          ← Production-ready code. Protected. Deploys run from here.
  │
  └── develop ← Integration branch. All features merge here first.
        │
        ├── feature/ryan-obstacle-env
        ├── feature/colby-vision60-eureka
        ├── feature/alex-spot-urdf-update
        └── fix/training-crash-nan-rewards
```

### Branch Naming Conventions

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New functionality | `feature/dylan-rough-terrain-env` |
| `fix/` | Bug fixes | `fix/spot-joint-limit-crash` |
| `experiment/` | Sandbox / exploratory work | `experiment/cole-reward-shaping` |
| `docs/` | Documentation only | `docs/update-readme-install` |
| `chore/` | Tooling, CI, cleanup | `chore/add-github-actions` |

### Rules

1. **Never commit directly to `main`.** Always go through a Pull Request from `develop`.
2. **Never commit directly to `develop` for non-trivial changes.** Use a feature branch and open a PR into `develop`.
3. **Delete branches after merging.** Keep the branch list clean.
4. **One concern per branch.** Don't mix a new environment with a bug fix.

### Creating a Branch

```bash
# Make sure you're starting from the latest develop
git checkout develop
git pull origin develop

# Create and switch to your feature branch
git checkout -b feature/your-name-short-description
```

---

## 5. Day-to-Day Git Workflow

Here's the full cycle from starting work to getting it merged.

### Step 1 — Sync before you start

```bash
git checkout develop
git pull origin develop
```

### Step 2 — Create a feature branch

```bash
git checkout -b feature/ryan-cluttered-env-v2
```

### Step 3 — Do your work

Edit files, write code, run experiments. Commit frequently (see Section 6).

### Step 4 — Stage and commit

```bash
# Stage specific files (preferred over `git add .`)
git add Environments/Training/cluttered_env_v2.py
git add Environments/Testing/test_cluttered_v2.py

# Commit with a descriptive message
git commit -m "feat: add cluttered terrain v2 with random box obstacles"
```

### Step 5 — Stay up to date with develop

Before pushing, pull the latest changes to avoid stale conflicts:

```bash
git pull origin develop --rebase
```

> **What does `--rebase` do?** It replays your commits on top of the latest `develop`, producing a cleaner history than a merge commit.

### Step 6 — Push your branch

```bash
git push origin feature/ryan-cluttered-env-v2
```

### Step 7 — Open a Pull Request

Go to GitHub and open a PR from `feature/ryan-cluttered-env-v2` → `develop`. Fill in the PR template (see Section 7).

### Step 8 — Address reviews, then merge

After approval, merge via **"Squash and merge"** to keep history clean.

### Step 9 — Clean up

```bash
git checkout develop
git pull origin develop
git branch -d feature/ryan-cluttered-env-v2
```

---

## 6. Commit Messages

Good commit messages make it possible to understand the project's history months later. Use the **Conventional Commits** format:

```
<type>(<scope>): <short description>

<optional longer description>
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | Adding new functionality (a new environment, a new reward function) |
| `fix` | Fixing a bug |
| `refactor` | Restructuring code without changing behavior |
| `docs` | Documentation changes |
| `chore` | Tooling, CI/CD, dependency updates |
| `test` | Adding or updating tests |
| `experiment` | Experimental / exploratory work |

### Examples

```bash
# Good
git commit -m "feat(env): add 50m cluttered room with random box obstacles"
git commit -m "fix(training): clamp reward to prevent NaN in policy gradient"
git commit -m "docs: update README with H100 SSH instructions"
git commit -m "chore: add pre-commit hooks for flake8"
git commit -m "experiment(reward): test velocity-weighted reward shaping"

# Bad — don't do these
git commit -m "stuff"
git commit -m "changes"
git commit -m "fix"
git commit -m "asdfasdf"
git commit -m "WIP"  # (commit properly or use git stash)
```

### Rules

- **Keep the subject line under 72 characters.**
- **Use the imperative mood:** "add feature" not "added feature."
- **Reference issues when relevant:** `fix(env): correct spawn height — closes #42`
- **Don't commit half-done work.** Use `git stash` to save work-in-progress.

---

## 7. Pull Requests & Code Reviews

### Opening a PR

Every PR should include:

```markdown
## What does this PR do?
<!-- One-paragraph summary -->

## Type of change
- [ ] New feature
- [ ] Bug fix
- [ ] Refactor
- [ ] Documentation
- [ ] Experiment

## How was this tested?
<!-- Describe how you verified the change works -->
<!-- e.g., "Ran headless training for 500 steps, policy converged" -->

## Screenshots / Logs (if applicable)

## Checklist
- [ ] My code follows the project conventions
- [ ] I have updated requirements.txt if dependencies changed
- [ ] I have not committed __pycache__, .pyc, or virtual env files
- [ ] I have run the code and it executes without errors
- [ ] Large files (URDF, USD, policy weights) use Git LFS
```

### Reviewing a PR

- **Be kind and constructive.** We're all learning.
- **Test the code locally** when possible. `git checkout` the branch and run it.
- **Approve or request changes** — don't let PRs sit for more than 48 hours.
- **Look for:** broken imports, hardcoded paths, missing `.gitignore` entries, large binary files.

### Merge Policy

| Target branch | Required approvals | CI must pass? |
|---------------|-------------------|---------------|
| `develop` | At least 1 teammate | Yes |
| `main` | At least 2 teammates | Yes |

---

## 8. Merge Conflicts

Merge conflicts happen when two people edit the same lines. Don't panic.

### How to resolve

```bash
# You're on your feature branch and want to update from develop
git pull origin develop --rebase

# Git will stop and show conflicts. Open the conflicted file(s).
# You'll see markers like:
<<<<<<< HEAD
your_code_here
=======
their_code_here
>>>>>>> develop

# Edit the file to keep the correct version (or combine both).
# Remove ALL conflict markers (<<<, ===, >>>).

# Stage the resolved file
git add <resolved-file>

# Continue the rebase
git rebase --continue
```

### Prevention Tips

- **Pull often.** Don't let your branch drift far from `develop`.
- **Communicate.** If two people need to edit the same file, coordinate.
- **Keep branches short-lived.** Merge within a few days, not weeks.

### When in doubt

```bash
# Abort and start over — no harm done
git rebase --abort
```

---

## 9. Production & Testing Environments

### Branch-to-Environment Mapping

```
main      →  Production   (stable, tested, "it works" code)
develop   →  Staging      (integration testing, may have rough edges)
feature/* →  Development  (individual work, anything goes)
```

### `main` (Production)

- Contains only code that has been **tested and reviewed**.
- The `Environments/Training/` and `Environments/Testing/` directories here represent the team's best, validated environments.
- `Results/` here stores final, publication-ready results.
- **Protected branch:** no direct pushes, requires PR + CI checks + 2 approvals.

### `develop` (Staging / Integration)

- Where all feature branches merge first.
- Team members pull from here to start new work.
- CI runs automated checks on every push (see Section 10).
- Once a milestone is stable on `develop`, a PR is opened into `main`.

### Feature branches (Development)

- Your personal sandbox. Experiment freely.
- Name them clearly: `experiment/colby-eureka-reward-v3`.
- Don't expect these to be stable — that's what `develop` and `main` are for.

### Directory Conventions

```
Environments/
├── Training/          <-- RL training environment scripts
│   ├── spot_cluttered_v1.py
│   ├── spot_cluttered_v2.py
│   └── vision60_flat.py
├── Testing/           <-- Evaluation / validation scripts
│   ├── test_spot_cluttered_v1.py
│   └── test_vision60_flat.py
Results/
├── spot_cluttered_v1/
│   ├── training_log.csv
│   ├── reward_curve.png
│   └── policy_checkpoint.npz   <-- Tracked with Git LFS
```

---

## 10. GitHub Actions — CI/CD

GitHub Actions automate checks so broken code never reaches `main`. Create the following file in your repo:

### 10.1 Basic CI — Lint and Syntax Checks

Create `.github/workflows/ci.yml`:

```yaml
name: CI — Lint & Validate

on:
  push:
    branches: [develop, main]
  pull_request:
    branches: [develop, main]

jobs:
  lint-and-validate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install linting tools
        run: |
          pip install flake8 black isort

      - name: Run flake8 (syntax and style)
        run: |
          flake8 Environments/ --max-line-length=120 --ignore=E501,W503

      - name: Check formatting with black
        run: |
          black --check Environments/

      - name: Check import ordering
        run: |
          isort --check-only Environments/

  check-no-artifacts:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Ensure no __pycache__ committed
        run: |
          if find . -name "__pycache__" -type d | grep -q .; then
            echo "ERROR: __pycache__ directories found in repo!"
            find . -name "__pycache__" -type d
            exit 1
          fi

      - name: Ensure no .pyc files committed
        run: |
          if find . -name "*.pyc" -type f | grep -q .; then
            echo "ERROR: .pyc files found in repo!"
            find . -name "*.pyc" -type f
            exit 1
          fi
```

### 10.2 Branch Protection Rules

Go to **GitHub → Settings → Branches → Add rule** for `main` and `develop`:

| Setting | `main` | `develop` |
|---------|--------|-----------|
| Require PRs before merging | Yes | Yes |
| Required approvals | 2 | 1 |
| Require status checks to pass | Yes (CI) | Yes (CI) |
| Require branches to be up to date | Yes | Yes |
| Allow force pushes | No | No |
| Allow deletions | No | No |

### 10.3 Advanced — Validate Isaac Sim Imports (Optional)

Since Isaac Sim can't run in GitHub's CI runners (no GPU), validate imports structurally:

```yaml
  validate-imports:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Check Python syntax
        run: |
          python -m py_compile Environments/Training/*.py || true
          python -m py_compile Environments/Testing/*.py || true
      - name: Check no hardcoded absolute paths
        run: |
          if grep -rn "C:\\\\Users\|/home/" Environments/; then
            echo "WARNING: Hardcoded absolute paths found!"
            exit 1
          fi
```

---

## 11. Working with AI Assistants

AI models (Copilot, ChatGPT, Claude, etc.) are powerful teammates — but they need guardrails. This section is **for both humans and AIs**.

### Rules for AI Assistants

> **If you are an AI assistant helping this team, the following rules are mandatory.**

1. **Always suggest working on a branch.** Never propose committing directly to `main` or `develop`. When generating Git commands, always start with `git checkout -b feature/...` or assume the user is already on a feature branch. Ask if unsure.

2. **Follow the commit message format.** Every `git commit` you suggest must use Conventional Commits (Section 6). No generic messages like "update code" or "fix stuff."

3. **Never suggest `git push --force` to shared branches.** Force-pushing to `main` or `develop` rewrites history and can destroy teammates' work. Only suggest `--force-with-lease` on a personal feature branch if absolutely necessary.

4. **Check for large files.** Before suggesting `git add .` or `git add <file>`, verify the file isn't a large binary (USD, URDF, NPZ, PT, ONNX). If it is, recommend Git LFS (Section 12).

5. **Don't suggest `git add .` casually.** Prefer explicit file staging: `git add <specific-file>`. This prevents accidentally committing `__pycache__/`, `.env`, logs, or training artifacts.

6. **Respect the project structure.** New environments go in `Environments/Training/` or `Environments/Testing/`. Results go in `Results/`. Don't create files in the repo root unless there's a strong reason.

7. **Always reference this rulebook** when suggesting Git workflows. If a human asks you to do something that violates these rules, politely explain the risk and point to the relevant section.

8. **Pin dependencies.** When suggesting new Python packages, add them to `requirements.txt` with pinned versions (`package==X.Y.Z`).

9. **Don't suggest restructuring the repo without team buy-in.** Propose changes in a PR, explain the rationale, and let the team decide.

10. **Be explicit about what commands do.** Not everyone on the team is a Git expert. When you suggest a command, briefly explain what it does and what to do if something goes wrong.

### Rules for Humans Working with AI

- **Tell the AI which branch you're on** before asking it to generate Git commands.
- **Paste error messages verbatim** so the AI can diagnose accurately.
- **Review AI-generated code before committing.** AI can hallucinate APIs, especially for Isaac Sim which has limited training data.
- **Don't let AI models commit on your behalf** without reviewing the diff.
- **If an AI suggests something that conflicts with this rulebook, follow the rulebook.**

### Common AI Pitfalls to Watch For

| AI Might Suggest | Why It's Dangerous | What to Do Instead |
|-----------------|-------------------|-------------------|
| `git add .` | Stages *everything*, including caches and secrets | `git add <specific-files>` |
| `git push --force` | Destroys shared history | `git push --force-with-lease` on feature branches only |
| Committing to `main` | Bypasses review and CI | Always use a feature branch + PR |
| `pip install <package>` without updating requirements.txt | Other team members won't have the dependency | `pip install <pkg> && pip freeze > requirements.txt` or add manually |
| Creating files in repo root | Clutters the project | Follow directory conventions |
| Suggesting SSH keys, tokens, or passwords in code | Security risk | Use environment variables or `.env` files (which are git-ignored) |

---

## 12. Managing Large Files (URDF, USD, Meshes, Policies)

Your project includes files that are too large for standard Git:

- **URDF files** (robot descriptions): `spirit_gazebo.urdf`, `vision60_v5.urdf`
- **USD files** (simulation scenes): `spirit40_fixed.usd`, `spirit40_v2.usd`
- **Policy weights**: `spot_nav_policy.npz` and future `.pt` / `.onnx` files
- **Mesh files**: any `.obj`, `.stl`, `.dae` associated with robots

### Why This Matters

Git stores **every version** of every file forever. A 50 MB USD file edited 10 times = 500 MB of repo bloat. This makes cloning, pulling, and pushing painfully slow.

### Solution: Git LFS (Large File Storage)

Git LFS replaces large files in your repo with lightweight pointers, storing the actual content on a remote server.

#### Setup (One Time Per Machine)

```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.usd"
git lfs track "*.urdf"
git lfs track "*.npz"
git lfs track "*.pt"
git lfs track "*.onnx"
git lfs track "*.stl"
git lfs track "*.obj"
git lfs track "*.dae"
git lfs track "*.fbx"

# This creates/updates a .gitattributes file — commit it!
git add .gitattributes
git commit -m "chore: configure Git LFS for large model and asset files"
```

#### Verification

```bash
# Check what's being tracked by LFS
git lfs ls-files

# Check tracking rules
cat .gitattributes
```

#### Important Notes

- Git LFS is **free up to 1 GB of storage and 1 GB/month bandwidth** on GitHub.
- If you exceed this, GitHub will prompt you to purchase additional data packs.
- **Alternative:** For very large assets (training datasets, full simulation scenes), consider storing them externally (Google Drive, S3, university file share) and documenting the location in the README.

---

## 13. The `.gitignore` — What Should Never Be Committed

Rule of thumb: **If it can be regenerated, don't commit it.**

| Category | Files/Patterns | Why |
|----------|---------------|-----|
| Python bytecode | `__pycache__/`, `*.pyc`, `*.pyo` | Generated on import; OS-specific |
| Virtual environments | `isaacSim_env/`, `venv/`, `.venv/` | Large, machine-specific |
| IDE settings | `.vscode/`, `.idea/` | Personal preferences |
| OS files | `.DS_Store`, `Thumbs.db`, `desktop.ini` | OS-generated junk |
| Secrets | `.env`, `*.pem`, `*.key` | Security risk |
| Training logs (large) | `wandb/`, `tensorboard/`, `outputs/` | Can be enormous; track selectively |
| Isaac Sim caches | `_isaac_sim/`, `*.log` | Local runtime artifacts |

---

## 14. Tools & Resources

### Recommended Git Clients

| Tool | Best For | Notes |
|------|----------|-------|
| **VS Code (built-in Git)** | Everyone | You're already using it. Source Control panel (Ctrl+Shift+G) shows diffs, staging, commits. Highly recommended. |
| **GitHub Desktop** | Git beginners | Visual interface for branching, committing, PRs. Great for learning. |
| **Git Bash / Terminal** | Power users | Full control. All commands in this doc work here. |
| **GitLens (VS Code Extension)** | Understanding history | Shows who changed each line, branch visualization, commit search. |

### VS Code Integration Tips

- **Open the Source Control panel** (`Ctrl+Shift+G`): see changed files, stage, commit, push.
- **Use the built-in terminal** (`Ctrl+`` `): run Git commands without leaving VS Code.
- **Install GitLens**: right-click any line to see its commit history.
- **Use the diff viewer**: click any changed file in Source Control to see exactly what changed.

### Useful Links

- [Git Handbook (GitHub)](https://guides.github.com/introduction/git-handbook/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [NVIDIA Isaac Sim Docs](https://developer.nvidia.com/isaac-sim)

---

## 15. Troubleshooting

### "I accidentally committed to `main`!"

```bash
# If you haven't pushed yet:
git reset HEAD~1          # Undo the commit (keeps your changes)
git checkout -b feature/my-fix   # Move changes to a branch
git add . && git commit -m "feat: my changes"
git push origin feature/my-fix

# If you already pushed — talk to the team before doing anything.
# You may need a revert:
git revert HEAD
git push origin main
```

### "I committed a huge file and now push is slow or failing"

```bash
# Remove the file from Git history (careful — this rewrites history)
git filter-branch --tree-filter 'rm -f path/to/huge-file.usd' HEAD

# Or use the cleaner BFG tool:
# https://rtyley.github.io/bfg-repo-cleaner/
bfg --delete-files huge-file.usd
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force-with-lease
```

> **Then set up Git LFS** (Section 12) so it doesn't happen again.

### "I have merge conflicts and I'm lost"

```bash
# Option 1: Abort and start fresh
git merge --abort    # or git rebase --abort

# Option 2: Use VS Code's merge editor
# VS Code highlights conflicts and gives Accept Current / Accept Incoming buttons.
# Use them. Resolve each conflict. Then:
git add <resolved-file>
git rebase --continue   # or git merge --continue
```

### "My local repo is totally messed up"

```bash
# Nuclear option — re-clone (you won't lose anything on GitHub)
cd ..
git clone https://github.com/YOUR-ORG/Immersive-Main-Fresh.git
```

> **Seriously, don't be afraid to re-clone.** It's often faster than untangling a mess.

### "`git pull` says 'divergent branches'"

```bash
# This usually means your local and remote have different histories.
# The safest fix:
git pull origin develop --rebase
```

### "I need to undo my last commit but keep the changes"

```bash
git reset --soft HEAD~1
# Your changes are now staged but uncommitted.
```

---

## 16. Team Etiquette

1. **Communicate before big changes.** Post in the team chat: "I'm refactoring the training loop — heads up."
2. **Review PRs promptly.** Don't let them rot. Aim for <48 hours.
3. **Don't rewrite shared history.** No `git push --force` on `main` or `develop`. Ever.
4. **Clean up after yourself.** Delete merged branches. Remove stale experiments.
5. **Document your experiments.** A training run without notes is a wasted training run. At minimum, record:
   - What you changed and why.
   - Hyperparameters used.
   - Results (reward curves, success rates, training time).
6. **Ask for help, not forgiveness.** If you're unsure about a Git operation, ask a teammate or your AI assistant (with this rulebook as context).
7. **Credit your AI.** If an AI assistant wrote significant code, note it in the commit message: `feat(env): add obstacle randomizer (co-authored with Copilot)`.

---

## 17. Quick-Reference Cheat Sheet

### Starting New Work

```bash
git checkout develop && git pull origin develop
git checkout -b feature/your-name-description
```

### Saving Progress

```bash
git add <specific-files>
git commit -m "feat(scope): describe change"
```

### Sharing Your Work

```bash
git pull origin develop --rebase
git push origin feature/your-name-description
# Then open a PR on GitHub: feature branch → develop
```

### Staying Up to Date

```bash
git checkout develop
git pull origin develop
git checkout feature/your-branch
git rebase develop
```

### Undoing Mistakes

```bash
git reset --soft HEAD~1     # Undo last commit, keep changes
git stash                   # Temporarily shelve changes
git stash pop               # Bring them back
git rebase --abort          # Cancel a messy rebase
```

### Quick Status Checks

```bash
git status                  # What's changed?
git log --oneline -10       # Last 10 commits
git branch -a               # All branches
git diff                    # Unstaged changes
git diff --staged           # Staged changes
```

---

## Appendix A: Setting Up the Repo from Scratch

If the team decides to consolidate into `Immersive-Main-Fresh` as the single canonical repo, here's a checklist:

- [ ] Create `develop` branch from `main`
- [ ] Update `.gitignore` (Section 3.1)
- [ ] Remove tracked `__pycache__/` and `.pyc` files (Section 3.2)
- [ ] Set up Git LFS and migrate large files (Section 12)
- [ ] Add `.github/workflows/ci.yml` (Section 10.1)
- [ ] Configure branch protection rules on GitHub (Section 10.2)
- [ ] Create PR template at `.github/pull_request_template.md` (Section 7)
- [ ] Move validated experiment code into `Environments/Training/` and `Environments/Testing/`
- [ ] Store policy weights in `Results/<experiment-name>/`
- [ ] Ensure everyone has pulled the clean state: `git pull origin develop`

---

## Appendix B: PR Template File

Save this as `.github/pull_request_template.md` in the repo:

```markdown
## What does this PR do?
<!-- One-paragraph summary -->

## Type of change
- [ ] New feature (new environment, reward function, training script)
- [ ] Bug fix
- [ ] Refactor / cleanup
- [ ] Documentation
- [ ] Experiment (exploratory, not for main)

## How was this tested?
<!-- How did you verify? Headless run? Visual check? Number of training steps? -->

## Results (if applicable)
<!-- Paste reward curves, success rates, screenshots, or logs -->

## Checklist
- [ ] Code follows project conventions
- [ ] requirements.txt updated if dependencies changed
- [ ] No __pycache__, .pyc, or virtual env files committed
- [ ] Large files (URDF, USD, policy weights) tracked with Git LFS
- [ ] Code runs without errors (at least `python -m py_compile <file>`)
- [ ] Commit messages follow Conventional Commits format
```

---

*Last updated: February 2026*
*Maintained by the Immersive Modeling & Simulation for Autonomy Capstone Team*
