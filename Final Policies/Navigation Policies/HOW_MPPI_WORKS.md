# How MPPI Works — Plain English Guide

**MPPI = Model Predictive Path Integral**

This document explains what MPPI is, how it works conceptually, and how those
concepts map directly to the code in `mppi_navigator.py`.

---

## The Problem MPPI Solves

The robot is somewhere in the arena. It needs to get to a waypoint without
hitting obstacles. Every 0.05 seconds (20 Hz) it must decide:

> *"How fast should I go forward, and how much should I turn?"*

MPPI answers that question without any neural network, training data, or
pre-built map. It only needs to know:
- Where am I right now?
- Where is the goal?
- Where are the obstacles?

---

## The Core Idea — The Ice Cream Shop Analogy

Imagine you're blindfolded in a parking lot and need to reach an ice cream shop.
You have 512 friends who each try a slightly different path from where you are
standing. After 1.25 seconds, you look at where everyone ended up:

- Friends who got **close to the shop** and **avoided the cars** → their paths were **good**
- Friends who **walked into a car** or ended up **facing the wrong way** → their paths were **bad**

You don't just copy the single best friend's first step. Instead, every friend
gets a **vote**, and friends with better paths get **louder votes** (higher
weight). Your actual first step is the **weighted average** of all 512 votes.

That's MPPI.

---

## The Five Steps (Every 0.05 Seconds)

### Step 1 — Sample 512 Random Plans

The robot imagines **512 possible futures**. Each future is a sequence of
25 commands (speed + turn rate), randomly varied from a starting guess.

```
Future 1:  [fast+left, fast+straight, medium+right, ...]   25 steps
Future 2:  [slow+right, fast+left, fast+straight, ...]     25 steps
...
Future 512:[medium+straight, medium+straight, ...]          25 steps
```

The randomness is Gaussian noise — most futures are close to the previous best
plan, but some explore more aggressively:

```python
# mppi_navigator.py line ~165
eps = np.random.randn(512, 25, 3)   # 512 futures × 25 steps × 3 commands
eps *= self._sigma                  # scale noise: vx=0.8, vy=0.0, omega=0.25
U = self._nominal + eps             # perturb around previous best plan
```

### Step 2 — Simulate Each Future on Paper

For every future, the robot runs a pretend simulation of itself moving through
space. No physics engine — just simple math:

```
new_x   = x + (vx × cos(yaw)) × dt
new_y   = y + (vx × sin(yaw)) × dt
new_yaw = yaw + omega × dt
```

This is called a **unicycle model** — it describes a robot that can go forward
and turn, but not slide sideways (like a bicycle).

```python
# mppi_navigator.py line ~240
x   += (vx * cos_y - vy * sin_y) * self.dt
y   += (vx * sin_y + vy * cos_y) * self.dt
yaw += omega * self.dt
```

All 512 simulations run at the same time using NumPy arrays — no loop over
each future individually.

---

### Step 3 — Score Each Future (The Cost Function)

After simulating 25 steps, each future gets a **score** called a **cost**.
**Lower cost = better path.** Three things add cost:

---

#### Cost 1 — Heading Error (Are you facing the goal?)

At every step of the simulation, the robot checks: *"Am I pointed toward the
goal?"* If not, that future gets penalised.

```python
# mppi_navigator.py line ~248
desired_yaw = arctan2(goal_y - y, goal_x - x)   # direction to goal
h_err = angle_difference(desired_yaw, yaw)        # how far off am I?
costs += w_heading * h_err²                       # quadratic penalty
```

A path that spends time pointed sideways pays more than one that stays
aimed at the goal. Weight: **w_heading = 1.5**

---

#### Cost 2 — Obstacle Proximity (Are you about to hit something?)

At every step, the robot checks its distance to every obstacle. If it gets
closer than the safety margin (`r_safe = 0.4 m`), the cost spikes:

```python
# mppi_navigator.py line ~254
clearance   = distance_to_obstacle - obstacle_radius
penetration = max(0,  r_safe - clearance)   # 0 if safely away
costs += w_obs * penetration²               # quadratic spike if close
```

This is a **soft wall** — the robot doesn't hard-stop at the obstacle edge,
it sees a cost gradient that rises steeply as it approaches. Weight: **w_obs = 80.0**
(highest weight — obstacle avoidance matters most)

---

#### Cost 3 — Terminal Goal Distance (Where did you end up?)

After all 25 steps, how far is the imaginary robot from the waypoint?
This is the biggest single cost signal — it drives the robot toward the goal.

```python
# mppi_navigator.py line ~272
costs += w_goal * distance(final_position, goal)
```

Weight: **w_goal = 20.0**

There is also a small **arena boundary cost** that penalises futures that
wander outside the 25 m arena.

---

### Step 4 — Vote on the Best Plan

Instead of picking the single cheapest future (which would be noisy and jumpy),
MPPI turns costs into **votes** using an exponential:

$$\text{weight}_k = \frac{e^{-J_k / \lambda}}{\sum_i e^{-J_i / \lambda}}$$

Where:
- $J_k$ = total cost of future $k$
- $\lambda$ = temperature = **0.03** (lower = more aggressive selection)

Futures with low cost get weights close to 1. Futures with high cost get
weights close to 0. All weights sum to 1.

```python
# mppi_navigator.py line ~175
costs -= costs.min()                        # stability trick
weights = exp(-costs / temperature)
weights /= weights.sum()
```

The final command is the **weighted average** of all 512 futures' first step:

```python
# mppi_navigator.py line ~180
optimal = einsum("k,kth->th", weights, U)  # weighted sum over K futures
```

---

### Step 5 — Warm Start (Remember for Next Time)

The winning plan covers 25 steps. MPPI saves it and uses it as the starting
point for the next call — just shifted forward by one step. This keeps the
robot's planned path smooth across calls.

```python
# mppi_navigator.py line ~184
self._nominal = roll(optimal, -1, axis=0)   # shift plan forward one step
self._nominal[-1] = optimal[-1]             # repeat last step at the end
```

Without this, the robot would start from a blank slate every 0.05 seconds and
might produce jumpy or inconsistent commands.

---

## Summary Table

| Step | What Happens | Code Location |
|---|---|---|
| 1 — Sample | Generate 512 random plans | `eps = randn(512, 25, 3)` |
| 2 — Simulate | Roll each plan forward with unicycle math | `_rollout()` method |
| 3 — Score | Add cost for wrong heading, obstacles, goal distance | `costs +=` lines in `_rollout()` |
| 4 — Vote | Convert costs to weights; take weighted-average first step | `weights = exp(-costs / λ)` |
| 5 — Warm start | Save the plan; shift it forward for next call | `self._nominal = roll(...)` |

---

## Why No Training or Neural Network?

RL trains a neural network over thousands of episodes to learn a policy. That
policy then runs instantly but can only handle situations similar to what it
was trained on.

MPPI never trains. Every call it solves from scratch using math. The
**cost function** is the designer's knowledge — by setting `w_obs = 80` you are
directly telling the algorithm "obstacle avoidance is important." There is no
data collection, no gradient descent, no checkpoint file.

| | RL Policy | MPPI |
|---|---|---|
| Needs training? | Yes (thousands of episodes) | No |
| Needs a checkpoint file? | Yes | No |
| Can handle new obstacles it never saw? | Maybe | Yes — always |
| Compute per step | ~0.1 ms (inference only) | ~8 ms (fresh solve) |
| Interpretable? | No (inside a neural net) | Yes (cost weights are readable) |

---

## The Three Cost Weights — How to Tune Them

The weights in `mppi_navigator.py` control what the robot cares about most.

| Weight | Default | Effect of raising it |
|---|---|---|
| `w_goal = 20.0` | Medium | Robot rushes toward waypoint more directly |
| `w_heading = 1.5` | Low | Robot aligns with goal direction more aggressively before moving |
| `w_obs = 80.0` | High | Robot gives obstacles a wider berth |
| `w_bound = 50.0` | Medium-high | Robot avoids the arena edge more strongly |

If the robot clips obstacles → raise `w_obs` or `r_safe`.
If the robot meanders → raise `w_goal`.
If the robot oscillates left-right → lower `sigma_omega` or raise `w_heading`.

---

## One-Line Summary

> MPPI imagines 512 possible futures, scores them on how good they are, and
> takes the weighted-best first step — every 0.05 seconds, from scratch,
> with no training required.
