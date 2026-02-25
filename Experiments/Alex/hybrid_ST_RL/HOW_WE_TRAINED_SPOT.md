# How We Trained a Robot Dog to Walk on Anything

## The Story of Teaching Spot to Handle the Real World

---

## The Goal

We want a Boston Dynamics Spot robot to walk confidently across any terrain
it might encounter in the real world — ice, mud, stairs, gravel, tall grass,
rubble. Not just survive, but actually get where it's going.

The catch: we can't train on the real robot. Real robots break, real training
takes forever, and you can't run 16,000 real robots at the same time. So we
train entirely in simulation, then transfer the learned behavior to the
physical robot.

This document explains how we did it, why we made the choices we made, and
what went wrong along the way.

---

## Part 1: What the Robot Sees and Does

### The Robot's Senses (What Goes In)

Imagine you're blindfolded and told to walk across a room full of obstacles.
You'd rely on:

- **Your inner ear** — Am I tilted? Am I falling? Which way is down?
- **Your muscles and joints** — Where are my legs right now? How fast are
  they moving?
- **Your feet** — What does the ground feel like under me?
- **Memory** — What did I just do? (So I don't repeat mistakes)

The robot gets the same information, just as numbers:

| What it knows | How many numbers | Human equivalent |
|--------------|-----------------|-----------------|
| Body velocity (linear) | 3 | Inner ear: "I'm moving forward at 1 m/s" |
| Body velocity (angular) | 3 | Inner ear: "I'm rotating left" |
| Gravity direction | 3 | Inner ear: "The ground tilts 10° right" |
| Velocity commands | 3 | Brain: "I want to go forward at 2 m/s" |
| Joint positions | 12 | Proprioception: "My left front knee is bent 30°" |
| Joint velocities | 12 | Proprioception: "My right rear hip is swinging forward" |
| Last action | 12 | Memory: "Here's what I just told my legs to do" |
| Height scan | 187 | Feeling the ground with a cane in a 1.6m × 1.0m grid |

That last one — the height scan — is the robot's only way to "see" the
terrain ahead. It's like sweeping a cane across the ground in front of you:
187 measurements of "how far down is the ground here?" arranged in a grid.
The robot doesn't know if it's looking at stairs or a boulder — it just
knows the shape of the ground within arm's reach.

**Total: 235 numbers, updated 50 times per second.**

### The Robot's Actions (What Comes Out)

The robot outputs 12 numbers — one target angle for each joint (3 joints per
leg × 4 legs). These aren't raw motor commands; they're more like
suggestions. The robot's internal controllers (springs and dampers running at
500 Hz) smoothly move each joint toward its target. The policy only updates
these targets 50 times per second.

Think of it like steering a car: you turn the wheel (set the target), and
the power steering system (PD controller) does the actual work of turning
the tires. You don't control individual hydraulic valves.

### The Brain (Neural Network)

The policy is a simple feedforward neural network:

```
235 inputs → 512 neurons → 256 neurons → 128 neurons → 12 outputs
```

No memory, no attention, no convolutions. Just three layers of multiply-
and-add with a smooth activation function (ELU). The entire thing has about
270,000 parameters — tiny by modern AI standards. GPT-4 has a trillion.

Why so simple? Because it has to run at 50 Hz on the robot's onboard
computer. A transformer would be smarter but too slow. And it turns out
that 270K parameters is more than enough to encode "how to walk on stuff"
— the hard part is getting the right training signal, not the right
architecture.

---

## Part 2: How Reinforcement Learning Works (The Training Loop)

### The Basic Idea

Reinforcement learning is trial and error at industrial scale.

Imagine you're learning to ride a bicycle. You try something, you fall, you
adjust. After a few hours, you can ride. Now imagine you could clone yourself
16,384 times, have all your clones try different things simultaneously, and
merge their experiences into one brain every 0.5 seconds. That's what we're
doing.

Each "iteration" of training works like this:

1. **Collect experience:** All 16,384 robots take 24 steps (about half a
   second of walking). Some fall, some succeed, some discover new tricks.
2. **Evaluate:** How well did each robot do? (The reward function scores
   every timestep.)
3. **Learn:** The neural network adjusts its weights to do more of what
   worked and less of what didn't. (PPO algorithm — details below.)
4. **Repeat.** 25,000 times.

### PPO: The Learning Algorithm

We use **Proximal Policy Optimization** (Schulman et al., 2017), which is
the standard algorithm for training locomotion policies. Here's the
intuition:

Imagine you're coaching a basketball team. After each game, you give
feedback: "When you had the ball at the three-point line, shooting was a
good idea (you scored). When you tried to dribble through three defenders,
that was a bad idea (you lost the ball)."

PPO does this, but with a safety constraint: **don't change too much at
once.** If the team had a winning season, don't overhaul the entire playbook
based on one bad game. PPO clips the size of each update so the policy never
changes drastically in a single step. This is critical for stability — RL
without this constraint tends to oscillate wildly or collapse entirely.

The specific mechanism: PPO computes the probability ratio between the new
policy and the old policy for each action taken. If the ratio exceeds a
threshold (we use ±20%), the gradient is clipped. Additionally, we use
**adaptive KL divergence** — if the policy changes too much (KL > 0.008),
the learning rate is automatically reduced. This double safety net prevents
the kind of catastrophic forgetting that killed early RL approaches.

### The Reward Function: What "Good Walking" Means

This is the most important design decision in the entire system. The reward
function defines what the robot learns to optimize. Get it wrong, and the
robot finds creative ways to cheat.

We use 19 reward terms, and the design philosophy is: **reward the goal,
penalize the shortcuts.**

#### Positive Rewards (The Goals)

**"Go where I tell you" — Velocity Tracking (weight: +7.0)**

The most important reward. The robot gets points for matching its commanded
velocity — forward speed, lateral speed, and turning rate. Without this,
the robot has no reason to move at all.

But we don't just reward raw speed. The reward uses an exponential kernel:
perfect tracking gets full credit, being 0.5 m/s off gets about 60% credit,
and being 2 m/s off gets almost nothing. This encourages precision, not just
"go fast in roughly the right direction."

**"Walk properly" — Gait Reward (weight: +10.0)**

This is the heaviest single reward, and it enforces a **diagonal trot** —
the front-left and back-right legs move together, then the front-right and
back-left. This is how real quadrupeds walk at moderate speeds.

Without this reward, the robot discovers degenerate gaits: dragging its feet,
bunny-hopping, shuffling, or vibrating in place. These "work" in simulation
(they move the robot forward) but are terrible on real hardware — they cause
excessive wear, instability, and don't transfer to the real world.

The gait reward measures how synchronized diagonal leg pairs are. When FL
and HR are in swing phase at the same time, the reward is high. When the
timing is off, the reward drops. Weight 10.0 means this is essentially
non-negotiable — the robot WILL trot, and everything else must work within
that constraint.

**"Pick up your feet" — Foot Clearance (weight: +3.5)**

Rewards the robot for lifting its feet 10cm above the ground during the
swing phase. Without this, the policy learns to slide its feet along the
ground — efficient on flat surfaces but catastrophic on stairs and rough
terrain. This reward forces the robot to develop a high-stepping gait that
can clear obstacles.

**"Proper timing" — Air Time (weight: +3.0)**

Rewards each foot for spending the right amount of time in the air (0.3
seconds per swing). Too short = shuffling. Too long = hopping. Just right
= a natural trot cadence.

**"Adjust your speed" — Velocity Modulation (weight: +2.0)**

This is one of our custom additions. On easy terrain, the robot should go
full speed. On hard terrain (ice, boulders), it should slow down rather
than charging ahead and falling.

The reward estimates terrain difficulty from the variance of foot contact
forces — if all four feet are experiencing wildly different forces, the
terrain is rough. It then sets an adaptive speed target: full speed on easy
ground, 50% speed on very hard ground. The robot gets credit for matching
this adapted target.

This prevents two failure modes we saw in testing:
- Freezing on hard terrain (maximizes survival but zero velocity reward)
- Charging through everything (maximizes velocity reward but falls)

#### Negative Penalties (Preventing Cheats)

Every positive reward creates an incentive to game the system. The penalties
close the loopholes:

**"Don't shake" — Action Smoothness (weight: -2.0)**

Without this, the robot discovers that rapidly oscillating its joints makes
it vibrate forward. It technically moves, but it would destroy real motors.
This penalty charges the robot for changing its joint commands too quickly.

**"Stay upright" — Base Orientation (weight: -5.0)**

Penalizes tilting. The robot gets charged for rolling or pitching its body.
Heavy weight (-5.0) because falling sideways is one of the worst possible
outcomes and the policy needs strong pressure to avoid it.

**"Don't bounce" — Base Motion (weight: -4.0)**

Penalizes vertical bouncing and lateral swaying. Without this, some policies
develop a galloping motion that moves the entire body up and down — wasteful
and unstable.

**"Don't slip" — Foot Slip (weight: -3.0)**

Penalizes feet that slide along the ground while in contact. On low-friction
surfaces, the robot needs to learn to place its feet carefully rather than
relying on friction to stop them.

**"Don't trip" — Stumble Penalty (weight: -2.0)**

Our custom addition. Fires when a foot hits something at shin height (above
15cm) with significant force. This means the robot is kicking the side of
an obstacle rather than stepping over it. The penalty pushes the policy
toward higher foot clearance near obstacles.

**"Be gentle" — Contact Force Smoothness (weight: -0.5)**

Penalizes sudden spikes in ground reaction forces — slamming feet down
rather than placing them gently. Important for low-friction surfaces where
hard impacts cause slipping, and for real hardware where impacts cause wear.

**"Stay at the right height" — Body Height Tracking (weight: -2.0)**

Penalizes deviating from Spot's natural standing height (0.42m). Without
this, the policy sometimes discovers that crouching lowers the center of
gravity (more stable) or rising up gives better terrain visibility. Both
are technically useful but make the gait unnatural and reduce the terrain
the robot can handle.

**Energy efficiency penalties** — Joint acceleration (-5e-4), joint torques
(-2e-3), joint velocity (-5e-2), and joint position (-1.0) all discourage
the robot from using excessive force, moving joints unnecessarily fast, or
deviating from its natural standing pose. These are like telling a runner
"finish the race, but don't waste energy doing it."

**"Symmetric gait" — Air Time Variance (weight: -1.0)**

Penalizes asymmetric gaits where some legs swing faster than others.
Without this, the policy sometimes develops a limp — one leg does all the
work while the others barely move. A symmetric trot distributes effort
evenly and is more stable.

### Vegetation Drag: The Weird One

One reward term deserves special attention because it's doing double duty.

The **vegetation drag** system simulates walking through grass, mud, or
shallow water. It applies actual physical drag forces to the robot's feet
(F = -drag_coefficient × velocity) whenever they're in contact with the
ground. The drag coefficient varies:

- On "friction plane" terrain: zero drag (the challenge is pure low friction)
- On "vegetation plane" terrain: always dragging (0.5 to 20 N·s/m)
- On all other terrains: randomly assigned (25% no drag, 25% light, 25%
  medium, 25% heavy)

This isn't just a reward — it changes the physics. The robot actually feels
resistance pulling its feet backward. The small penalty (weight -0.001)
gives the policy a hint that drag is costly, but the real training signal
comes from the physics itself: robots in heavy drag that don't adapt their
gait will fall behind on velocity tracking.

No standard robotics simulator includes vegetation drag, so we built this
from scratch.

---

## Part 3: The Training Journey

### Attempt 1: The 48-Hour Baseline (Success)

Our first real training run. 8,192 simulated Spots walking on 6 terrain
types (stairs up, stairs down, boxes, rough ground, slopes up, slopes down)
for 48 hours on an NVIDIA H100.

**What it learned:** Walk, trot, follow velocity commands, climb moderate
stairs, handle rough ground. Terrain curriculum reached level 4-5.

**What it couldn't do:** Walk on ice (friction below 0.5), handle vegetation
drag (never trained on it), or navigate the hardest obstacle fields. These
weaknesses were expected — you can't do what you've never seen.

**Key numbers:** 27,500 iterations, 5.3 billion timesteps, 6 terrain types.

This checkpoint (`model_27500.pt`) became the foundation for everything
that followed.

### Attempt 2: The 100-Hour Failure (Disaster)

Ambitious plan: 12 terrain types, friction down to 0.05 (oil on steel),
bigger network, 65,536 parallel robots, 100+ hours. Train the ultimate
policy from scratch.

After 10,000 iterations and 17.6 billion timesteps:
- Terrain level: 0 (easiest, never promoted)
- Every single episode ended with the robot falling
- Average episode length: 7 seconds (just learned to stand)
- Reward: flat, no improvement trend

**The robot never learned to walk.** It spent 100 hours learning to stand
still.

#### Why It Failed: The Ice Cream Shop Analogy

Imagine you've never cooked before, and someone drops you in a kitchen and
says: "Make 12 different ice cream flavors, some of which require techniques
you've never heard of, and also the oven randomly changes temperature, the
floor is sometimes covered in oil, and someone will shove you every 10
seconds. Go."

You would stand very still and try not to fall.

That's what the robot did. Three specific problems:

**Problem 1: Starting from nothing.** We used a bigger network
([1024,512,256] instead of [512,256,128]) so we couldn't load the working
checkpoint. The robot started with completely random behavior. It had to
simultaneously learn to balance, walk, steer, AND handle extreme conditions.
That's like asking someone to learn to drive in a blizzard on a mountain
road with no guardrails — on their first day with a learner's permit.

**Problem 2: Maximum difficulty from day one.** Friction ranged from 0.05
(a wet ice rink) to 1.5 (sandpaper). In the same training batch, some
robots were on ice and some were on sandpaper. The ice robots needed slow,
careful movements. The sandpaper robots could charge aggressively. The
neural network tried to find one policy that works for both — and ended up
with a policy that worked for neither. It froze.

This is called **contradictory gradients** in the literature. Narvekar et
al. (2020) showed that presenting all tasks simultaneously leads to
catastrophic interference — the learning signals cancel each other out.

**Problem 3: The DR trap.** We had a terrain curriculum that was supposed
to promote robots to harder terrain as they improved. But the robots kept
failing not because of terrain difficulty, but because of the extreme
domain randomization (super low friction, hard pushes). The curriculum
couldn't advance because the policy kept falling due to DR, not terrain.
The terrain curriculum and the DR were working against each other.

### Attempt 3: Hybrid Student-Teacher RL (Current)

The fix was obvious in hindsight: **start from what works, and make things
harder gradually.**

#### The Driver's Ed Analogy

The 48-hour policy is like a teenager who passed their driver's test. They
can drive on normal roads in good weather. Now we want them to handle:
- Ice and snow (low friction)
- Off-road trails (rough terrain)
- Steep mountain roads (stairs/slopes)
- Dense fog (sensor noise)

The failed approach was: "Here's the keys, the road is covered in black ice,
there are boulders everywhere, a storm is blowing the car sideways, and we
changed the car to one you've never driven. Good luck."

The hybrid approach:
1. **Start with the car they know** — same [512,256,128] architecture, load
   the working checkpoint.
2. **Start on roads they can handle** — friction near what they trained on,
   gentle pushes, moderate terrain.
3. **Gradually increase difficulty** — over 15,000 iterations, slowly widen
   the friction range, increase push forces, add mass perturbations.
4. **12 terrain types from the start** — but the terrain curriculum puts
   them on the easy rows first, and the DR is gentle, so they can actually
   learn.

By the time the friction drops to 0.1 (about the hardness of walking on a
wet tile floor in socks), the robot has already been walking successfully
for thousands of iterations. It has momentum — a working gait that just
needs to adapt, not a blank slate that needs to learn everything at once.

#### Progressive Domain Randomization: The Core Innovation

Here's the schedule — each parameter starts easy and linearly ramps to its
final value over 15,000 iterations (about 60% of training):

| What changes | Start (easy) | End (hard) | Real-world equivalent |
|-------------|-------------|-----------|----------------------|
| Floor friction | 0.3 – 1.3 | 0.1 – 1.5 | Wet tile → oil on steel |
| Push force | ±0.5 m/s | ±1.0 m/s | Gentle nudge → hard shove |
| Wind force | ±3.0 N | ±6.0 N | Light breeze → strong gust |
| Wind torque | ±1.0 Nm | ±2.5 Nm | Slight twist → hard spin |
| Payload mass | ±5.0 kg | ±7.0 kg | Light backpack → heavy payload |
| Push frequency | Every 10-15s | Every 6-13s | Occasional → frequent |

**Why linear interpolation?** Simplicity. Fancier schedules (exponential,
cosine) exist but add complexity without proven benefit for this use case.
The key insight from curriculum learning (Bengio et al., 2009) is that
*any* progressive schedule works better than no schedule. The exact shape
matters less than the fact that difficulty increases gradually.

**Why 60% expansion, 40% at full difficulty?** The first 60% teaches the
policy to adapt to increasing challenge. The last 40% hardens it against
the full range. If we expanded too fast (say 5K iterations), the policy
wouldn't have time to stabilize at each difficulty level. If we expanded
too slow (say 24K), we'd waste compute on easy conditions the policy
already handles.

#### 12 Terrain Types: The Obstacle Course

The terrain is a 10×40 grid of 8m×8m patches — 400 patches total. Rows
represent difficulty (row 0 = easiest, row 9 = hardest). Columns represent
terrain type. The 12 types are:

**The Stair Family (20%):** Stairs going up (5-25cm steps), stairs going
down (same range, but harder because you must control momentum), and noisy
heightfield stairs (like real stairs with debris on them). Stairs test
precise foot placement and the ability to read terrain height changes from
the height scan.

**The Boulder Field (10%):** Random grid boxes (5-25cm height) arranged
unpredictably. This is the rubble/construction site proxy. Tests the
robot's ability to step over and around obstacles it can't see until
they're right in front of it.

**Precision Challenges (10%):** Stepping stones (hop between small
platforms) and gaps (step across voids). These push the policy toward
precise foot placement — you can't just tromp forward, you need to put
your feet exactly where the ground is.

**Natural Terrain (25%):** Random rough ground (2-15cm noise), hills going
up and down (up to 0.5 slope), and wave terrain (undulating ground like a
natural landscape). These test general-purpose locomotion over unstructured
surfaces.

**Isolated Challenges (10%):** Two flat planes that test specific physical
properties in isolation:
- **Friction plane:** Perfectly flat, but friction ranges from "normal road"
  to "wet ice rink." The ONLY challenge is low friction. This teaches the
  robot to walk carefully when the ground is slippery, without the
  confounding factor of also trying to navigate obstacles.
- **Vegetation plane:** Perfectly flat, but invisible drag forces pull the
  robot's feet backward (simulating grass, mud, or shallow water). The ONLY
  challenge is drag. This teaches the robot to push through resistance.

Separating these challenges onto dedicated terrain types is like how a
driving school has a skid pad (just practice sliding) separate from the
obstacle course (just practice steering). You learn each skill
independently before combining them.

**Compound Challenges (25%):** Discrete obstacles (scattered blocks),
repeated boxes (regular patterns), and more. These combine multiple
challenges — you might face rough ground with obstacles on a slope.

#### The Terrain Curriculum: Automatic Difficulty Matching

The system automatically adjusts which terrain row each robot trains on.
After each episode:
- If the robot traversed >50% of the expected distance → **promote** it to
  a harder row (row 3 → row 4)
- If the robot fell quickly → **demote** it to an easier row (row 4 → row 3)

This is self-balancing. If the policy gets better, more robots get promoted,
and the average difficulty increases automatically. If a policy update goes
wrong and performance drops, robots get demoted, and training focuses on
re-establishing basic competence before trying harder things again.

The beauty of this system (validated by Rudin et al., 2022) is that we
never need to manually set a training schedule. The curriculum finds the
**learning frontier** — the difficulty level where the policy is challenged
but not overwhelmed — and keeps training right at that edge.

#### When the Fine-Tuning Crashed (The Dashboard Problem)

We launched Stage 1 on the H100 on February 23, 2026. It failed within
50 minutes.

The robot started walking at terrain level 3.46 (decent difficulty).
Within 300 iterations, it collapsed to terrain level 0, falling in 100%
of episodes. The walking ability we carefully loaded from the 48hr
checkpoint was completely destroyed.

**The car dashboard analogy:**

Remember the driver's ed analogy? We gave our experienced driver their
familiar car (the [512,256,128] network) and loaded their driving skills
(the checkpoint). But we forgot one thing: we also loaded the car's
**dashboard gauges** — and the gauges were calibrated for the old car.

In RL terms, the "dashboard" is the **value function** (the critic
network). It tells the learning algorithm "how well am I doing right
now?" The old value function was calibrated for 14 reward terms. Our new
environment has 19 reward terms (5 new penalties). So the dashboard kept
saying "you should be earning 15 points per step!" when the robot was
actually earning 10 (because the 5 new penalties were dragging down its
score).

Here's the chain of events:

1. **Dashboard reads high.** The critic thinks the robot should be
   earning +15 reward per step (what it learned in the old environment).
2. **Actual reward is +10.** Because 5 new penalty terms are now active.
3. **The system concludes: "Walking is underperforming."** Expected +15,
   got +10. Walking must be a bad strategy.
4. **The learning algorithm shifts away from walking.** It starts trying
   to find something "better" — which means standing still (the only
   strategy that avoids the new penalties entirely).
5. **Exploration dies.** The action noise collapses from 0.65 to 0.15.
   The robot stops trying different things and commits to standing still.
6. **Game over.** With no exploration, the robot can never rediscover
   that walking actually works — it just needed to adapt slightly for
   the new penalties.

It's like a navigation system that thinks you should be going 80 mph
(because that's what the old highway allowed), but you're on a winding
mountain road doing 45. The system keeps yelling "YOU'RE TOO SLOW!" and
you keep overcorrecting until you drive off the road entirely.

The critical insight: **the robot's driving skills (actor) transferred
perfectly. Its internal GPS/speedometer (critic) was the problem.**
Loading both was worse than loading just the skills and letting the
robot recalibrate its gauges from scratch.

#### The Four Fixes (Attempt #2)

We implemented four fixes, all in the training script:

**Fix 1: Only load the driving skills, not the gauges.**

Instead of loading the entire checkpoint (actor + critic), we only load
the actor weights and the action noise parameter. The critic starts from
random — a blank dashboard that has to learn the new environment from
scratch. This is actually better than a miscalibrated dashboard, for the
same reason that a blank map is better than a wrong map.

**Fix 2: Let the gauges calibrate before driving.**

For the first 1,000 iterations, the actor is **frozen**. The robot walks
using its loaded skills, but those skills don't change. Meanwhile, the
fresh critic watches 393 million data points and learns "here's what the
reward actually looks like in this new environment."

It's like a new employee who shadows for a week before making any
decisions. "Just watch, learn the new system, and then we'll let you
start making changes."

After 1,000 iterations, the actor unfreezes and starts learning again —
but now the critic's guidance is calibrated. The first real update is
based on accurate "you're doing well" / "you're doing poorly" signals,
not miscalibrated ones.

**Fix 3: Never stop exploring.**

We added a **noise floor** — the action noise can never drop below 0.4.
In Attempt #1, the noise collapsed from 0.65 to 0.15, killing
exploration. Now, even if the value function has a bad day and pushes
noise down, it can't go below 0.4.

It's like telling a jazz musician: "You can settle into your style, but
you must always improvise at least a little. Never play the exact same
solo twice." This ensures the robot keeps trying variations of its
walking strategy, which is how it discovers adaptations for new terrains.

**Fix 4: Reset the speedometer when driving starts.**

This was a sneaky bug we found during smoke testing on the H100. During
the 1,000-iteration warmup (Fix 2), the actor is frozen, so every
iteration produces identical behavior. The adaptive learning rate sees
"the policy isn't changing at all" and keeps doubling the learning rate:
1e-4 → 2e-4 → 4e-4 → ... → 0.01. That's a 100× increase.

When the actor unfreezes, the first update hits with 100× the intended
force. It's like revving a car engine to 8,000 RPM in neutral, then
suddenly dropping it into gear — the wheels spin out and you lose
control.

The fix: at the exact moment the actor unfreezes, we reset the learning
rate back to its original value (1e-4). Clean start, no accumulated
momentum.

#### Attempt #2: The Noise Explosion

The fixes worked in our local 30-iteration smoke test. We launched the
production run on the H100. It failed within 45 minutes.

But this time the failure was the **exact opposite** of Attempt #1.

In Attempt #1, the action noise collapsed (0.65 → 0.15) — the robot
stopped exploring and froze. In Attempt #2, the action noise
**exploded** (0.65 → 5.75+) — the robot started producing completely
random movements and fell over instantly.

**The jazz musician who can't stop improvising:**

Remember Fix 3 — the noise floor that said "always improvise a little"?
It had a floor (minimum 0.4) but no ceiling. And during the 1,000-
iteration warmup (Fix 2), something went wrong with the ceiling.

Here's the key insight we missed: in RSL-RL's code, the action noise
(`std`) is stored as a **separate parameter**, not inside the actor
network. Our `freeze_actor()` function froze everything inside the
actor — all the weights that decide "what should the robot do?" — but
left the noise parameter free. It's like we told the jazz musician "play
the same notes every time" but forgot to add "and at the same volume."

During warmup, the actor is frozen. The PPO algorithm has a component
called the **entropy bonus** that encourages exploration. The actor
can't change its notes (frozen), so the only way the entropy bonus can
increase exploration is by turning up the volume — increasing the noise.
And it did. Relentlessly. For 247 iterations, the noise climbed from
0.65 to 5.75, at which point the "music" was pure static.

The noise floor (0.4) couldn't help — it was a floor, and the noise
was going up, not down.

**Why our smoke test didn't catch it:** The local test used
`actor_freeze_iters=3` — only 3 warmup iterations. In 3 iterations,
the noise barely budged (0.65 → 0.68). The production run used 1,000
warmup iterations, giving the entropy bonus 1,000 chances to crank the
volume. The bug was a time bomb — it only explodes if you wait long
enough.

#### The Two Additional Fixes (Attempt #3)

**Fix 5: Also freeze the volume knob.**

When freezing the actor for warmup, we now also freeze the noise
parameter. The robot walks using its loaded skills at its loaded noise
level (0.65), and nothing changes until the warmup ends. It's like
telling the musician "play these exact notes at this exact volume for
the first set. Then you can improvise."

**Fix 6: Put a ceiling on the noise.**

In addition to the floor (0.4 minimum), we added a ceiling (1.5
maximum). So the noise can never drop below 0.4 (preventing Attempt
#1's collapse) or rise above 1.5 (preventing Attempt #2's explosion).
The range [0.4, 1.5] brackets the checkpoint's converged noise (0.65)
with room to explore in both directions.

1.5 is about 2.3× the normal noise level — plenty of room for the
robot to experiment with bolder movements when it needs to, but not
enough to produce pure random noise.

#### Attempt #3: The Unfreeze Catastrophe

Attempt #3's warmup phase looked perfect. For 1,000 iterations, the
noise stayed locked at 0.65, terrain levels climbed to 3.47, and the
critic learned the new reward landscape. Everything was working.

Then the actor unfroze at iteration 1,000. And the policy collapsed
in 30 iterations.

```
Before unfreeze: episode length 206 steps, terrain level 3.47
5 iters later:   episode length 158
10 iters later:  episode length 80
15 iters later:  episode length 42
25 iters later:  episode length 16
50 iters later:  episode length 7
200 iters later: episode length 2.5 — robots fall immediately
```

**The aggressive driving instructor:**

Think of it this way. During warmup, the driving student (actor) was
sitting in the passenger seat while the instructor (critic) studied
the new roads for 3 hours. The instructor got a perfect feel for the
roads. Then the student takes the wheel — and the instructor starts
screaming corrections at maximum volume.

"TURN LEFT! NO, HARDER! BRAKE! ACCELERATE! TURN RIGHT!"

The corrections aren't wrong — but they're too loud and too frequent.
Each correction is 20% of the wheel. Five corrections per second. In
30 seconds, the student is so confused they drive off the road.

That's what happened. The PPO updates were too large:

- **Learning rate 1e-4:** Each update moved the weights significantly
- **Clip ratio 0.2:** Allowed 20% policy change per update
- **5 learning epochs × 8 mini-batches = 40 gradient steps per iteration**
- All combined: each iteration could change the policy dramatically

The first PPO update after unfreeze nudged the actor's behavior. The
critic's value predictions — calibrated for the frozen actor — became
slightly wrong. The next update, guided by slightly wrong predictions,
nudged the actor further. The critic fell further behind. Each update
made the next update worse. Within 30 iterations, it was a runaway
cascade.

The noise std also crept from 0.65 to 0.78 after unfreeze (entropy
bonus pushing it up), adding random jitter to an already-spiraling
situation.

**The key insight:** Fine-tuning a trained policy requires much gentler
updates than training from scratch. It's the difference between
teaching someone to drive from zero (big corrections are fine — they
don't know anything yet) and coaching an experienced driver on new
roads (gentle suggestions, not yanking the wheel).

#### The Three Additional Fixes (Attempt #4)

**Fix 7: Whisper, don't shout.**

We made the PPO updates dramatically smaller:

| Setting | Attempt #3 | Attempt #4 | Change |
|---------|-----------|-----------|--------|
| Learning rate | 1e-4 | 1e-5 | 10× lower |
| Clip ratio | 0.2 | 0.1 | Half the max change |
| Entropy bonus | 0.005 | 0.0 | Turned off completely |
| Learning epochs | 5 | 3 | Fewer passes over data |
| KL target | 0.008 | 0.005 | Tighter constraint |

Combined, each iteration now changes the policy about **30× less** than
Attempt #3. The driving instructor now whispers suggestions instead of
shouting commands.

**Fix 8: Freeze the noise permanently.**

In Attempt #3, we unfroze the noise std when the actor unfroze. It
crept from 0.65 to 0.78. In Attempt #4, the noise std stays frozen
at 0.65 forever — the robot explores at exactly the level the 48hr
policy converged to. Combined with entropy = 0.0, there's no force
pushing the noise in either direction.

It's like telling our experienced driver: "Drive with the same
confidence level you've always had. Don't suddenly get bolder or more
cautious. Just drive the way you know, and make small adjustments
for the new roads."

**Fix 9: Ease into it.**

Even with the lower learning rate (1e-5), the very first update after
1,000 iterations of freezing might be too abrupt. So we added a
gradual warmup:

- At unfreeze (iter 1000): LR starts at 2e-6 (5× below target)
- Over the next 1,000 iterations: LR linearly ramps to 1e-5
- After iter 2000: full learning rate, adaptive KL takes over

It's like slowly releasing the clutch after shifting gears. You don't
dump it — you ease into the new gear to avoid stalling.

#### Attempt #4: The Slow Fade

Attempt #4 made it past the unfreeze point — the first time any attempt
survived beyond iteration 1,030. The ultra-conservative settings worked
to prevent the instant 30-iteration crash of Attempt #3.

But the policy still died. Just slower.

```
Before unfreeze: episode length ~150 steps, terrain level ~3.0
50 iters later:  episode length ~120 (ok so far...)
100 iters later: episode length ~80 (hmm, declining...)
150 iters later: episode length ~20 (uh oh)
200 iters later: episode length ~4 — robots fall immediately
3000 iters later: episode length ~5 — never recovered
```

It's the same critic-staleness cascade as Attempt #3, just in slow
motion. Even at 1/30th the update strength, every tiny change to the
actor made the critic's predictions slightly wrong, and those slightly
wrong predictions made the next actor update slightly off, and so on.

**The fundamental realization:** The freeze-then-unfreeze approach
doesn't work. Period. It's not about the learning rate or the clip
ratio or the warmup schedule. The problem is training a critic on a
frozen actor and then expecting that critic to guide a moving actor.
It's like a GPS that was calibrated with you standing still — the
moment you start walking, it gives bad directions, and the worse
your path gets, the worse the directions get.

| What we measured | #1 (collapse) | #2 (explosion) | #3 (crash) | #4 (slow fade) |
|-----------------|--------------|----------------|-----------|---------------|
| Noise std | 0.15 (died) | 5.75 (died) | 0.78 (crept) | 0.65 (locked) |
| Failure mode | Noise collapse | Noise explosion | Instant crash | Gradual collapse |
| Failed at | iter ~300 | iter ~247 | iter ~1030 | iter ~1200 |
| Root cause | Wrong critic | Unfrozen std | PPO too aggressive | Critic-staleness fundamental |

### The Pivot: Attempt #5 — Start From Scratch

After four failed fine-tuning attempts, we asked a simple question:
**what if we just trained from scratch?**

The original 48-hour policy learned to walk from nothing. It worked.
Every failure since then was caused by the surgery we performed on
that policy — expanding dimensions, resetting critics, freezing
actors, warming up learning rates. The surgery itself was the problem.

So we threw out the fine-tuning approach entirely and started fresh:

- **No checkpoint** — train from random initialization
- **No freezing** — actor and critic learn together from iteration 0
- **No warmup** — standard adaptive KL handles the learning rate
- **No surgery** — the 235-dim observation space is there from day one

The height scan dimensions (187 values) are mostly zeros on flat
terrain. The network learns to walk using the 48 proprioceptive
inputs first, and as the terrain curriculum introduces harder ground,
the height scan values become non-zero and the network gradually
learns to use them.

**The terrain curriculum:**

All robots start on flat terrain (difficulty level 0). As they learn
to walk and track velocity commands, the curriculum automatically
promotes them to harder terrain:

| Terrain | What it is | Easy (level 0) | Hard (level 9) |
|---------|-----------|---------------|---------------|
| Flat | True flat ground | Flat | Flat |
| Uneven | Random rough ground | 2cm bumps | 15cm bumps |
| Boulders | Scattered boxes | 5cm high | 25cm high |
| Stairs up | Ascending stairs | 5cm steps | 25cm steps |
| Stairs down | Descending stairs | 5cm steps | 25cm steps |
| Friction | Slippery flat surface | Low grip | Low grip |
| Vegetation | Flat + drag resistance | Light drag | Heavy drag |

**The tradeoff:** We spend ~48 hours re-learning basic locomotion
that the 48hr policy already knew. But we completely avoid the fragile
fine-tuning surgery that failed four times. No freeze/unfreeze
boundary means no critic-staleness cascade.

#### What Happened: Attempt #5 Stalled

Training launched February 25, 2026. By iteration 2,000+, the reward
curve was flat at ~1.08, episode length stuck at ~7 seconds (robots
falling almost immediately), and body contact termination was at 100%.
The terrain curriculum never promoted a single robot — every one was
still on flat terrain at difficulty level 0.

The robot wasn't learning. It was flailing and dying, over and over.

### The Fix: Attempt #6 — Learning from Kumar et al. (2023)

We found our answer in a paper by Kumar et al. (2023), "Enhancing
Efficiency of Quadrupedal Locomotion over Challenging Terrains with
Extensible Feet." They trained the same Spot robot in Isaac Gym with
PPO, the same [512, 256, 128] architecture, and a similar terrain
curriculum. Their training showed clear reward improvement by
iteration 1,000. Ours was flat.

Comparing their setup to ours revealed five problems:

**1. Observation normalization was OFF (the critical fix)**

Kumar et al. used observation normalization — a running mean/std
normalizer that scales every input to zero mean, unit variance before
it enters the network. Our config had `actor_obs_normalization=False`
and `critic_obs_normalization=False`.

Why this matters: our 235-dim observation vector mixes values at
wildly different scales. Joint positions are ±0.5 radians, joint
velocities can hit ±30 rad/s, and height scan values range from
-1.0 to 1.0. Without normalization, the network's first layer is
dominated by whichever input has the largest magnitude. The height
scan (187 dimensions of small values) gets drowned out by a few
large velocity readings. The network can't learn from terrain
information if it can't even "see" it.

**Fix:** Enabled `actor_obs_normalization=True` and
`critic_obs_normalization=True`. This uses RSL-RL's built-in
`EmpiricalNormalization` module — a running mean/std tracker that
normalizes observations online during training.

**2. Initial noise was too high**

`init_noise_std=1.0` meant the policy's initial action distribution
was extremely wide — the robot was essentially flailing randomly with
maximum-amplitude joint commands. Combined with un-normalized
observations, the early gradient signal was pure noise.

**Fix:** Reduced to `init_noise_std=0.5`. Still enough exploration,
but actions start closer to the default pose.

**3. Termination was too aggressive**

We terminated episodes when the body OR any leg segment touched the
ground (`body_names=["body", ".*leg"]`). For a from-scratch policy,
shin contact during early stumbling is inevitable and informative.
Killing the episode on every shin scrape meant robots only survived
~7 seconds — not enough time to learn anything useful.

Kumar et al. only terminated on base (torso) contact.

**Fix:** Changed to body-only termination (`body_names=["body"]`).
Leg contact is still penalized via rewards but doesn't end the
episode. This gives the robot more timesteps to learn from its
mistakes instead of dying on every stumble.

**4. Spawn perturbations were too aggressive**

Robots spawned with ±1.5 m/s velocity and ±0.7 rad/s roll/pitch —
essentially launching mid-tumble. This is useful for robustness
testing of a trained policy, but terrible for a policy that can't
stand yet.

**Fix:** Reduced to ±0.5 m/s velocity and ±0.3 rad/s angular rates.
Robots now spawn in a survivable state.

**5. Too many reward terms**

19 reward terms created a noisy gradient signal. Five of them
(vegetation drag, velocity modulation, body height tracking, contact
force smoothness, stumble) are irrelevant when the robot can't stand.
They add noise to the critic's value estimates during the critical
first 1,000 iterations.

**Fix:** Zeroed out the 5 niche terms (weight=0.0), keeping 14 core
terms. The disabled terms can be re-enabled later once basic
locomotion is established.

#### Results: Immediate Improvement

After deploying Attempt #6 to the H100, the difference was dramatic.
At just iteration 14:

| Metric | Attempt #5 (iter 2000+) | Attempt #6 (iter 14) |
|--------|------------------------|---------------------|
| Episode length | 6.97s | 247s |
| Body contact rate | 100% | 42.7% |
| Terrain level | 0.0 | 0.48 |
| Throughput | 33.7K steps/s | 43.8K steps/s |

Robots were surviving 35× longer, more than half were avoiding body
contact entirely, and the terrain curriculum was already promoting
robots to harder terrain — all within the first 14 iterations.
Observation normalization was the single most impactful fix.

---

## Part 4: Scaling — Why 16,384 Robots?

### The Batch Size Argument

Each training iteration collects 24 steps from every robot. With 16,384
robots, that's 393,216 data points per update. The neural network sees
every terrain type, every friction value, every push direction in a single
batch.

With only 64 robots (what we run locally for testing), the network gets
1,536 data points — a noisy, unrepresentative sample. The gradient update
might overfit to whatever happened to those 64 robots. One bad batch could
undo several good ones.

**Analogy:** Imagine you're trying to figure out what Americans eat for
breakfast. If you ask 64 people, and by chance 40 of them are from the
South, you'll conclude that grits are the national breakfast food. If you
ask 16,384 people, you'll get a representative sample of every region.

The same applies to terrain. With 16,384 robots across 12 terrain types,
each terrain gets ~1,365 robots. That's a statistically robust sample of
"how do robots perform on ice?" and "how do robots perform on stairs?"
in every single batch.

### Why Not More?

Diminishing returns. Going from 8K to 16K doubles the per-terrain sample
and meaningfully improves gradient quality. Going from 16K to 32K (1,365
→ 2,730 per terrain) helps less but costs 80% more wall time per iteration.

The failed 100hr run used 65,536 robots. Each iteration took 43 seconds.
That's 5,461 robots per terrain — extreme overkill. The extra statistical
power didn't help because the fundamental training approach was wrong (no
warm start, no progressive DR). More data per batch doesn't fix bad
curriculum design.

### GPU Parallelism: Why It's Not Slower

NVIDIA's H100 GPU runs the physics simulation on the GPU itself. It
doesn't simulate robots one at a time — it processes all 16,384
simultaneously using its thousands of cores. Simulating 16K robots takes
about 10 seconds per iteration, while 8K takes about 6 seconds. That's
only 67% longer for 100% more data.

The reason: GPU cores work in parallel. 16,384 robots uses more cores but
doesn't require more sequential steps. The bottleneck is memory bandwidth
and synchronization, not compute. It's like a factory with 10,000 workers
— adding 8,000 more workers (from 8K to 16K robots) doesn't double the
time because they all work simultaneously. It just uses more workstations.

### The Budget Math

| Config | Steps/iter | Time/iter | Total iters | Total steps | Wall time |
|--------|-----------|-----------|-------------|-------------|-----------|
| 8K × 30K | 196K | ~6s | 30,000 | 5.9B | ~50h |
| **16K × 25K** | **393K** | **~10s** | **25,000** | **9.8B** | **~69h** |
| 32K × 15K | 786K | ~18s | 15,000 | 11.8B | ~75h |
| 64K × 10K | 1.57M | ~40s | 10,000 | 15.7B | ~111h |

We chose 16K × 25K because:
- 9.8 billion total steps (1.9× the successful 48hr run, accounting for
  2× more terrain types)
- ~69 hours fits our 72-hour compute budget
- 1,365 robots per terrain type (good statistical coverage)
- Fewer iterations than 48hr run (25K vs 27.5K) because each iteration
  is 2× more informative

---

## Part 5: The Teacher-Student Trick (If Needed)

Stage 1 might be enough. But if the robot still struggles on specific
terrains, we have a backup plan inspired by the CMU Extreme Parkour paper
(Cheng et al., 2024).

### The Idea: Learning with Cheat Codes

Imagine two students taking the same exam. One gets the exam with all the
answers in the margins (the teacher). The other gets the normal exam (the
student). The teacher will obviously do better.

Now here's the trick: we have the student watch the teacher take the exam.
The student can't see the answers in the margins, but they can see what the
teacher *does* — which answers they pick, how confident they are. Over time,
the student develops an intuition for the right answers even without seeing
the cheat sheet.

In our case:
- **Teacher** gets 254 numbers of input (the standard 235 plus 19
  "privileged" numbers: exact friction coefficient, terrain type,
  clean contact forces, slope direction)
- **Student** gets the standard 235 numbers (what the real robot would have)

The teacher learns faster because it knows exactly what kind of ground it's
standing on. "The friction is 0.15 — I need to be very careful." The student
has to figure this out indirectly from how the ground *feels* (through joint
forces and foot contacts).

### How Distillation Works

The student trains with a combined objective:

```
loss = (1 - β) × "Get high reward" + β × "Do what the teacher does"
```

Early in training, β = 0.8 — the student mostly copies the teacher (80%
imitation, 20% independent learning). As training progresses, β drops to
0.2 — the student mostly relies on its own experience but still gets a hint
from the teacher.

The annealing is important. Pure imitation (β = 1.0) would cap the student
at the teacher's performance — it could never be better. Pure RL (β = 0.0)
would ignore the teacher entirely. The gradual transition gives the student
a running start from the teacher's knowledge, then lets RL refine and
potentially exceed the teacher's performance.

### Weight Surgery: Upgrading the Brain

The teacher uses the same [512, 256, 128] network as the student, but its
input layer is wider: 254 columns instead of 235. We can't just load the
Stage 1 checkpoint because the matrix sizes don't match.

The solution: **weight surgery.** We take the first layer's weight matrix
(shape [512, 235]) and pad it with 19 new columns of zeros to make it
[512, 254]. All other layers are copied directly.

Why zeros? Because it means the teacher's initial behavior is identical to
the Stage 1 policy — the 19 new inputs are multiplied by zero, so they have
no effect. As training progresses, the teacher gradually learns to pay
attention to the new privileged information. This is much safer than random
initialization, which would immediately disrupt the policy's existing
walking ability.

---

## Part 6: What Could Go Wrong (And How We Prevent It)

### Catastrophic Forgetting

**The risk:** The robot already knows how to walk. If we update the network
too aggressively, it could forget how to walk before learning anything new.
Like a tennis player who takes golf lessons and comes back unable to serve.

**The prevention:** We use a learning rate 10× lower than the original
training (1e-5 vs 3e-4 in the original, down from 1e-4 in Attempt #3
which was still too aggressive — see "The Unfreeze Catastrophe" above).
We also set a tight KL divergence target (0.005), halve the clip ratio
(0.1 instead of the standard 0.2), and use a 1,000-iteration LR warmup
that starts at 2e-6 and gradually ramps to 1e-5. This multi-layered
approach was developed after Attempt #3 proved that even "conservative"
PPO settings (LR=1e-4, clip=0.2) are too aggressive for fine-tuning,
or eliminated entirely by training from scratch (Attempt #5).

We also don't load the old optimizer's momentum. The 48hr training used
gradient momentum tuned for LR=3e-4. Applying that momentum at LR=1e-5
would cause the first few updates to overshoot — the accumulated momentum
would push the weights too far. Fresh optimizer, fresh start.

### Reward Hacking

**The risk:** The robot finds a way to maximize reward without actually
walking. Classic examples: vibrating in place (action smoothness penalty
catches this), standing still on easy terrain (velocity tracking catches
this), or crouching to minimize orientation penalty (height tracking
catches this).

**The prevention:** The 19 reward terms are designed to form a closed
system where every shortcut is penalized. Each positive reward creates an
incentive, and each penalty closes a loophole. This adversarial design
process — imagining how the robot could cheat, then adding a penalty for
it — is the most important and most underappreciated part of reward
engineering.

### Sim-to-Real Gap

**The risk:** The policy works perfectly in simulation but fails on the real
robot because simulation doesn't perfectly match reality.

**The prevention:** Domain randomization. By training across a wide range of
physics parameters (friction 0.1-1.5, mass ±7 kg, push ±1.0 m/s), the
policy can't rely on any specific physics parameter being a certain value.
It has to develop a strategy that works across the entire range. The real
world is just one point in that range.

We also add noise to every observation (±0.15 m/s on velocity, ±0.05 rad
on joint positions, ±0.15m on height scan). This simulates sensor noise
on the real robot. A policy trained on clean data would break when it
encounters noisy real sensors. A policy trained on noisy data learns to be
robust to uncertainty — it doesn't flinch at small sensor glitches.

The height scan noise (±0.15m) is particularly important. Real depth
sensors are noisy, especially on reflective or transparent surfaces. By
training with heavy noise, the policy learns to "read" terrain from
rough height patterns rather than relying on precise measurements.

---

## Part 7: The Numbers That Matter

### Training Configuration Summary

| Parameter | Value | Why |
|-----------|-------|-----|
| Parallel robots | 16,384 | 1,365 per terrain type — good coverage |
| Iterations | 25,000 | 9.8B total steps — 1.9× the 48hr budget |
| Terrain types | 12 | Stairs, boulders, slopes, ice, grass, gaps, waves, obstacles |
| Terrain grid | 10×40 = 400 patches | 10 difficulty rows × 40 type columns |
| Network | [512, 256, 128] | Same as working checkpoint — must match |
| Learning rate | 1e-4 | 3× lower for fine-tuning stability |
| Episode length | 30 seconds | Long enough to evaluate sustained walking |
| DR expansion | 15,000 iterations | 60% of training progressively harder |
| Wall time | ~69 hours | Fits 72-hour compute budget |
| GPU | NVIDIA H100 NVL 96GB | ~34K steps/second throughput |

### What Success Looks Like

| Environment | 48hr Baseline | Target | What It Tests |
|-------------|--------------|--------|--------------|
| Friction | 69% fall rate | < 40% falls | Walking on ice |
| Grass | 19% fall rate | < 15% falls | Pushing through vegetation |
| Boulder | 70% fall rate | < 50% falls | Navigating rubble |
| Stairs | 0.22 m/s | > 0.2 m/s | Climbing real stairs |

---

## References

1. **Kumar, A., Li, Z., Zeng, J., Pathak, D., Sreenath, K., & Malik, J.**
   (2023). "Enhancing Efficiency of Quadrupedal Locomotion over Challenging
   Terrains with Extensible Feet." arXiv: 2305.01998. — Spot robot trained
   from scratch in Isaac Gym with PPO, [512, 256, 128] architecture, terrain
   curriculum, and observation normalization. Direct comparison of their
   setup to ours revealed the root causes of Attempt #5's stall and
   informed all five fixes in Attempt #6.

2. Schulman et al. (2017). "Proximal Policy Optimization Algorithms." — The
   PPO algorithm we use for all training.

3. Cheng, Shi, Agarwal & Pathak (2024). "Extreme Parkour with Legged
   Robots." ICRA 2024. — Teacher-student distillation and terrain curriculum
   for quadruped locomotion. Primary inspiration for Stage 2.

4. Rudin, Hoeller, Reist & Hutter (2022). "Learning to Walk in Minutes
   Using Massively Parallel Deep Reinforcement Learning." CoRL. — RSL-RL
   framework, terrain curriculum, and proof that massive parallelism works.

5. Hoeller et al. (2024). "ANYmal Parkour." Science Robotics. — Privileged
   observation teacher-student paradigm for terrain navigation.

6. Peng et al. (2018). "Sim-to-Real Transfer with Dynamics Randomization."
   ICRA. — Domain randomization as the primary sim-to-real technique.

7. Narvekar et al. (2020). "Curriculum Learning for RL: A Framework and
   Survey." JMLR. — Why progressive difficulty works better than all-at-once.

8. Bengio et al. (2009). "Curriculum Learning." ICML. — Original curriculum
   learning paper showing that starting easy improves convergence.

9. Kirkpatrick et al. (2017). "Overcoming Catastrophic Forgetting." PNAS.
   — Why fine-tuning with a low learning rate prevents losing existing skills.

10. Hwangbo et al. (2019). "Learning Agile and Dynamic Motor Skills for
    Legged Robots." Science Robotics. — Contact force penalties and
    proprioceptive locomotion foundations.

11. Ross, Gordon & Bagnell (2011). "DAgger: A Reduction of Imitation
    Learning." AISTATS. — The annealing behavior cloning approach used in
    student distillation.

---

*Written February 23, 2026. Updated February 25, 2026. Four fine-tuning
attempts, nine fixes, one fundamental realization (you can't train a
critic on a frozen actor), and one critical lesson from Kumar et al.
(2023): observation normalization is not optional. Attempt #5 started
fresh but stalled because the network couldn't see its own inputs at
the right scale. Attempt #6 fixed that — and the robot started learning
within 14 iterations. 16,384 simulated Spots on the H100, with terrain
curriculum that grows as they improve.*
