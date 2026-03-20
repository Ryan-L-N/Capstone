# For the Dogs: How Our Robot Learns to See and Navigate

## What Does the Robot Actually Do?

Picture a four-legged robot about the size of a large dog — think Boston Dynamics Spot, because that is exactly what we are working with. Right now, this robot can walk over rough terrain: rocks, stairs, slopes, uneven ground. It keeps its balance, coordinates all four legs, and does not fall over. That alone took months of training.

But walking is only half the problem. A robot that can walk but cannot decide *where* to walk is like a car with a great engine and no steering wheel. Phase C — the piece we are building now — gives the robot eyes and a sense of direction. We are teaching it to look at the world through a camera, understand what is in front of it, and choose a path through obstacles on its own.

---

## How Does It See?

The robot does not see the world the way you do. It does not use a color camera. Instead, it has a **depth camera** — the same kind of sensor inside an Xbox Kinect or the Face ID module on an iPhone. Instead of capturing colors, it measures the *distance* to every point in front of it.

Imagine a 64-by-64 grid of tiny distance measurements. Each square in that grid says "the nearest object in this direction is X meters away." If you turned that into a picture, nearby obstacles would look dark and far-away open space would look bright — like a grayscale photo where brightness means distance.

The robot can see about 30 meters ahead. That is roughly the length of a basketball court. Plenty of room to spot a boulder, a staircase, or a gap in the terrain and plan around it.

On top of the depth camera, the robot also sweeps 180 invisible laser beams across the ground around its feet (called a LiDAR scan). Think of it like tapping the ground with a very long cane in every direction at once — it tells the robot exactly how bumpy or tilted the ground is right where it is standing.

---

## Two Brains, One Robot

Here is the key idea that makes the whole system work: the robot has **two separate decision-makers**, and they do completely different jobs.

**The Walking Brain (Phase B — already trained).** This brain knows *how* to walk. It coordinates all four legs, keeps the robot balanced, adjusts foot placement on uneven ground, and recovers from stumbles. It makes 50 decisions per second — one every 20 milliseconds. It is fast and reactive, like your reflexes.

**The Navigation Brain (Phase C — what we are training now).** This brain knows *where* to walk. It looks at the depth camera, figures out which direction is safe, and sends high-level commands: go forward, sidestep left, turn right, slow down. It makes 10 decisions per second — one every 100 milliseconds. It is slower and more deliberate, like conscious thought.

The best analogy is **a rider on a horse**. The rider (navigation brain) decides the direction and speed. The horse (walking brain) handles the footwork — where to place each hoof, how to balance on a slope, when to shorten its stride. The rider does not micromanage the horse's legs, and the horse does not decide the destination. They work together.

In practice, the walking brain is "frozen" — we do not change it anymore. It already graduated from training. The navigation brain sits on top and learns to send it good commands.

---

## How It Learns

We use a technique called **reinforcement learning**. The basic idea is simple: the robot tries things, and we tell it which things were good and which were bad. Over time, it figures out what works.

Think of it like training a dog with treats.

Every fraction of a second, the robot earns or loses points based on what it is doing:

- **Moving toward the goal: +10 points.** The single biggest reward. Go forward, get treats.
- **Staying alive (not falling over): +1 point.** Just surviving is worth something.
- **Traveling a long distance: +2 points.** Encourages covering ground, not just inching along.
- **Crouching or belly-sliding: -1.5 points.** We do not want the robot gaming the system by crawling. Walk upright.
- **Jerky, erratic movements: -1 point.** Smooth and steady, not twitchy. Jerky commands would wear out real motors fast.
- **Spinning in circles: -0.5 points.** Actually go somewhere. Do not just rotate in place.

The robot runs through millions of attempts — literally billions of individual decisions — across thousands of simulated copies of itself running in parallel on a powerful GPU. Each copy explores different strategies. The ones that earn the most points get reinforced; the ones that fail get discarded. Over hours and days of training, the robot converges on a strategy that reliably earns high scores: look ahead, avoid obstacles, walk forward smoothly.

---

## The Terrain Obstacle Course

We do not throw the robot into the hardest environment on day one. Instead, we built a **curriculum** — a series of difficulty levels, like grade school through college.

There are 6 difficulty levels and 8 different terrain types mixed together:

- **Level 1:** Smooth, flat floor. Basically a living room.
- **Level 2:** Gentle slopes and small bumps. A well-maintained hiking trail.
- **Level 3:** Uneven rocks and moderate slopes. A scramble in the woods.
- **Level 4:** Stairs, gaps, and larger obstacles. An obstacle course.
- **Level 5:** Steep inclines, tall stairs, rough rubble. Serious mountaineering.
- **Level 6:** Boulders, waves, and combinations of everything. The final exam.

The robot starts at Level 1. When it proves it can handle the current level — measured by how far it walks without falling — it gets **promoted** to the next level. If it struggles, it stays where it is or gets sent back down. This way, it builds skills gradually instead of being overwhelmed.

---

## The AI Coach

Here is something a little unusual about our setup: we have a **second AI watching the first AI train**.

That second AI is Claude (made by Anthropic — the same company behind this document's writing assistant). It acts as a coach. Every hundred training rounds, it looks at the robot's performance stats and asks: Is the robot improving? Is it developing bad habits? Is it stuck?

Based on what it sees, the coach can adjust the "reward recipe" — the point values described above. For example:

- If the robot is crawling on its belly instead of walking upright, the coach increases the height penalty to discourage that.
- If the robot is stuck at a difficulty level and not making progress, the coach might boost the forward-movement reward to encourage bolder exploration.
- If the robot is moving but wobbling dangerously, the coach might increase the smoothness penalty.

Think of it like a personal trainer adjusting your workout plan week by week based on your progress, instead of giving you a fixed program and hoping for the best.

---

## Safety Rails

Letting one AI adjust another AI's training sounds risky, so we built strict guardrails:

- **Small changes only.** The coach can adjust any single reward by at most 20% at a time. No dramatic overhauls.
- **Three changes max.** Per consultation, the coach can tweak at most 3 reward weights. We learned the hard way (Trial 11k) that changing too many things at once can collapse the whole policy.
- **No sign flips.** The coach cannot turn a reward into a penalty or vice versa. The fundamental structure of "good behavior = positive, bad behavior = negative" stays locked.
- **Emergency stop.** If the math breaks down (a condition called NaN — "Not a Number," which is essentially a computer saying "I have no idea what happened"), training halts automatically. No corrupt data gets saved, no wasted GPU time.

These rails exist because we burned real compute time learning these lessons. Every rule here corresponds to a specific failure we experienced and documented.

---

## Why This Matters

A robot that can walk is impressive. A robot that can *see, plan, and navigate* on its own is useful.

**Search and rescue:** After an earthquake or building collapse, a robot that can pick its way through rubble to find survivors — without a human steering it with a joystick — could save lives.

**Military reconnaissance:** Scouting dangerous terrain without putting soldiers at risk. The robot goes first, maps the area, and reports back.

**Industrial inspection:** Power plants, mines, construction sites — environments that are dangerous or difficult for people. A robot that can walk to the right spot, look around, and come back is a practical tool.

**Space exploration:** Uneven, rocky terrain with no GPS and significant communication delays. A robot on Mars cannot wait 20 minutes for a human to tell it where to step next.

All of these applications need the same core capability: a legged robot that can look at unfamiliar terrain and figure out, on its own, where to go. That is exactly what Phase C is building. The walking brain gives it legs. The navigation brain gives it purpose.

---

*Written for the Capstone team and stakeholders — March 2026.*
