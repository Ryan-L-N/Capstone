# Immersive Modeling and Simulation for Autonomy

## Problem

Quadrupedal robots face significant challenges in cluttered terrain, frequently falling, getting stuck, or failing to navigate the space entirely. While this might appear straightforward to solve, it remains a genuinely difficult problem even with advanced sensing and computation.

## Motivation

Our specific goal of this repository is to use one of Isaac Sim's built-in quadruped Spot models, develop a custom reinforcement learning policy for locomotion that outperforms Isaac Sim's default Flat Terrain and Rough Terrain policies in cluttered environments.

Our motivation for this is to enhance quadruped locomotion in unstructured environments (like those typically found on a battlefield).

## Related Work

1. NVIDIA, "Isaac Sim Documentation," https://developer.nvidia.com/isaac-sim
2. Boston Dynamics, "Spot Robot," https://www.bostondynamics.com/spot
3. Ghost Robotics, "Vision 60," https://www.ghostrobotics.io/vision-60
4. Robotics and AI Institute, "EVORA: Deep Evidential Traversability Learning for Risk-Aware Off-Road Autonomy," https://rai-inst.com/resources/papers/evora-deep-evidential-traversability-learning-for-risk-aware-off-road-autonomy/
5. Motion Planning for Quadrupedal Locomotion: Coupled Planning, Terrain Mapping and Whole-Body Control, "https://arxiv.org/pdf/2003.05481"

## Anticipated Tasks

- [x] Set up Isaac Sim Environment
- [ ] Create testing sim environment
- [ ] Create training sim environment
- [ ] Train initial RL policy
- [ ] Evaluate Baseline models
- [ ] Evaluate RL model
- [ ] Refine training

## Capacity Gaps

- Knowledge Gaps
- Compute and Graphics Infrastructure

## Capability Gaps

- Unknown Isaac Sim limitation or reinforcement learning nuance prevents us from developing a competitive RL policy **(HIGH)**

## AI2C Fit

This project emphasizes robotic autonomy that could be translated into an Army priority.

## RFI's for Customer

None at this time.

## Mentor Info

**Dr. Sean Gart** — ARL Autonomy

## Customer Info

Capstone Students
