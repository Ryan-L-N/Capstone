"""Per-episode metrics collector — accumulates data during simulation.

Collected per episode:
  - completion (bool): reached x >= 49m within 120s
  - progress (float): max x-position (0-50m)
  - zone_reached (int): highest zone fully traversed (1-5)
  - time_to_complete (float): seconds to x=50m or timeout
  - stability_score (float): mean(|roll|) + mean(|pitch|) + 10*var(height) + 0.5*mean(||ang_vel||)
  - mean_roll, mean_pitch, height_variance, mean_ang_vel
  - fall_detected (bool): base_height < 0.15m
  - fall_location (float): x-position of fall
  - fall_zone (int): zone where fall occurred
  - mean_velocity (float): average forward velocity
  - total_energy (float): sum of |joint_torques * joint_vel|
  - episode_length (int): number of control steps

Output: JSONL files (one JSON object per line, one line per episode).
Supports append mode for resumability across batches.

Reuses patterns from:
- ARL_DELIVERY/06_Core_Library/data_collector.py (JSON logging)
- ARL_DELIVERY/05_Training_Package/isaac_lab_spot_configs/mdp/rewards.py (stability terms)
"""

# TODO: Implementation
# - MetricsCollector class
#   - __init__(self, num_envs, output_dir, env_name, policy_name)
#   - step(self, root_pos, root_quat, root_vel, joint_torques, joint_vel)
#   - check_falls(self) -> detects height < 0.15m
#   - check_zone_transitions(self)
#   - compute_episode_metrics(self, env_ids) -> dict per env
#   - save_batch(self, batch_idx)
#   - all_done(self) -> bool


class MetricsCollector:
    """Accumulates per-step data and exports per-episode metrics."""

    def __init__(self, num_envs, output_dir, env_name, policy_name):
        raise NotImplementedError("TODO: Implement MetricsCollector")
