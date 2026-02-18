"""Main entry point for headless and rendered capstone evaluation.

Runs N episodes with M parallel environments, collecting per-episode metrics
and optionally capturing video.

Usage (headless):
    ./isaaclab.sh -p src/run_capstone_eval.py --headless \
        --num_envs 512 --num_episodes 1000 \
        --policy flat --env friction \
        --output_dir results/

Usage (rendered with video):
    ./isaaclab.sh -p src/run_capstone_eval.py \
        --num_envs 5 --num_episodes 10 \
        --policy rough --env stairs \
        --rendered --capture_video \
        --output_dir results/rendered/

Reuses patterns from:
- ARL_DELIVERY/05_Training_Package/spot_rough_48h_cfg.py (AppLauncher, headless setup)
- ARL_DELIVERY/03_Rough_Terrain_Policy/spot_rough_terrain_policy.py (policy loading)
"""

# TODO: Implementation
# 1. Parse CLI args (--headless, --num_envs, --num_episodes, --policy, --env, --rendered, --capture_video, --output_dir)
# 2. Initialize Isaac Lab AppLauncher (headless or rendered)
# 3. Build environment from zone_params via envs.build_environment()
# 4. Load policy checkpoint (flat or rough)
# 5. Create WaypointFollower
# 6. Create MetricsCollector
# 7. Main loop:
#    while episodes_completed < num_episodes:
#        env.reset()
#        waypoint_follower.reset()
#        for step in range(MAX_STEPS):
#            commands = waypoint_follower.compute_commands(root_pos, root_yaw)
#            obs = env.get_observations()
#            actions = policy(obs)
#            env.step(actions)
#            metrics_collector.step(...)
#            if metrics_collector.all_done(): break
#        metrics_collector.save_batch()
# 8. Generate summary


def main():
    raise NotImplementedError("TODO: Implement main evaluation loop")


if __name__ == "__main__":
    main()
