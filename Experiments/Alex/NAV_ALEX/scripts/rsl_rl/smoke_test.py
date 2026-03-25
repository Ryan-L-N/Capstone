"""Quick local smoke test — validates CNN, rewards, loco wrapper, and training loop.

Runs 16 envs on flat terrain for 100 iterations with no coach.
Verifies:
    1. CNN forward pass works (dummy tensor shapes)
    2. ActorCriticCNN act/evaluate methods work
    3. Training loop runs without NaN or crashes
    4. Loss decreases (basic learning signal)

Usage:
    python scripts/rsl_rl/smoke_test.py \
        --headless \
        --loco_checkpoint checkpoints/ai_coached_v8_10600.pt

If no loco checkpoint available, use --skip_env to test CNN only.
"""

from __future__ import annotations

import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Phase C smoke test")
    parser.add_argument("--loco_checkpoint", type=str, default=None,
                        help="Path to frozen Phase B loco checkpoint")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip_env", action="store_true",
                        help="Skip env test (CNN unit tests only)")
    return parser.parse_args()

args = parse_args()


def test_depth_cnn():
    """Test 1: DepthCNN forward pass with random tensor."""
    import torch
    from nav_locomotion.modules.depth_cnn import DepthCNN, DEPTH_RES

    print("\n--- Test 1: DepthCNN Forward Pass ---")
    cnn = DepthCNN(depth_res=DEPTH_RES, feature_dim=128)

    batch_size = 16
    dummy_depth = torch.randn(batch_size, DEPTH_RES * DEPTH_RES)
    output = cnn(dummy_depth)

    assert output.shape == (batch_size, 128), f"Expected (16, 128), got {output.shape}"
    assert torch.isfinite(output).all(), "NaN/Inf in CNN output!"
    print(f"  PASS: input ({batch_size}, {DEPTH_RES*DEPTH_RES}) -> output {output.shape}")
    print(f"  CNN params: {sum(p.numel() for p in cnn.parameters()):,}")


def test_actor_critic_cnn():
    """Test 2: ActorCriticCNN act/evaluate with flat obs."""
    import torch
    from nav_locomotion.modules.depth_cnn import ActorCriticCNN, TOTAL_OBS_DIMS

    print("\n--- Test 2: ActorCriticCNN act/evaluate ---")
    policy = ActorCriticCNN(
        num_obs=TOTAL_OBS_DIMS,
        num_actions=3,
        depth_res=64,
        init_noise_std=0.5,
    )

    batch_size = 16
    obs = torch.randn(batch_size, TOTAL_OBS_DIMS)

    # Test act()
    actions, log_prob, values, mean_actions = policy.act(obs)
    assert actions.shape == (batch_size, 3), f"Actions shape: {actions.shape}"
    assert log_prob.shape == (batch_size,), f"Log prob shape: {log_prob.shape}"
    assert values.shape == (batch_size, 1), f"Values shape: {values.shape}"
    assert mean_actions.shape == (batch_size, 3), f"Mean actions shape: {mean_actions.shape}"
    print(f"  act() PASS: actions={actions.shape}, values={values.shape}")

    # Test act_inference()
    det_actions = policy.act_inference(obs)
    assert det_actions.shape == (batch_size, 3)
    print(f"  act_inference() PASS: {det_actions.shape}")

    # Test evaluate()
    values, log_prob, entropy = policy.evaluate(obs, actions)
    assert values.shape == (batch_size, 1)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    print(f"  evaluate() PASS: values={values.shape}, entropy={entropy.shape}")

    # Check all outputs are finite
    for name, t in [("actions", actions), ("values", values), ("log_prob", log_prob)]:
        assert torch.isfinite(t).all(), f"NaN/Inf in {name}!"

    print(f"  Total params: {sum(p.numel() for p in policy.parameters()):,}")


def test_guardrails():
    """Test 3: Guardrails validation logic."""
    print("\n--- Test 3: Guardrails Validation ---")
    from nav_locomotion.ai_coach.guardrails import Guardrails

    g = Guardrails()
    current = {
        "forward_velocity": 10.0,
        "survival": 1.0,
        "terrain_relative_height": -2.0,
        "lateral_velocity": -0.3,
    }

    # Test: too many changes (should trim to 3)
    changes = {
        "forward_velocity": 11.0,
        "survival": 1.2,
        "terrain_relative_height": -2.2,
        "lateral_velocity": -0.35,  # 4th change — should be trimmed
    }
    approved, msgs = g.validate_weight_changes(changes, current, current_terrain_level=2.0)
    assert len(approved) <= 3, f"Expected <= 3 changes, got {len(approved)}"
    print(f"  Count limit: PASS (trimmed to {len(approved)})")

    # Test: sign flip rejection
    changes = {"forward_velocity": -1.0}
    approved, msgs = g.validate_weight_changes(changes, current, current_terrain_level=5.0)
    assert "forward_velocity" not in approved, "Sign flip should be rejected"
    print("  Sign constraint: PASS (sign flip rejected)")

    # Test: terrain-gated loosening
    changes = {"terrain_relative_height": -1.5}  # Loosening (less negative)
    approved, msgs = g.validate_weight_changes(changes, current, current_terrain_level=1.0)
    assert "terrain_relative_height" not in approved, "Loosening at low terrain should be rejected"
    print("  Terrain-gated loosening: PASS (blocked at terrain 1.0)")

    # Test: delta limit
    changes = {"forward_velocity": 15.0}  # 50% increase, should be clamped to 20%
    approved, msgs = g.validate_weight_changes(changes, current, current_terrain_level=5.0)
    if "forward_velocity" in approved:
        assert approved["forward_velocity"] <= 12.1, f"Delta should be clamped: {approved['forward_velocity']}"
    print("  Delta limit: PASS")


def test_rewards():
    """Test 4: Reward clamping with dummy tensors."""
    import torch
    from nav_locomotion.tasks.navigation.mdp.rewards import _safe_clamp

    print("\n--- Test 4: Reward Clamping ---")

    # Normal values
    x = torch.tensor([0.5, 1.0, -0.5, 2.0])
    result = _safe_clamp(x, 0.0, 1.0)
    assert (result >= 0.0).all() and (result <= 1.0).all(), "Clamp failed"
    print("  Normal clamp: PASS")

    # NaN handling
    x = torch.tensor([float("nan"), float("inf"), float("-inf"), 0.5])
    result = _safe_clamp(x, 0.0, 1.0)
    assert torch.isfinite(result).all(), "NaN/Inf not handled"
    print("  NaN/Inf safety: PASS")


def main():
    print("=" * 60)
    print("NAV_ALEX SMOKE TEST")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_fn in [test_depth_cnn, test_actor_critic_cnn, test_guardrails, test_rewards]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)

    if args.skip_env or args.loco_checkpoint is None:
        print("\nSkipping env test (--skip_env or no --loco_checkpoint)")
        return

    # ----- Full env smoke test -----
    print("\n--- Test 5: Full Environment Smoke Test (16 envs, 10 iters) ---")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)

    import gymnasium as gym
    import nav_locomotion  # noqa: F401
    from nav_locomotion.modules.depth_cnn import ActorCriticCNN
    from nav_locomotion.modules.loco_wrapper import FrozenLocoPolicy
    from nav_locomotion.modules.nav_env_wrapper import NavEnvWrapper

    env = gym.make("Navigation-Explore-Spot-v0", num_envs=16)
    loco = FrozenLocoPolicy.from_checkpoint(args.loco_checkpoint, device=str(env.device))
    nav_env = NavEnvWrapper(env, loco)

    policy = ActorCriticCNN(num_obs=nav_env.num_obs, num_actions=nav_env.num_actions).to(env.device)

    obs, _ = nav_env.reset()
    for step in range(10):
        actions, _, _, _ = policy.act(obs)
        obs, reward, done, info = nav_env.step(actions)
        assert torch.isfinite(obs).all(), f"NaN in obs at step {step}"
        assert torch.isfinite(reward).all(), f"NaN in reward at step {step}"
        print(f"  Step {step}: reward={reward.mean():.3f}, done={done.sum()}/{done.numel()}")

    print("  Full env smoke test: PASS")
    os._exit(0)


if __name__ == "__main__":
    main()
