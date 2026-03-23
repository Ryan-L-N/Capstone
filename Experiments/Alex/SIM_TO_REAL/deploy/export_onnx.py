"""Export trained policy to ONNX for edge deployment.

Converts the PyTorch actor network to ONNX format, verifiable against
the original PyTorch output (max diff < 1e-5).

Optional TensorRT optimization for NVIDIA Jetson onboard compute.

Usage:
    python deploy/export_onnx.py \
        --checkpoint checkpoints/student/best.pt \
        --output policy.onnx

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import numpy as np


def build_actor(hidden_dims, num_obs=235, num_actions=12):
    """Build a standalone actor network matching the training architecture."""
    layers = []
    input_dim = num_obs
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ELU())
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, num_actions))
    return nn.Sequential(*layers)


def export_to_onnx(checkpoint_path, output_path, hidden_dims=None):
    """Export actor network from checkpoint to ONNX.

    Args:
        checkpoint_path: Path to model_XXXX.pt checkpoint.
        output_path: Output .onnx file path.
        hidden_dims: Actor hidden dims (default [512, 256, 128]).
    """
    if hidden_dims is None:
        hidden_dims = [512, 256, 128]

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    # Detect obs dim from first layer
    first_weight = state.get("actor.0.weight")
    if first_weight is not None:
        num_obs = first_weight.shape[1]
        num_actions = state["actor.{}.weight".format(len(hidden_dims) * 2)].shape[0]
        print(f"Detected: {num_obs} obs dims, {num_actions} action dims")
    else:
        num_obs = 235
        num_actions = 12

    # Build actor and load weights
    actor = build_actor(hidden_dims, num_obs, num_actions)

    actor_state = {
        k.replace("actor.", ""): v
        for k, v in state.items()
        if k.startswith("actor.")
    }
    actor.load_state_dict(actor_state)
    actor.eval()

    # Create dummy input
    dummy_obs = torch.randn(1, num_obs)

    # PyTorch reference output
    with torch.no_grad():
        pytorch_output = actor(dummy_obs).numpy()

    # Export to ONNX
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        actor,
        dummy_obs,
        output_path,
        opset_version=11,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    # Verify ONNX output matches PyTorch
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        onnx_output = session.run(None, {"observation": dummy_obs.numpy()})[0]

        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        print(f"Verification: max diff = {max_diff:.2e} (threshold: 1e-5)")

        if max_diff < 1e-5:
            print("PASS: ONNX output matches PyTorch output")
        else:
            print("WARNING: Output mismatch exceeds threshold!")
    except ImportError:
        print("onnxruntime not installed — skipping verification")

    # File size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {size_mb:.2f} MB")
    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export policy to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.hidden_dims)
