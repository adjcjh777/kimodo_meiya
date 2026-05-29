#!/usr/bin/env python3
"""Generate a test walking motion using Kimodo G1 34-joint model."""

import sys
from pathlib import Path
import numpy as np
import torch

# Add kimodo to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from kimodo.model.loading import load_model
from kimodo.skeleton import build_skeleton

def main():
    print("Loading Kimodo G1 34-joint model...")
    model, _ = load_model(
        "kimodo-g1-rp-v1",
        return_motion_representation=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")

    # Generate a simple walking motion
    print("\nGenerating test walking motion...")
    prompt = "a person walking forward"
    duration_seconds = 3.0
    fps = model.fps
    num_frames = int(duration_seconds * fps)

    print(f"  Prompt: {prompt}")
    print(f"  Duration: {duration_seconds}s ({num_frames} frames at {fps} FPS)")

    with torch.no_grad():
        output = model.generate(
            prompts=[prompt],
            num_frames=num_frames,
            batch_size=1,
            guidance_scale=2.5,
            seed=42
        )

    # Extract motion data
    local_rot_mats = output["local_rot_mats"][0].cpu().numpy()  # [T, 34, 3, 3]
    root_positions = output["root_positions"][0].cpu().numpy()  # [T, 3]

    print(f"\nGenerated motion shape:")
    print(f"  local_rot_mats: {local_rot_mats.shape}")
    print(f"  root_positions: {root_positions.shape}")

    # Save as Kimodo NPZ
    output_path = Path(__file__).with_name("test_walk_34j.npz")
    np.savez(output_path,
             local_rot_mats=local_rot_mats,
             root_positions=root_positions)
    print(f"\nSaved to: {output_path}")

    return output_path

if __name__ == "__main__":
    main()
