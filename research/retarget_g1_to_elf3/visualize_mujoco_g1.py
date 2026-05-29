#!/usr/bin/env python3
"""Visualize original Kimodo G1 motion in MuJoCo.

This reads Kimodo NPZ output and plays it on the repository's floating-base
G1 MJCF model (`g1.xml`). It does not require torch; it reuses the pure NumPy
qpos conversion helper from the 23DOF investigation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.retarget_23dof_g1.npz_global_to_23dof import (  # noqa: E402
    G1_XML,
    local_rots_to_qpos36,
    recover_local_rots,
)


def load_qpos36(npz_path: str, xml_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=False)

    if "qpos36" in data:
        qpos = data["qpos36"]
    elif "qpos_g1" in data:
        qpos = data["qpos_g1"]
    elif "qpos" in data and data["qpos"].shape[-1] == 36:
        qpos = data["qpos"]
    elif "local_rot_mats" in data and "root_positions" in data:
        qpos = local_rots_to_qpos36(data["local_rot_mats"], data["root_positions"], xml_path)
    elif "global_rot_mats" in data and "posed_joints" in data:
        local_rots = recover_local_rots(data["global_rot_mats"])
        root_positions = data["posed_joints"][:, 0]
        qpos = local_rots_to_qpos36(local_rots, root_positions, xml_path)
    else:
        raise ValueError(
            "Unsupported NPZ. Expected qpos36/qpos/qpos_g1, local_rot_mats+root_positions, "
            f"or global_rot_mats+posed_joints. Available keys: {list(data.keys())}"
        )

    if qpos.ndim == 3:
        qpos = qpos[0]
    if qpos.shape[-1] != 36:
        raise ValueError(f"Expected qpos shape [T, 36], got {qpos.shape}")
    return qpos


def visualize_motion(model: mujoco.MjModel, qpos_data: np.ndarray, fps: float) -> None:
    data = mujoco.MjData(model)
    n_frames = qpos_data.shape[0]

    print(f"Loaded motion: {n_frames} frames, model.nq={model.nq}")
    print(f"Playing at {fps} FPS ({n_frames / fps:.2f} seconds)")
    print("\nViewer controls:")
    print("  - Left mouse: rotate camera")
    print("  - Right mouse: pan camera")
    print("  - Scroll: zoom")
    print("  - Space: pause/resume")
    print("  - ESC: exit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        paused = False

        while viewer.is_running():
            start_time = time.time()

            if not paused:
                data.qpos[:] = qpos_data[frame_idx]
                mujoco.mj_forward(model, data)
                frame_idx = (frame_idx + 1) % n_frames

            viewer.sync()

            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize original Kimodo G1 NPZ in MuJoCo")
    parser.add_argument("motion", help="Path to generated Kimodo G1 NPZ")
    parser.add_argument("--xml", default=G1_XML, help="Path to Kimodo G1 MJCF XML")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    print(f"Loading motion: {args.motion}")
    qpos = load_qpos36(args.motion, args.xml)
    print(f"Using qpos36: shape {qpos.shape}")

    print(f"Loading MJCF: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    if model.nq != qpos.shape[1]:
        raise ValueError(f"Model expects nq={model.nq}, but qpos has {qpos.shape[1]} columns")

    visualize_motion(model, qpos, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
