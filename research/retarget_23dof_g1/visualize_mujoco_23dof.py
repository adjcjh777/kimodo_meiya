#!/usr/bin/env python3
"""Visualize 23 DOF G1 motion in MuJoCo viewer.

Usage:
    python visualize_mujoco_23dof.py <motion.npz>

Args:
    motion.npz: Path to the 23 DOF retargeted NPZ file (output from retarget_g1_23dof.py)
"""

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

DEFAULT_URDF = Path(__file__).with_name("g1_23dof_mujoco_viewer.urdf")


def load_urdf_model(urdf_path: str) -> mujoco.MjModel:
    """Load URDF file into MuJoCo model.

    Handles the common issue where URDF mesh filenames already include a directory
    prefix (e.g. ``meshes/foo.STL``) that conflicts with the ``<compiler meshdir>``
    attribute, causing double-prefixed paths like ``meshes/meshes/foo.STL``.
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    original_cwd = os.getcwd()

    # Read URDF and fix double-prefixed mesh paths:
    # The URDF has meshdir="meshes" AND filenames like "meshes/foo.STL",
    # so MuJoCo resolves to "meshes/meshes/foo.STL".  Strip the mujoco
    # compiler meshdir to let the URDF's own relative paths work.
    with open(urdf_path, "r") as f:
        urdf_text = f.read()
    # Replace meshdir="meshes" with meshdir="" so paths stay as-is
    urdf_text = urdf_text.replace('meshdir="meshes"', 'meshdir=""')

    try:
        os.chdir(urdf_dir)
        model = mujoco.MjModel.from_xml_string(urdf_text)
    finally:
        os.chdir(original_cwd)
    return model


def visualize_motion(model: mujoco.MjModel, qpos_data: np.ndarray, fps: float = 30.0):
    """Visualize motion in MuJoCo viewer.

    Args:
        model: MuJoCo model
        qpos_data: Joint positions, shape [T, nq] where nq is number of DoF
        fps: Frame rate
    """
    data = mujoco.MjData(model)
    n_frames = qpos_data.shape[0]

    print(f"Loaded motion: {n_frames} frames, {model.nq} DoF")
    print(f"Playing at {fps} FPS ({n_frames/fps:.2f} seconds)")
    print("\nViewer controls:")
    print("  - Left mouse: rotate camera")
    print("  - Right mouse: pan camera")
    print("  - Scroll: zoom")
    print("  - Space: pause/resume")
    print("  - ESC: exit")

    # Start viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        paused = False

        while viewer.is_running():
            start_time = time.time()

            if not paused:
                # Set joint positions
                data.qpos[:] = qpos_data[frame_idx]
                mujoco.mj_forward(model, data)

                # Advance frame (loop)
                frame_idx = (frame_idx + 1) % n_frames

            # Update viewer
            viewer.sync()

            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="Visualize 23 DOF G1 motion in MuJoCo")
    parser.add_argument("motion", help="Path to 23 DOF motion NPZ file")
    parser.add_argument(
        "--urdf",
        default=str(DEFAULT_URDF),
        help="Path to G1 23 DOF URDF file",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30)")
    args = parser.parse_args()

    # Load motion data
    print(f"Loading motion: {args.motion}")
    motion_data = np.load(args.motion)

    # Extract joint positions
    if "qpos_23dof" in motion_data:
        qpos = motion_data["qpos_23dof"]
        print(f"Using qpos_23dof: shape {qpos.shape}")
    elif "posed_joints_34" in motion_data:
        # Fall back to 34-joint data (first 23 joints)
        qpos = motion_data["posed_joints_34"][:, :23]
        print(f"Using posed_joints_34 (first 23 joints): shape {qpos.shape}")
    else:
        print(f"Available keys: {list(motion_data.keys())}")
        raise ValueError("No suitable joint position data found in NPZ")

    # Load MuJoCo model
    print(f"Loading URDF: {args.urdf}")
    model = load_urdf_model(args.urdf)

    # Check DoF compatibility.
    #
    # qpos_23dof from retarget_g1_23dof.py is 30 columns:
    #   root xyz (3) + root quat wxyz (4) + 23 revolute joint angles.
    # The current g1_23dof.urdf has its floating base joint commented out, so
    # MuJoCo loads it as a fixed-base model with model.nq == 23. In that case,
    # use only the joint-angle tail; truncating the first 23 columns would feed
    # root xyz/quaternion into leg joints.
    if qpos.shape[1] == 30 and model.nq == 23:
        print("Model is fixed-base 23-DoF; using qpos_23dof[:, 7:] joint angles.")
        qpos = qpos[:, 7:]
    elif qpos.shape[1] != model.nq:
        print(f"Warning: Motion has {qpos.shape[1]} DoF, but model expects {model.nq} DoF")
        print("Adjusting qpos to match model...")
        if qpos.shape[1] < model.nq:
            padding = np.zeros((qpos.shape[0], model.nq - qpos.shape[1]))
            qpos = np.concatenate([qpos, padding], axis=1)
        else:
            qpos = qpos[:, :model.nq]

    # Visualize
    visualize_motion(model, qpos, args.fps)


if __name__ == "__main__":
    main()
