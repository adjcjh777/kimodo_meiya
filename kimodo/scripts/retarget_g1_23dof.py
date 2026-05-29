#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Retarget Kimodo G1 34-joint (29-DOF) motion to Unitree G1 23-DOF.

Kimodo G1 model outputs 34 joints (29-DOF + 4 end-effectors + root).
The 23-DOF variant locks waist roll/pitch and removes wrist pitch/yaw.

Usage:
    # From Kimodo NPZ -> 23-DOF NPZ
    python -m kimodo.scripts.retarget_g1_23dof input.npz -o output_23dof.npz

    # Batch convert all NPZ files in a directory
    python -m kimodo.scripts.retarget_g1_23dof input_dir/ -o output_dir/
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Kimodo G1 34-joint CSV (36-dim qpos) joint layout:
#   [0:7]   = root (pos xyz + quat wxyz)
#   [7:13]  = left leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
#   [13:19] = right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
#   [19]    = waist_yaw
#   [20]    = waist_roll          <-- DROP (locked in 23-DOF)
#   [21]    = waist_pitch         <-- DROP (locked in 23-DOF)
#   [22:27] = left arm:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
#   [27]    = left_wrist_pitch    <-- DROP (not in 23-DOF)
#   [28]    = left_wrist_yaw      <-- DROP (not in 23-DOF)
#   [29:34] = right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
#   [34]    = right_wrist_pitch   <-- DROP (not in 23-DOF)
#   [35]    = right_wrist_yaw     <-- DROP (not in 23-DOF)

# Indices into the 36-dim qpos to keep for 23-DOF (7 root + 23 joints = 30)
QPOS36_KEEP_INDICES = (
    list(range(0, 7))     # root (7)
    + list(range(7, 19))  # left leg + right leg (12)
    + [19]                # waist_yaw (1)
    + list(range(22, 27)) # left arm 5 joints
    + list(range(29, 34)) # right arm 5 joints
)  # total: 7 + 12 + 1 + 5 + 5 = 30

# Kimodo G1 34-joint bone_order indices to keep for 23-DOF skeleton
# (0-indexed into bone_order_names_with_parents)
BONE34_KEEP_INDICES = (
    [0]                   # pelvis (root)
    + list(range(1, 7))   # left leg (6)
    + list(range(8, 14))  # right leg (6)
    + [15]                # waist_yaw
    + list(range(18, 23)) # left arm (shoulder×3 + elbow + wrist_roll)
    + list(range(26, 31)) # right arm (shoulder×3 + elbow + wrist_roll)
)  # total: 1 + 6 + 6 + 1 + 5 + 5 = 24 bones

# Joint names for the 23-DOF output (matching URDF joint order)
JOINT_NAMES_23DOF = [
    # Left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
    # Left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    # Right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]


def retarget_qpos36_to_23dof(qpos36: np.ndarray) -> np.ndarray:
    """Convert (T, 36) Kimodo G1 qpos to (T, 30) 23-DOF qpos.

    Args:
        qpos36: Array of shape (T, 36) with Kimodo's 29-DOF G1 qpos format.

    Returns:
        Array of shape (T, 30) with 23-DOF qpos (root 7 + 23 joint angles).
    """
    if qpos36.ndim == 1:
        qpos36 = qpos36[np.newaxis]
    if qpos36.shape[-1] != 36:
        raise ValueError(f"Expected 36-dim qpos, got shape {qpos36.shape}")
    return qpos36[:, QPOS36_KEEP_INDICES]


def retarget_kimodo_npz_to_23dof(
    input_path: str,
    output_path: str,
) -> None:
    """Load a Kimodo G1 NPZ and save as 23-DOF NPZ.

    Output NPZ contains:
    - local_rot_mats: [T, 24, 3, 3] local rotation matrices for 23-DOF skeleton
    - global_rot_mats: [T, 24, 3, 3] global rotation matrices
    - posed_joints: [T, 24, 3] joint positions
    - root_positions: [T, 3] root trajectory
    - qpos_23dof: [T, 30] qpos format (7 root + 23 joint angles)
    - foot_contacts: [T, 4] foot contact labels (if present in input)
    """
    from kimodo.skeleton import G1Skeleton34, build_skeleton
    from kimodo.exports.mujoco import MujocoQposConverter
    import torch

    # Load the Kimodo NPZ
    data = np.load(input_path, allow_pickle=False)
    local_rot_mats = data["local_rot_mats"]
    root_positions = data["root_positions"]

    # Handle batched input [B, T, J, ...] -> take first sample
    if local_rot_mats.ndim == 5:
        local_rot_mats = local_rot_mats[0]
    if root_positions.ndim == 3:
        root_positions = root_positions[0]

    T, J = local_rot_mats.shape[:2]
    print(f"Loaded Kimodo NPZ: {T} frames, {J} joints")

    # Build skeleton and converter to get 36-dim qpos
    skel = build_skeleton(34)
    converter = MujocoQposConverter(skel)

    # Convert to 36-dim qpos
    qpos36 = converter.dict_to_qpos(
        {
            "local_rot_mats": torch.from_numpy(local_rot_mats).float(),
            "root_positions": torch.from_numpy(root_positions).float(),
        },
        device="cpu",
        numpy=True,
    )
    print(f"Converted to 36-dim qpos: shape {qpos36.shape}")

    # Slice to 30-dim (23 DOF + root 7)
    qpos30 = retarget_qpos36_to_23dof(qpos36)
    print(f"Retargeted to 30-dim qpos (23 DOF): shape {qpos30.shape}")

    # Build 23-DOF NPZ with the kept joint subset
    # local_rot_mats: [T, 24, 3, 3]
    local_rot_23 = local_rot_mats[:, BONE34_KEEP_INDICES]
    posed_joints = data.get("posed_joints")
    global_rot_mats = data.get("global_rot_mats")

    save_dict = {
        "local_rot_mats": local_rot_23,
        "root_positions": root_positions,
        "qpos_23dof": qpos30,
    }

    if posed_joints is not None:
        if posed_joints.ndim == 4:
            posed_joints = posed_joints[0]
        save_dict["posed_joints_34"] = posed_joints  # keep full 34 for reference

    if global_rot_mats is not None:
        if global_rot_mats.ndim == 5:
            global_rot_mats = global_rot_mats[0]
        save_dict["global_rot_mats_23"] = global_rot_mats[:, BONE34_KEEP_INDICES]

    if "foot_contacts" in data:
        fc = data["foot_contacts"]
        if fc.ndim == 3:
            fc = fc[0]
        save_dict["foot_contacts"] = fc

    np.savez(output_path, **save_dict)
    print(f"Saved 23-DOF Kimodo NPZ to {output_path}")

    # Print summary
    joint_angles = qpos30[:, 7:]  # skip root
    print(f"\nJoint angle ranges (degrees):")
    for i, name in enumerate(JOINT_NAMES_23DOF):
        deg = np.degrees(joint_angles[:, i])
        print(f"  {name:35s}  [{deg.min():7.1f}, {deg.max():7.1f}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retarget Kimodo G1 (34-joint, 29-DOF) motion to G1 23-DOF.",
    )
    parser.add_argument("input", help="Input file (.npz) or directory")
    parser.add_argument("-o", "--output", required=True, help="Output file (.npz) or directory")
    args = parser.parse_args(argv)

    input_path = args.input
    output_path = args.output

    # Handle directory input
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        npz_files = [f for f in os.listdir(input_path) if f.endswith(".npz")]
        if not npz_files:
            print(f"No .npz files found in {input_path}", file=sys.stderr)
            return 1

        print(f"Processing {len(npz_files)} NPZ files...")
        for filename in npz_files:
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)
            print(f"\n{'='*60}")
            print(f"Processing: {filename}")
            print('='*60)
            try:
                retarget_kimodo_npz_to_23dof(input_file, output_file)
            except Exception as e:
                print(f"Error processing {filename}: {e}", file=sys.stderr)
                continue
        print(f"\nAll files processed. Output saved to {output_path}")
    else:
        # Single file
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}", file=sys.stderr)
            return 1

        # Ensure output has .npz extension
        if not output_path.endswith(".npz"):
            output_path += ".npz"

        retarget_kimodo_npz_to_23dof(input_path, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
