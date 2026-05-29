#!/usr/bin/env python3
"""Convert Kimodo G1 demo NPZ files to 23DOF qpos NPZ without torch."""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np


G1_XML = "kimodo/assets/skeletons/g1skel34/xml/g1.xml"

G1_BONES = [
    ("pelvis_skel", None),
    ("left_hip_pitch_skel", "pelvis_skel"),
    ("left_hip_roll_skel", "left_hip_pitch_skel"),
    ("left_hip_yaw_skel", "left_hip_roll_skel"),
    ("left_knee_skel", "left_hip_yaw_skel"),
    ("left_ankle_pitch_skel", "left_knee_skel"),
    ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
    ("left_toe_base", "left_ankle_roll_skel"),
    ("right_hip_pitch_skel", "pelvis_skel"),
    ("right_hip_roll_skel", "right_hip_pitch_skel"),
    ("right_hip_yaw_skel", "right_hip_roll_skel"),
    ("right_knee_skel", "right_hip_yaw_skel"),
    ("right_ankle_pitch_skel", "right_knee_skel"),
    ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
    ("right_toe_base", "right_ankle_roll_skel"),
    ("waist_yaw_skel", "pelvis_skel"),
    ("waist_roll_skel", "waist_yaw_skel"),
    ("waist_pitch_skel", "waist_roll_skel"),
    ("left_shoulder_pitch_skel", "waist_pitch_skel"),
    ("left_shoulder_roll_skel", "left_shoulder_pitch_skel"),
    ("left_shoulder_yaw_skel", "left_shoulder_roll_skel"),
    ("left_elbow_skel", "left_shoulder_yaw_skel"),
    ("left_wrist_roll_skel", "left_elbow_skel"),
    ("left_wrist_pitch_skel", "left_wrist_roll_skel"),
    ("left_wrist_yaw_skel", "left_wrist_pitch_skel"),
    ("left_hand_roll_skel", "left_wrist_yaw_skel"),
    ("right_shoulder_pitch_skel", "waist_pitch_skel"),
    ("right_shoulder_roll_skel", "right_shoulder_pitch_skel"),
    ("right_shoulder_yaw_skel", "right_shoulder_roll_skel"),
    ("right_elbow_skel", "right_shoulder_yaw_skel"),
    ("right_wrist_roll_skel", "right_elbow_skel"),
    ("right_wrist_pitch_skel", "right_wrist_roll_skel"),
    ("right_wrist_yaw_skel", "right_wrist_pitch_skel"),
    ("right_hand_roll_skel", "right_wrist_yaw_skel"),
]

QPOS36_KEEP_INDICES = (
    list(range(0, 7))
    + list(range(7, 19))
    + [19]
    + list(range(22, 27))
    + list(range(29, 34))
)

BONE34_KEEP_INDICES = (
    [0]
    + list(range(1, 7))
    + list(range(8, 14))
    + [15]
    + list(range(18, 23))
    + list(range(26, 31))
)


def quat_wxyz_to_matrix(q: list[float] | np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm([w, x, y, z])
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    tr = float(np.trace(m))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def recover_local_rots(global_rots: np.ndarray) -> np.ndarray:
    name_to_idx = {name: i for i, (name, _) in enumerate(G1_BONES)}
    local = np.empty_like(global_rots)
    for i, (_, parent) in enumerate(G1_BONES):
        if parent is None:
            local[:, i] = global_rots[:, i]
        else:
            p = name_to_idx[parent]
            local[:, i] = np.einsum("tji,tjk->tik", global_rots[:, p], global_rots[:, i])
    return local


def parse_mujoco_mapping(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    class_axes = {}
    for xml_class in tree.findall(".//default"):
        cls = xml_class.get("class")
        joints = xml_class.findall("joint")
        if cls and joints and joints[0].get("axis"):
            class_axes[cls] = joints[0].get("axis")

    parent_map = {child: parent for parent in root.iter() for child in parent}
    hinge_joints = root.find("worldbody").findall(".//joint")
    bone_names = [name for name, _ in G1_BONES]
    mujoco_to_kimodo = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    rot_offsets_f2q = np.repeat(np.eye(3)[None], len(G1_BONES), axis=0)
    axes_kimodo = []
    kimodo_indices = []

    for joint in hinge_joints:
        skel_name = joint.get("name").replace("_joint", "_skel")
        idx = bone_names.index(skel_name)
        axis_text = joint.get("axis") or class_axes[joint.get("class")]
        axis = np.array([float(x) for x in axis_text.split()])
        axes_kimodo.append([np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])][int(np.argmax(axis))])
        kimodo_indices.append(idx)

        body = parent_map[joint]
        if body.get("quat"):
            rot = quat_wxyz_to_matrix([float(x) for x in body.get("quat").split()])
            rot_offsets_f2q[idx] = (mujoco_to_kimodo @ rot @ mujoco_to_kimodo.T).T

    return np.asarray(kimodo_indices), np.asarray(axes_kimodo, dtype=np.float64), rot_offsets_f2q


def local_rots_to_qpos36(local_rots: np.ndarray, root_positions: np.ndarray, xml_path: str) -> np.ndarray:
    kimodo_indices, axes_kimodo, offsets_f2q = parse_mujoco_mapping(xml_path)
    local_f2q = np.einsum("jmn,tjnk->tjmk", offsets_f2q, local_rots)

    kimodo_to_mujoco = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mujoco_to_kimodo = kimodo_to_mujoco.T

    qpos = np.zeros((local_rots.shape[0], 36), dtype=np.float32)
    qpos[:, :3] = np.einsum("ij,tj->ti", kimodo_to_mujoco, root_positions)

    for t in range(local_rots.shape[0]):
        root_mujoco = kimodo_to_mujoco @ local_f2q[t, 0] @ mujoco_to_kimodo
        qpos[t, 3:7] = matrix_to_quat_wxyz(root_mujoco)

    hinge = local_f2q[:, kimodo_indices]
    x_dof = np.arctan2(hinge[..., 2, 1], hinge[..., 2, 2])
    y_dof = np.arctan2(hinge[..., 0, 2], hinge[..., 0, 0])
    z_dof = np.arctan2(hinge[..., 1, 0], hinge[..., 1, 1])
    xyz = np.stack([x_dof, y_dof, z_dof], axis=-1)
    qpos[:, 7:] = np.sum(xyz * axes_kimodo[None], axis=-1)
    return qpos


def convert(input_path: str, output_path: str, xml_path: str) -> None:
    data = np.load(input_path, allow_pickle=False)
    global_rots = data["global_rot_mats"]
    posed_joints = data["posed_joints"]
    local_rots = recover_local_rots(global_rots)
    root_positions = posed_joints[:, 0]
    qpos36 = local_rots_to_qpos36(local_rots, root_positions, xml_path)
    qpos23 = qpos36[:, QPOS36_KEEP_INDICES]

    save = {
        "qpos_23dof": qpos23,
        "local_rot_mats": local_rots[:, BONE34_KEEP_INDICES],
        "root_positions": root_positions,
        "posed_joints_34": posed_joints,
        "global_rot_mats_23": global_rots[:, BONE34_KEEP_INDICES],
    }
    if "foot_contacts" in data:
        save["foot_contacts"] = data["foot_contacts"]
    np.savez(output_path, **save)
    print(f"saved {output_path}: qpos_23dof {qpos23.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--xml", default=G1_XML)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    convert(args.input, args.output, args.xml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
