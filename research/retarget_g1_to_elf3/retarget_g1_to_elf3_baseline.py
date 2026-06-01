#!/usr/bin/env python3
"""Baseline qpos retarget from Kimodo G1 to BXI ELF3.

This is intentionally a direct semantic qpos remap, not task-space IK. It is
meant to create a fast visual baseline so signs, offsets, root handling, and
joint limits can be inspected in MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.retarget_g1_to_elf3.visualize_mujoco_g1 import load_qpos36  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MAP = THIS_DIR / "joint_map_g1_to_elf3.json"
DEFAULT_ELF3_XML = THIS_DIR / "assets" / "bxi_elf3" / "xmls" / "elf3.xml"
DEFAULT_G1_XML = REPO_ROOT / "kimodo" / "assets" / "skeletons" / "g1skel34" / "xml" / "g1.xml"
DEFAULT_FPS = 30.0
DEFAULT_FOOT_GROUND_CLEARANCE = 0.0
ROOT_QPOS_COLUMNS = (
    "root_x",
    "root_y",
    "root_z",
    "root_quat_w",
    "root_quat_x",
    "root_quat_y",
    "root_quat_z",
)
ELF3_JOINT_NAMES = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    "l_hip_y_joint",
    "l_hip_x_joint",
    "l_hip_z_joint",
    "l_knee_y_joint",
    "l_ankle_y_joint",
    "l_ankle_x_joint",
    "r_hip_y_joint",
    "r_hip_x_joint",
    "r_hip_z_joint",
    "r_knee_y_joint",
    "r_ankle_y_joint",
    "r_ankle_x_joint",
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",
    "r_shoulder_x_joint",
    "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
)
ELF3_BODY_NAMES = (
    "torso_link",
    "waist_y_link",
    "waist_x_link",
    "waist_z_link",
    "l_hip_y_link",
    "l_hip_x_link",
    "l_hip_z_link",
    "l_knee_y_link",
    "l_ankle_y_link",
    "l_ankle_x_link",
    "r_hip_y_link",
    "r_hip_x_link",
    "r_hip_z_link",
    "r_knee_y_link",
    "r_ankle_y_link",
    "r_ankle_x_link",
    "l_shoulder_y_link",
    "l_shoulder_x_link",
    "l_shoulder_z_link",
    "l_elbow_y_link",
    "l_wrist_x_link",
    "l_wrist_y_link",
    "l_wrist_z_link",
    "r_shoulder_y_link",
    "r_shoulder_x_link",
    "r_shoulder_z_link",
    "r_elbow_y_link",
    "r_wrist_x_link",
    "r_wrist_y_link",
    "r_wrist_z_link",
)
ELF3_QPOS_COLUMNS = (
    *ROOT_QPOS_COLUMNS,
    *ELF3_JOINT_NAMES,
)


def load_map(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    joints = mapping.get("joints")
    if not isinstance(joints, list) or len(joints) != 29:
        raise ValueError(f"{path} must contain 29 joint mappings")
    return mapping


def qpos_ranges(model: mujoco.MjModel) -> dict[int, tuple[float, float]]:
    ranges: dict[int, tuple[float, float]] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qadr = int(model.jnt_qposadr[j])
        ranges[qadr] = (float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1]))
    return ranges


def foot_collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(("l_foot", "r_foot")) and name.endswith("_collision"):
            geom_ids.append(geom_id)
    if not geom_ids:
        raise ValueError("No ELF3 foot collision geoms found; cannot apply foot-ground correction")
    return geom_ids


def geom_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    # Capsules are aligned with the MuJoCo geom local z-axis.
    radius = float(model.geom_size[geom_id, 0])
    half_len = float(model.geom_size[geom_id, 1])
    axis_z = float(data.geom_xmat[geom_id].reshape(3, 3)[2, 2])
    return float(data.geom_xpos[geom_id, 2]) - radius - half_len * abs(axis_z)


def foot_bottom_heights(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    geom_ids = foot_collision_geom_ids(model)
    bottoms = np.zeros(qpos.shape[0], dtype=np.float32)
    for frame_idx, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        bottoms[frame_idx] = min(geom_bottom_z(model, data, geom_id) for geom_id in geom_ids)
    return bottoms


def apply_foot_ground_correction(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    ground_clearance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    foot_bottom_before = foot_bottom_heights(model, qpos)
    root_z_correction = (ground_clearance - foot_bottom_before).astype(np.float32)
    corrected = qpos.copy()
    corrected[:, 2] += root_z_correction
    return corrected.astype(np.float32), foot_bottom_before, root_z_correction


def model_hinge_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        names.append(name or f"joint_{joint_id}")
    return tuple(names)


def model_body_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names: list[str] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        names.append(name or f"body_{body_id}")
    return tuple(names)


def qvel_from_qpos(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    qpos64 = qpos.astype(np.float64)
    qvel64 = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    if qpos.shape[0] < 2:
        return qvel64.astype(np.float32)
    dt = 1.0 / fps
    for frame_idx in range(1, qpos.shape[0]):
        mujoco.mj_differentiatePos(model, qvel64[frame_idx], dt, qpos64[frame_idx - 1], qpos64[frame_idx])
    qvel64[0] = qvel64[1]
    return qvel64.astype(np.float32)


def angular_velocity_from_quats(quats: np.ndarray, fps: float) -> np.ndarray:
    ang_vel = np.zeros(quats.shape[:-1] + (3,), dtype=np.float32)
    if quats.shape[0] < 2:
        return ang_vel

    dt = 1.0 / fps
    for frame_idx in range(1, quats.shape[0]):
        for body_idx in range(quats.shape[1]):
            prev_inv = np.array(
                [
                    quats[frame_idx - 1, body_idx, 0],
                    -quats[frame_idx - 1, body_idx, 1],
                    -quats[frame_idx - 1, body_idx, 2],
                    -quats[frame_idx - 1, body_idx, 3],
                ],
                dtype=np.float64,
            )
            rel = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(rel, quats[frame_idx, body_idx].astype(np.float64), prev_inv)
            vel = np.zeros(3, dtype=np.float64)
            mujoco.mju_quat2Vel(vel, rel, dt)
            ang_vel[frame_idx, body_idx] = vel
    ang_vel[0] = ang_vel[1]
    return ang_vel


def body_trajectory_from_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    body_pos = np.zeros((qpos.shape[0], model.nbody - 1, 3), dtype=np.float32)
    body_quat = np.zeros((qpos.shape[0], model.nbody - 1, 4), dtype=np.float32)

    for frame_idx, row in enumerate(qpos):
        data.qpos[:] = row
        data.qvel[:] = qvel[frame_idx]
        mujoco.mj_forward(model, data)
        body_pos[frame_idx] = data.xpos[1:]
        body_quat[frame_idx] = data.xquat[1:]

    # Match the reference NPZ field contract. Linear/angular velocities are in
    # world coordinates and follow the body order with world body excluded.
    if qpos.shape[0] > 1:
        body_lin_vel = np.gradient(body_pos, 1.0 / fps, axis=0).astype(np.float32)
    else:
        body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = angular_velocity_from_quats(body_quat, fps)
    return body_pos, body_quat, body_lin_vel, body_ang_vel


def mjlab_compatible_fields(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    fps: float,
    joint_names: tuple[str, ...] | None = None,
    body_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    if joint_names is None:
        joint_names = model_hinge_joint_names(model)
    if body_names is None:
        body_names = model_body_names(model)
    if len(joint_names) != qpos.shape[1] - 7:
        raise ValueError(f"Expected {qpos.shape[1] - 7} joint names, got {len(joint_names)}")
    if len(body_names) != model.nbody - 1:
        raise ValueError(f"Expected {model.nbody - 1} body names, got {len(body_names)}")

    qvel = qvel_from_qpos(model, qpos, fps)
    body_pos, body_quat, body_lin_vel, body_ang_vel = body_trajectory_from_qpos(model, qpos, qvel, fps)
    return {
        "fps": np.array([fps], dtype=np.float64),
        "joint_pos": qpos[:, 7:].astype(np.float32),
        "joint_vel": qvel[:, 6:].astype(np.float32),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "joint_names": np.array(joint_names),
        "body_names": np.array(body_names),
    }


def retarget_qpos(
    g1_qpos: np.ndarray,
    mapping: dict[str, Any],
    elf3_model: mujoco.MjModel,
    *,
    root_z_offset: float | None,
    clamp: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if g1_qpos.ndim != 2 or g1_qpos.shape[1] != 36:
        raise ValueError(f"Expected G1 qpos shape [T, 36], got {g1_qpos.shape}")
    if elf3_model.nq != 36:
        raise ValueError(f"Expected ELF3 model.nq=36, got {elf3_model.nq}")

    elf3_qpos = np.zeros_like(g1_qpos, dtype=np.float32)
    elf3_qpos[:, :7] = g1_qpos[:, :7]

    if root_z_offset is None:
        root_z_offset = float(mapping.get("root", {}).get("recommended_initial_z_offset", 0.0))
    elf3_qpos[:, 2] += root_z_offset

    range_by_qpos = qpos_ranges(elf3_model)
    clip_reports: list[dict[str, Any]] = []

    for entry in mapping["joints"]:
        src = int(entry["g1_qpos"])
        dst = int(entry["elf3_qpos"])
        sign = float(entry.get("sign", 1.0))
        offset = float(entry.get("offset", 0.0))
        values = sign * g1_qpos[:, src] + offset

        report: dict[str, Any] = {
            "joint": entry["elf3"],
            "min_before": float(np.min(values)),
            "max_before": float(np.max(values)),
            "clipped_low": 0,
            "clipped_high": 0,
        }
        if clamp and dst in range_by_qpos:
            lo, hi = range_by_qpos[dst]
            report["clipped_low"] = int(np.sum(values < lo))
            report["clipped_high"] = int(np.sum(values > hi))
            values = np.clip(values, lo, hi)
            report["range"] = [lo, hi]
        report["min_after"] = float(np.min(values))
        report["max_after"] = float(np.max(values))
        if report["clipped_low"] or report["clipped_high"]:
            clip_reports.append(report)
        elf3_qpos[:, dst] = values

    return elf3_qpos, clip_reports


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    mapping: dict[str, Any],
    elf3_model: mujoco.MjModel,
    g1_model: mujoco.MjModel,
    g1_xml: Path,
    root_z_offset: float | None,
    clamp: bool,
    foot_ground_correction: bool,
    ground_clearance: float,
    save_csv: bool,
    fps: float,
) -> None:
    g1_qpos = load_qpos36(str(input_path), str(g1_xml))
    elf3_qpos, clip_reports = retarget_qpos(
        g1_qpos,
        mapping,
        elf3_model,
        root_z_offset=root_z_offset,
        clamp=clamp,
    )

    foot_bottom_before = np.zeros(elf3_qpos.shape[0], dtype=np.float32)
    root_z_ground_correction = np.zeros(elf3_qpos.shape[0], dtype=np.float32)
    if foot_ground_correction:
        elf3_qpos, foot_bottom_before, root_z_ground_correction = apply_foot_ground_correction(
            elf3_model,
            elf3_qpos,
            ground_clearance=ground_clearance,
        )

    src = np.load(input_path, allow_pickle=False)
    g1_qpos_columns = (*ROOT_QPOS_COLUMNS, *model_hinge_joint_names(g1_model))
    save_dict: dict[str, Any] = {
        "qpos_elf3": elf3_qpos,
        "qpos_g1": g1_qpos,
        "source_file": str(input_path),
        "joint_map": json.dumps(mapping, ensure_ascii=True),
        "qpos_elf3_columns": np.array(ELF3_QPOS_COLUMNS),
        "qpos_g1_columns": np.array(g1_qpos_columns),
        "qpos_columns": np.array(ELF3_QPOS_COLUMNS),
        "root_z_offset": float(root_z_offset if root_z_offset is not None else mapping["root"]["recommended_initial_z_offset"]),
        "foot_ground_correction_enabled": np.array([foot_ground_correction], dtype=np.bool_),
        "foot_ground_clearance": np.array([ground_clearance], dtype=np.float32),
        "foot_bottom_before_ground_correction": foot_bottom_before,
        "root_z_ground_correction": root_z_ground_correction,
        **mjlab_compatible_fields(
            elf3_model,
            elf3_qpos,
            fps=fps,
            joint_names=ELF3_JOINT_NAMES,
            body_names=ELF3_BODY_NAMES,
        ),
    }
    for key in ("foot_contacts", "global_root_heading", "smooth_root_pos"):
        if key in src:
            save_dict[key] = src[key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **save_dict)
    print(f"saved {output_path}: qpos_elf3 {elf3_qpos.shape}")
    if save_csv:
        csv_path = output_path.with_suffix(".csv")
        np.savetxt(csv_path, elf3_qpos, delimiter=",")
        print(f"saved {csv_path}: columns={len(ELF3_QPOS_COLUMNS)}")
    if clip_reports:
        print("  clipped joints:")
        for report in clip_reports:
            print(
                "   "
                f"{report['joint']}: low={report['clipped_low']} high={report['clipped_high']} "
                f"before=[{report['min_before']:.3f}, {report['max_before']:.3f}] "
                f"after=[{report['min_after']:.3f}, {report['max_after']:.3f}]"
            )
    if foot_ground_correction:
        print(
            "  foot-ground correction: "
            f"foot_bottom_before=[{float(foot_bottom_before.min()):.3f}, "
            f"{float(np.median(foot_bottom_before)):.3f}, {float(foot_bottom_before.max()):.3f}] "
            f"root_z_shift=[{float(root_z_ground_correction.min()):.3f}, "
            f"{float(np.median(root_z_ground_correction)):.3f}, {float(root_z_ground_correction.max()):.3f}]"
        )


def iter_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.npz"))
    return [input_path]


def output_for(input_file: Path, input_root: Path, output_path: Path) -> Path:
    if input_root.is_dir():
        return output_path / input_file.name
    if output_path.suffix == ".npz":
        return output_path
    return output_path / input_file.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="G1 NPZ file or directory")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output ELF3 NPZ file or directory")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    parser.add_argument("--elf3-xml", type=Path, default=DEFAULT_ELF3_XML)
    parser.add_argument("--root-z-offset", type=float, default=None)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--no-foot-ground-correction",
        action="store_true",
        help="Do not align ELF3 foot collision bottoms to the ground after retargeting.",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=DEFAULT_FOOT_GROUND_CLEARANCE,
        help="Target minimum foot clearance after correction.",
    )
    parser.add_argument("--no-clamp", action="store_true", help="Do not clamp to ELF3 joint limits.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write ELF3 qpos CSV sidecars.")
    args = parser.parse_args()

    if args.ground_clearance < 0:
        raise ValueError("--ground-clearance must be non-negative")

    mapping = load_map(args.map)
    elf3_model = mujoco.MjModel.from_xml_path(str(args.elf3_xml))
    g1_model = mujoco.MjModel.from_xml_path(str(args.g1_xml))
    inputs = iter_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No NPZ files found: {args.input}")

    for input_file in inputs:
        convert_file(
            input_file,
            output_for(input_file, args.input, args.output),
            mapping=mapping,
            elf3_model=elf3_model,
            g1_model=g1_model,
            g1_xml=args.g1_xml,
            root_z_offset=args.root_z_offset,
            clamp=not args.no_clamp,
            foot_ground_correction=not args.no_foot_ground_correction,
            ground_clearance=args.ground_clearance,
            save_csv=not args.no_csv,
            fps=args.fps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
