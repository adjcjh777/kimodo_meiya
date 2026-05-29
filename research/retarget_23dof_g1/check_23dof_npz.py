#!/usr/bin/env python3
"""Headless sanity checks for retargeted G1 23DOF NPZ files."""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np


DEFAULT_URDF = (
    "/home/chengjunhao/qikorobotagent-import-initial-migration/"
    "web/public/robots/unitree/g1_23dof.urdf"
)


def load_urdf_model(urdf_path: str) -> mujoco.MjModel:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    original_cwd = os.getcwd()
    with open(urdf_path, "r") as f:
        urdf_text = f.read()
    urdf_text = urdf_text.replace('meshdir="meshes"', 'meshdir=""')

    try:
        os.chdir(urdf_dir)
        return mujoco.MjModel.from_xml_string(urdf_text)
    finally:
        os.chdir(original_cwd)


def joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        names.append(name or f"joint_{i}")
    return names


def select_model_qpos(model: mujoco.MjModel, qpos_23dof: np.ndarray) -> np.ndarray:
    if qpos_23dof.ndim != 2:
        raise ValueError(f"qpos_23dof must be [T, D], got {qpos_23dof.shape}")
    if qpos_23dof.shape[1] == model.nq:
        return qpos_23dof
    if qpos_23dof.shape[1] == 30 and model.nq == 23:
        return qpos_23dof[:, 7:]
    raise ValueError(f"Cannot map qpos_23dof shape {qpos_23dof.shape} to model.nq={model.nq}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", help="Retargeted 23DOF NPZ containing qpos_23dof")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Target G1 23DOF URDF")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit checked frames; 0 means all")
    args = parser.parse_args()

    motion = np.load(args.motion, allow_pickle=False)
    if "qpos_23dof" not in motion:
        raise ValueError(f"{args.motion} does not contain qpos_23dof; keys={list(motion.keys())}")

    model = load_urdf_model(args.urdf)
    qpos_src = motion["qpos_23dof"]
    qpos = select_model_qpos(model, qpos_src)
    if args.max_frames > 0:
        qpos = qpos[: args.max_frames]

    data = mujoco.MjData(model)
    z_min = float("inf")
    z_max = float("-inf")
    bad_frames = []

    for frame, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.xpos).all():
            bad_frames.append(frame)
            continue
        z_min = min(z_min, float(data.xpos[:, 2].min()))
        z_max = max(z_max, float(data.xpos[:, 2].max()))

    limited = np.asarray(model.jnt_limited, dtype=bool)
    ranges = np.asarray(model.jnt_range)
    hinge_joint_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    limit_violations = []
    for q_col, j_id in enumerate(hinge_joint_ids):
        if not limited[j_id]:
            continue
        lo, hi = ranges[j_id]
        vals = qpos[:, q_col]
        below = np.where(vals < lo - 1e-6)[0]
        above = np.where(vals > hi + 1e-6)[0]
        if below.size or above.size:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or str(j_id)
            limit_violations.append((name, below.size, above.size, float(vals.min()), float(vals.max()), lo, hi))

    print(f"motion: {args.motion}")
    print(f"source qpos_23dof shape: {qpos_src.shape}")
    print(f"model nq: {model.nq}")
    print(f"checked qpos shape: {qpos.shape}")
    print(f"joints: {', '.join(joint_names(model))}")
    print(f"geometry z range across checked frames: [{z_min:.4f}, {z_max:.4f}]")

    if bad_frames:
        print(f"non-finite frames: {bad_frames[:20]}", file=sys.stderr)
    if limit_violations:
        print("joint limit violations:", file=sys.stderr)
        for name, below, above, vmin, vmax, lo, hi in limit_violations:
            print(
                f"  {name}: below={below} above={above} value=[{vmin:.4f}, {vmax:.4f}] limit=[{lo:.4f}, {hi:.4f}]",
                file=sys.stderr,
            )

    return 1 if bad_frames or limit_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
