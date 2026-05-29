#!/usr/bin/env python3
"""Add MJLab-style trajectory fields and real MJCF names to G1/ELF3 NPZ files.

The reference ELF3 file used by downstream code stores joint/body trajectories as:

  fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

This script preserves existing Kimodo/qpos fields and adds the compatible fields
in-place by default. It is intentionally metadata/format alignment; it does not
change the underlying retargeted qpos values.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.retarget_g1_to_elf3.retarget_g1_to_elf3_baseline import (  # noqa: E402
    DEFAULT_ELF3_XML,
    DEFAULT_FPS,
    DEFAULT_G1_XML,
    ELF3_BODY_NAMES,
    ELF3_JOINT_NAMES,
    ELF3_QPOS_COLUMNS,
    ROOT_QPOS_COLUMNS,
    mjlab_compatible_fields,
    model_body_names,
    model_hinge_joint_names,
)
from research.retarget_g1_to_elf3.visualize_mujoco_g1 import load_qpos36  # noqa: E402


def load_existing(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def qpos_from_existing(path: Path, data: dict[str, Any], robot: str, g1_xml: Path) -> np.ndarray:
    if robot == "elf3":
        if "qpos_elf3" not in data:
            raise ValueError(f"{path} is missing qpos_elf3")
        qpos = data["qpos_elf3"]
    else:
        qpos = data["qpos_g1"] if "qpos_g1" in data else load_qpos36(str(path), str(g1_xml))

    if qpos.ndim == 3:
        qpos = qpos[0]
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{path}: expected qpos shape [T, 36], got {qpos.shape}")
    return qpos.astype(np.float32)


def infer_robot(path: Path, data: dict[str, Any]) -> str:
    if "qpos_elf3" in data:
        return "elf3"
    if "qpos_g1" in data or "global_rot_mats" in data or "local_rot_mats" in data:
        return "g1"
    raise ValueError(f"Cannot infer robot type for {path}; use --robot explicitly.")


def output_path_for(input_path: Path, output: Path | None, input_root: Path) -> Path:
    if output is None:
        return input_path
    if input_root.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        return output / input_path.name
    if output.suffix == ".npz":
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output / input_path.name


def save_npz(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez(tmp_path, **data)
    os.replace(tmp_path, path)


def align_file(
    input_path: Path,
    output_path: Path,
    *,
    robot: str,
    elf3_model: mujoco.MjModel,
    g1_model: mujoco.MjModel,
    g1_xml: Path,
    fps: float,
) -> None:
    data = load_existing(input_path)
    resolved_robot = infer_robot(input_path, data) if robot == "auto" else robot
    model = elf3_model if resolved_robot == "elf3" else g1_model
    qpos = qpos_from_existing(input_path, data, resolved_robot, g1_xml)

    if resolved_robot == "elf3":
        joint_names = ELF3_JOINT_NAMES
        body_names = ELF3_BODY_NAMES
        qpos_columns = ELF3_QPOS_COLUMNS
        if "qpos_g1" in data:
            g1_qpos_columns = (*ROOT_QPOS_COLUMNS, *model_hinge_joint_names(g1_model))
            data.setdefault("qpos_g1_columns", np.array(g1_qpos_columns))
    else:
        joint_names = model_hinge_joint_names(model)
        body_names = model_body_names(model)
        qpos_columns = (*ROOT_QPOS_COLUMNS, *joint_names)
        data.setdefault("qpos_g1", qpos)
        data.setdefault("qpos_g1_columns", np.array(qpos_columns))

    data.update(
        mjlab_compatible_fields(
            model,
            qpos,
            fps=fps,
            joint_names=joint_names,
            body_names=body_names,
        )
    )
    data["joint_names"] = np.array(joint_names)
    data["body_names"] = np.array(body_names)
    data["qpos_columns"] = np.array(qpos_columns)
    if resolved_robot == "elf3":
        data["qpos_elf3_columns"] = np.array(qpos_columns)

    save_npz(output_path, data)
    mode = "updated" if input_path == output_path else "saved"
    print(f"{mode} {output_path}: robot={resolved_robot} joint_pos={data['joint_pos'].shape}")


def iter_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.npz"))
    return [path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ file or directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file/dir; default updates in place")
    parser.add_argument("--robot", choices=("auto", "elf3", "g1"), default="auto")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--elf3-xml", type=Path, default=DEFAULT_ELF3_XML)
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    args = parser.parse_args()

    inputs = iter_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No NPZ files found: {args.input}")

    elf3_model = mujoco.MjModel.from_xml_path(str(args.elf3_xml))
    g1_model = mujoco.MjModel.from_xml_path(str(args.g1_xml))
    for input_path in inputs:
        align_file(
            input_path,
            output_path_for(input_path, args.output, args.input),
            robot=args.robot,
            elf3_model=elf3_model,
            g1_model=g1_model,
            g1_xml=args.g1_xml,
            fps=args.fps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
