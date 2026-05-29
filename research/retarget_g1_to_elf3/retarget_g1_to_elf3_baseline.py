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
ELF3_QPOS_COLUMNS = (
    "root_x",
    "root_y",
    "root_z",
    "root_quat_w",
    "root_quat_x",
    "root_quat_y",
    "root_quat_z",
    "waist_y",
    "waist_x",
    "waist_z",
    "l_hip_y",
    "l_hip_x",
    "l_hip_z",
    "l_knee_y",
    "l_ankle_y",
    "l_ankle_x",
    "r_hip_y",
    "r_hip_x",
    "r_hip_z",
    "r_knee_y",
    "r_ankle_y",
    "r_ankle_x",
    "l_shoulder_y",
    "l_shoulder_x",
    "l_shoulder_z",
    "l_elbow_y",
    "l_wrist_x",
    "l_wrist_y",
    "l_wrist_z",
    "r_shoulder_y",
    "r_shoulder_x",
    "r_shoulder_z",
    "r_elbow_y",
    "r_wrist_x",
    "r_wrist_y",
    "r_wrist_z",
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
    g1_xml: Path,
    root_z_offset: float | None,
    clamp: bool,
    save_csv: bool,
) -> None:
    g1_qpos = load_qpos36(str(input_path), str(g1_xml))
    elf3_qpos, clip_reports = retarget_qpos(
        g1_qpos,
        mapping,
        elf3_model,
        root_z_offset=root_z_offset,
        clamp=clamp,
    )

    src = np.load(input_path, allow_pickle=False)
    save_dict: dict[str, Any] = {
        "qpos_elf3": elf3_qpos,
        "qpos_g1": g1_qpos,
        "source_file": str(input_path),
        "joint_map": json.dumps(mapping, ensure_ascii=True),
        "qpos_elf3_columns": np.array(ELF3_QPOS_COLUMNS),
        "root_z_offset": float(root_z_offset if root_z_offset is not None else mapping["root"]["recommended_initial_z_offset"]),
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
    parser.add_argument("--no-clamp", action="store_true", help="Do not clamp to ELF3 joint limits.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write ELF3 qpos CSV sidecars.")
    args = parser.parse_args()

    mapping = load_map(args.map)
    elf3_model = mujoco.MjModel.from_xml_path(str(args.elf3_xml))
    inputs = iter_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No NPZ files found: {args.input}")

    for input_file in inputs:
        convert_file(
            input_file,
            output_for(input_file, args.input, args.output),
            mapping=mapping,
            elf3_model=elf3_model,
            g1_xml=args.g1_xml,
            root_z_offset=args.root_z_offset,
            clamp=not args.no_clamp,
            save_csv=not args.no_csv,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
