#!/usr/bin/env python3
"""Visualize ELF3 retargeted qpos NPZ in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


DEFAULT_ELF3_XML = Path(__file__).resolve().parent / "assets" / "bxi_elf3" / "xmls" / "elf3.xml"


def load_qpos(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "qpos_elf3" not in data:
        raise ValueError(f"{path} does not contain qpos_elf3; keys={list(data.keys())}")
    qpos = data["qpos_elf3"]
    if qpos.ndim == 3:
        qpos = qpos[0]
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos_elf3 shape [T, 36], got {qpos.shape}")
    return qpos


def visualize(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> None:
    data = mujoco.MjData(model)
    n_frames = qpos.shape[0]

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
        while viewer.is_running():
            start_time = time.time()
            data.qpos[:] = qpos[frame_idx]
            mujoco.mj_forward(model, data)
            frame_idx = (frame_idx + 1) % n_frames
            viewer.sync()
            time.sleep(max(0.0, (1.0 / fps) - (time.time() - start_time)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--xml", type=Path, default=DEFAULT_ELF3_XML)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    qpos = load_qpos(args.motion)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    if model.nq != qpos.shape[1]:
        raise ValueError(f"Model expects nq={model.nq}, but qpos has {qpos.shape[1]} columns")
    visualize(model, qpos, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
