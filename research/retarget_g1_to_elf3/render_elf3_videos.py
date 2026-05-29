#!/usr/bin/env python3
"""Render ELF3 qpos NPZ files to MP4 videos with MuJoCo offscreen rendering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before importing mujoco in headless sessions.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_ELF3_XML = Path(__file__).resolve().parent / "assets" / "bxi_elf3" / "xmls" / "elf3.xml"
DEFAULT_INPUT = Path(__file__).resolve().parent / "generated_elf3_10s"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "videos_elf3_10s"


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


def iter_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.npz"))
    return [path]


def setup_camera(model: mujoco.MjModel, qpos: np.ndarray, distance_scale: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    center = np.array(
        [
            float((qpos[:, 0].min() + qpos[:, 0].max()) * 0.5),
            float((qpos[:, 1].min() + qpos[:, 1].max()) * 0.5),
            float((qpos[:, 2].min() + qpos[:, 2].max()) * 0.5),
        ]
    )
    span_xy = float(max(np.ptp(qpos[:, 0]), np.ptp(qpos[:, 1]), 1.0))
    camera.lookat[:] = center
    camera.lookat[2] = max(camera.lookat[2], 0.75)
    camera.distance = (2.4 + 0.45 * span_xy) * distance_scale
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera


def render_video(
    model: mujoco.MjModel,
    input_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    camera_name: str | None,
    camera_distance_scale: float,
) -> None:
    qpos = load_qpos(input_path)
    data = mujoco.MjData(model)
    free_camera = None if camera_name else setup_camera(model, qpos, camera_distance_scale)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, width=width, height=height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for frame_idx, row in enumerate(qpos):
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            if camera_name:
                renderer.update_scene(data, camera=camera_name)
            else:
                renderer.update_scene(data, camera=free_camera)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"{input_path.stem}  {frame_idx + 1}/{len(qpos)}",
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (30, 30, 30),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                f"{input_path.stem}  {frame_idx + 1}/{len(qpos)}",
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            writer.write(bgr)
    finally:
        writer.release()
        renderer.close()

    print(f"saved {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="ELF3 NPZ file or directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xml", type=Path, default=DEFAULT_ELF3_XML)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--camera",
        default="auto",
        help="Named MJCF camera to use. Use --camera auto for an auto-framed free camera.",
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=1.35,
        help="Scale auto camera distance; larger values place the camera farther away.",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    inputs = iter_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No NPZ files found: {args.input}")

    for input_path in inputs:
        render_video(
            model,
            input_path,
            args.output_dir / f"{input_path.stem}.mp4",
            width=args.width,
            height=args.height,
            fps=args.fps,
            camera_name=None if args.camera == "auto" else args.camera,
            camera_distance_scale=args.camera_distance_scale,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
