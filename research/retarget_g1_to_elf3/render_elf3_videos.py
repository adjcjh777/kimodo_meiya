#!/usr/bin/env python3
"""Render ELF3 qpos NPZ files to MP4 videos with MuJoCo offscreen rendering."""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Must be set before importing mujoco in headless sessions.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_ELF3_XML = Path(__file__).resolve().parent / "assets" / "bxi_elf3" / "xmls" / "elf3.xml"
DEFAULT_INPUT = Path(__file__).resolve().parent / "generated_elf3_10s"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "videos_elf3_10s"


def has_named_element(parent: ET.Element, tag: str, name: str) -> bool:
    return any(child.tag == tag and child.get("name") == name for child in parent)


def ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is not None:
        return child

    insert_at = 0
    for idx, existing in enumerate(list(root)):
        if existing.tag in {"compiler", "option", "size", "default"}:
            insert_at = idx + 1
    child = ET.Element(tag)
    root.insert(insert_at, child)
    return child


def add_video_scene_elements(root: ET.Element, *, ground_size: float) -> None:
    asset = ensure_child(root, "asset")
    if not has_named_element(asset, "texture", "video_skybox"):
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "video_skybox",
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.3 0.5 0.7",
                "rgb2": "0 0 0",
                "width": "32",
                "height": "512",
            },
        )
    if not has_named_element(asset, "texture", "video_ground_checker"):
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "video_ground_checker",
                "type": "2d",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "0.2 0.3 0.4",
                "rgb2": "0.1 0.2 0.3",
                "markrgb": "0.8 0.8 0.8",
                "width": "300",
                "height": "300",
            },
        )
    if not has_named_element(asset, "material", "video_ground"):
        ET.SubElement(
            asset,
            "material",
            {
                "name": "video_ground",
                "texture": "video_ground_checker",
                "texuniform": "true",
                "texrepeat": "5 5",
                "reflectance": "0.2",
            },
        )

    visual = ensure_child(root, "visual")
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            {
                "diffuse": "0.75 0.75 0.75",
                "ambient": "0.18 0.18 0.18",
                "specular": "0.10 0.10 0.10",
            },
        )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("ELF3 XML does not contain a worldbody")

    if not has_named_element(worldbody, "light", "video_key_light"):
        worldbody.insert(
            0,
            ET.Element(
                "light",
                {
                    "name": "video_key_light",
                    "pos": "0 -3 4",
                    "dir": "0 0 -1",
                    "directional": "true",
                    "diffuse": "0.65 0.65 0.65",
                    "ambient": "0.12 0.12 0.12",
                },
            ),
        )
    if not has_named_element(worldbody, "geom", "video_floor"):
        worldbody.insert(
            0,
            ET.Element(
                "geom",
                {
                    "name": "video_floor",
                    "type": "plane",
                    "pos": "0 0 0",
                    "size": f"{ground_size:g} {ground_size:g} 0.01",
                    "material": "video_ground",
                    "contype": "1",
                    "conaffinity": "1",
                },
            ),
        )


def load_model_for_render(xml_path: Path, *, add_ground: bool, ground_size: float) -> mujoco.MjModel:
    if not add_ground:
        return mujoco.MjModel.from_xml_path(str(xml_path))

    tree = ET.parse(xml_path)
    root = tree.getroot()
    add_video_scene_elements(root, ground_size=ground_size)

    # Keep the temporary XML next to the source XML so relative meshdir paths
    # such as "./meshes" continue to resolve.
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".xml",
        prefix=".render_scene_",
        dir=xml_path.parent,
        delete=False,
    ) as tmp:
        tree.write(tmp, encoding="utf-8", xml_declaration=False)
        tmp_path = Path(tmp.name)

    try:
        return mujoco.MjModel.from_xml_path(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


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
    max_frames: int | None,
) -> None:
    qpos = load_qpos(input_path)
    if max_frames is not None:
        qpos = qpos[:max_frames]
    if len(qpos) == 0:
        raise ValueError(f"No frames to render for {input_path}")

    print(f"rendering {input_path} -> {output_path} ({len(qpos)} frames)", flush=True)
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

    print(f"saved {output_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="ELF3 NPZ file or directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xml", type=Path, default=DEFAULT_ELF3_XML)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-ground", action="store_true", help="Render the raw MJCF scene without the added floor/skybox.")
    parser.add_argument("--ground-size", type=float, default=8.0, help="Half-size of the added checkerboard floor plane.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug option: render only the first N frames.")
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

    if args.ground_size <= 0:
        raise ValueError("--ground-size must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    model = load_model_for_render(args.xml, add_ground=not args.no_ground, ground_size=args.ground_size)
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
            max_frames=args.max_frames,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
