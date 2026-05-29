#!/usr/bin/env python3
"""Generate one Unitree G1/Kimodo G1 motion per prompt entry.

This intentionally calls ``kimodo.scripts.generate`` once per prompt so each
action becomes its own named output file. Kimodo's built-in multi-prompt mode
creates one continuous sequence, which is not what this batch is for.

The output contract keeps both Kimodo NPZ and G1 MuJoCo CSV sidecars.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROMPTS = Path(__file__).with_name("prompts_10_g1.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("generated_g1_10s")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_G1_XML = REPO_ROOT / "kimodo" / "assets" / "skeletons" / "g1skel34" / "xml" / "g1.xml"
DEFAULT_FPS = 30.0


def check_generation_environment() -> None:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "Current Python environment cannot generate Kimodo motions because 'torch' is not installed.\n"
            "Create/activate a Kimodo environment with Python 3.10+ and PyTorch 2.0+, then run this script again.\n"
            "For example:\n"
            "  conda create -n kimodo python=3.10\n"
            "  conda activate kimodo\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
            "  pip install -e .\n"
            "Or run generation inside the Docker environment, whose PyTorch base image is intended for Kimodo."
        )


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "motion"


def load_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")

    prompts: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry {idx} must be an object")
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Prompt entry {idx} is missing 'prompt'")
        action = slugify(str(item.get("action") or f"motion_{idx:02d}"))
        duration = float(item.get("duration", 4.0))
        prompts.append({"action": action, "prompt": prompt, "duration": duration})
    return prompts


def load_existing_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez(tmp_path, **data)
    os.replace(tmp_path, path)


def enrich_generated_g1_npz(path: Path, *, fps: float, g1_xml: Path) -> None:
    import mujoco

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from research.retarget_g1_to_elf3.retarget_g1_to_elf3_baseline import (
        ROOT_QPOS_COLUMNS,
        mjlab_compatible_fields,
        model_body_names,
        model_hinge_joint_names,
    )
    from research.retarget_g1_to_elf3.visualize_mujoco_g1 import load_qpos36

    model = mujoco.MjModel.from_xml_path(str(g1_xml))
    qpos = load_qpos36(str(path), str(g1_xml)).astype(np.float32)
    joint_names = model_hinge_joint_names(model)
    body_names = model_body_names(model)
    qpos_columns = (*ROOT_QPOS_COLUMNS, *joint_names)

    data = load_existing_npz(path)
    data.update(
        {
            "qpos_g1": qpos,
            "qpos_g1_columns": np.array(qpos_columns),
            "qpos_columns": np.array(qpos_columns),
            **mjlab_compatible_fields(
                model,
                qpos,
                fps=fps,
                joint_names=joint_names,
                body_names=body_names,
            ),
        }
    )
    save_npz_atomic(path, data)
    print(f"mjlab npz fields: joint_pos={data['joint_pos'].shape}, body_pos_w={data['body_pos_w'].shape}")


def run_generation(
    entry: dict[str, Any],
    *,
    output_dir: Path,
    model: str,
    diffusion_steps: int,
    seed: int | None,
    fps: float,
    g1_xml: Path,
    dry_run: bool,
) -> Path:
    output_stem = output_dir / entry["action"]
    cmd = [
        sys.executable,
        "-m",
        "kimodo.scripts.generate",
        entry["prompt"],
        "--model",
        model,
        "--duration",
        str(entry["duration"]),
        "--num_samples",
        "1",
        "--diffusion_steps",
        str(diffusion_steps),
        "--output",
        str(output_stem),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print("\n" + "=" * 80)
    print(f"action: {entry['action']}")
    print(f"prompt: {entry['prompt']}")
    print("cmd:", " ".join(cmd))
    if dry_run:
        return output_stem.with_suffix(".npz")

    subprocess.run(cmd, check=True)
    enrich_generated_g1_npz(output_stem.with_suffix(".npz"), fps=fps, g1_xml=g1_xml)

    return output_stem.with_suffix(".npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="kimodo-g1-rp-v1")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="FPS metadata used for MJLab-style velocity fields.")
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running generation.")
    args = parser.parse_args()

    if "g1" not in args.model.lower():
        raise ValueError(f"This batch is only for Unitree G1/Kimodo G1 models; got {args.model!r}")
    if not args.dry_run:
        check_generation_environment()

    prompts = load_prompts(args.prompts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for entry in prompts:
        outputs.append(
            run_generation(
                entry,
                output_dir=args.output_dir,
                model=args.model,
                diffusion_steps=args.diffusion_steps,
                seed=args.seed,
                fps=args.fps,
                g1_xml=args.g1_xml,
                dry_run=args.dry_run,
            )
        )

    print("\nGenerated NPZ files:")
    for path in outputs:
        print(f"  {path}")
    print("\nKimodo G1 CSV sidecars are kept next to each NPZ file.")
    if args.dry_run:
        print("Dry run only; no files were generated or enriched.")
    else:
        print("Each NPZ is enriched in-place with MJLab-style fields and real G1 joint/body names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
