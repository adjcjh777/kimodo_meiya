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


DEFAULT_PROMPTS = Path(__file__).with_name("prompts_10_g1.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("generated_g1_10s")


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


def run_generation(
    entry: dict[str, Any],
    *,
    output_dir: Path,
    model: str,
    diffusion_steps: int,
    seed: int | None,
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

    return output_stem.with_suffix(".npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="kimodo-g1-rp-v1")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
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
                dry_run=args.dry_run,
            )
        )

    print("\nGenerated NPZ files:")
    for path in outputs:
        print(f"  {path}")
    print("\nKimodo G1 CSV sidecars are kept next to each NPZ file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
