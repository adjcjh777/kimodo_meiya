"""Evaluate trained policy on reference motions."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import yaml
from stable_baselines3 import PPO
from tqdm import tqdm

try:
    from ..env.elf3_env import ELF3TrackingEnv
    from ..env.reference_motion import (
        POLICY_IDS,
        ReferenceMotionManager,
        load_policy_route,
        load_yaml,
        policy_motion_stems,
    )
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from research.retarget_g1_to_elf3.rl.env.elf3_env import ELF3TrackingEnv
    from research.retarget_g1_to_elf3.rl.env.reference_motion import (
        POLICY_IDS,
        ReferenceMotionManager,
        load_policy_route,
        load_yaml,
        policy_motion_stems,
    )


def load_config(config_path: str) -> dict:
    """Load configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    return Path.cwd()


def _resolve_config_path(path: str | Path, config_path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    config_path = Path(config_path).resolve()
    repo_root = _find_repo_root(config_path)
    candidates = [
        config_path.parent / raw,
        repo_root / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / raw).resolve()


def _policy_file(config: dict, config_path: str | Path, key: str, filename: str) -> Path:
    return _resolve_config_path(
        config.get(key, Path("research/retarget_g1_to_elf3/rl/config") / filename),
        config_path,
    )


def _load_policy_context(config: dict, config_path: str, policy_id: str | None) -> dict:
    if policy_id is None:
        return {
            "policy_id": None,
            "policy_config": None,
            "all_motion_stems": None,
            "reward_weights": None,
        }
    if policy_id not in POLICY_IDS:
        raise ValueError(f"Unknown policy_id {policy_id!r}; expected one of {POLICY_IDS}")

    route_path = _policy_file(config, config_path, "policy_route_path", "policy_route.yaml")
    policy_configs_path = _policy_file(config, config_path, "policy_configs_path", "policy_configs.yaml")
    motion_dir = _resolve_config_path(config["motion_dir"], config_path)

    route = load_policy_route(route_path)
    ReferenceMotionManager.validate_policy_route(motion_dir, route)
    policy_configs = load_yaml(policy_configs_path).get("policies", {})
    if policy_id not in policy_configs:
        raise ValueError(f"policy_configs.yaml is missing {policy_id}")

    policy_config = deepcopy(policy_configs[policy_id])
    return {
        "policy_id": policy_id,
        "policy_config": policy_config,
        "all_motion_stems": policy_motion_stems(route, policy_id),
        "reward_weights": policy_config.get("reward_weights"),
    }


def _flatten_clip_sets(clips_by_policy: dict) -> list[str]:
    stems: list[str] = []
    for policy_id in POLICY_IDS:
        stems.extend(clips_by_policy.get(policy_id, []))
    return stems


def _select_eval_stems(
    config: dict,
    config_path: str,
    policy_id: str | None,
    eval_set: str,
    clip_name: str | None = None,
    legacy_root: bool = False,
) -> list[str] | None:
    eval_sets_path = _policy_file(config, config_path, "eval_sets_path", "eval_sets.yaml")
    eval_sets = load_yaml(eval_sets_path)
    if eval_set not in eval_sets:
        raise ValueError(f"Unknown eval set {eval_set!r}; available: {sorted(eval_sets)}")
    if legacy_root and eval_set != "debug_eval":
        raise ValueError("Root-level legacy clips are only allowed with --eval-set debug_eval")

    eval_cfg = eval_sets[eval_set]
    if legacy_root:
        legacy_clips = eval_cfg.get("legacy_root_clips", [])
        if clip_name is not None:
            if clip_name not in legacy_clips:
                raise ValueError(f"{clip_name!r} is not listed in debug_eval.legacy_root_clips")
            return [clip_name]
        return list(legacy_clips)

    if clip_name is not None:
        return [clip_name]

    clips_by_policy = eval_cfg.get("clips", {})
    if policy_id is not None:
        stems = list(clips_by_policy.get(policy_id, []))
        if not stems:
            raise ValueError(f"Eval set {eval_set!r} has no clips for {policy_id}")
        return stems

    if clips_by_policy:
        return _flatten_clip_sets(clips_by_policy)
    return None


def evaluate_single_episode(
    env: ELF3TrackingEnv,
    model: PPO,
    clip,
    max_steps: int = 5000,
    record_trajectory: bool = False,
) -> dict:
    """Run a single evaluation episode."""
    obs, info = env.reset(options={"clip": clip})
    done = False
    step = 0
    terminated = False
    truncated = False
    termination_reason = None
    simulation_unstable = False

    total_reward = 0.0
    tracking_errors = []
    joint_velocities = []
    action_deltas = []
    root_heights = []
    last_action = None
    trajectory = []

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1
        simulation_unstable = simulation_unstable or bool(info.get("simulation_unstable", False))
        if info.get("termination_reason"):
            termination_reason = info["termination_reason"]

        if "tracking_error" in info:
            tracking_errors.append(info["tracking_error"])

        joint_velocities.append(np.linalg.norm(env.data.qvel[env.joint_qvel_slice]))
        root_heights.append(env.data.qpos[2])

        if last_action is not None:
            action_deltas.append(np.linalg.norm(action - last_action))
        last_action = action.copy()

        if record_trajectory:
            trajectory.append({
                "qpos": env.data.qpos.copy(),
                "qvel": env.data.qvel.copy(),
                "action": action.copy(),
                "reward": reward,
            })

        done = terminated or truncated

    return {
        "clip_name": clip.name,
        "category": clip.category,
        "is_legacy_root": clip.is_legacy_root,
        "n_steps": step,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": termination_reason,
        "simulation_unstable": simulation_unstable,
        "total_reward": total_reward,
        "mean_reward": total_reward / max(step, 1),
        "mean_tracking_error": np.mean(tracking_errors) if tracking_errors else 0,
        "max_tracking_error": np.max(tracking_errors) if tracking_errors else 0,
        "mean_joint_velocity": np.mean(joint_velocities) if joint_velocities else 0,
        "mean_action_delta": np.mean(action_deltas) if action_deltas else 0,
        "mean_root_height": np.mean(root_heights) if root_heights else 0,
        "min_root_height": np.min(root_heights) if root_heights else 0,
        "trajectory": trajectory if record_trajectory else None,
    }


def evaluate_policy(
    model_path: str,
    config_path: str = None,
    policy_id: str = None,
    eval_set: str = "benchmark_eval",
    clip_name: str = None,
    legacy_root: bool = False,
    n_episodes_per_clip: int = 3,
    max_steps: int = 5000,
    record_trajectory: bool = False,
    verbose: int = 1,
) -> dict:
    """Evaluate a trained policy on the selected eval clips."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "default.yaml")

    config = load_config(config_path)
    policy_context = _load_policy_context(config, config_path, policy_id)
    clip_stems = _select_eval_stems(config, config_path, policy_id, eval_set, clip_name, legacy_root)

    env = ELF3TrackingEnv(
        config_path=config_path,
        policy_id=policy_id,
        motion_stems=clip_stems,
        load_mode=eval_set,
        include_classified_clips=not legacy_root,
        include_root_clips=legacy_root,
        reward_weights=policy_context.get("reward_weights"),
    )
    model = PPO.load(model_path)
    motion_manager = env.motion_manager

    if policy_id is not None and eval_set == "benchmark_eval" and len(motion_manager.clips) != len(clip_stems):
        raise ValueError(
            f"benchmark_eval expected {len(clip_stems)} clips for {policy_id}, "
            f"loaded {len(motion_manager.clips)}"
        )

    if verbose > 0:
        print(f"{'='*60}")
        print(f"Evaluating Policy: {model_path}")
        print(f"{'='*60}")
        print(f"Policy id: {policy_id or 'all'}")
        print(f"Eval set: {eval_set}")
        print(f"Legacy root clips: {legacy_root}")
        print(f"Total clips: {len(motion_manager.clips)}")
        print(f"Episodes per clip: {n_episodes_per_clip}")
        print(f"Max steps: {max_steps}")
        print(f"{'='*60}\n")

    all_results = []
    for clip in tqdm(motion_manager.clips, desc="Clips", disable=(verbose < 1)):
        for episode in range(n_episodes_per_clip):
            result = evaluate_single_episode(
                env, model, clip, max_steps, record_trajectory
            )
            result["episode"] = episode
            all_results.append(result)

    by_category = {}
    for category in ReferenceMotionManager.ALL_CATEGORIES:
        category_results = [r for r in all_results if r["category"] == category]
        if category_results:
            by_category[category] = {
                "n_clips": len(set(r["clip_name"] for r in category_results)),
                "mean_reward": np.mean([r["mean_reward"] for r in category_results]),
                "mean_tracking_error": np.mean([r["mean_tracking_error"] for r in category_results]),
                "mean_survival": np.mean([r["n_steps"] for r in category_results]),
                "survival_rate": np.mean([1 if r["truncated"] else 0 for r in category_results]),
                "mean_action_smoothness": np.mean([r["mean_action_delta"] for r in category_results]),
                "simulation_unstable_rate": np.mean([r["simulation_unstable"] for r in category_results]),
            }

    termination_counts = {}
    for result in all_results:
        reason = result["termination_reason"]
        if reason is None:
            reason = "timeout" if result["truncated"] else "max_steps"
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    overall = {
        "total_clips": len(motion_manager.clips),
        "total_episodes": len(all_results),
        "mean_reward": np.mean([r["mean_reward"] for r in all_results]),
        "mean_tracking_error": np.mean([r["mean_tracking_error"] for r in all_results]),
        "mean_survival": np.mean([r["n_steps"] for r in all_results]),
        "survival_rate": np.mean([1 if r["truncated"] else 0 for r in all_results]),
        "mean_action_smoothness": np.mean([r["mean_action_delta"] for r in all_results]),
        "mean_joint_velocity": np.mean([r["mean_joint_velocity"] for r in all_results]),
        "simulation_unstable_rate": np.mean([r["simulation_unstable"] for r in all_results]),
        "termination_counts": termination_counts,
    }

    if verbose > 0:
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print("\nOverall:")
        print(f"  Total episodes: {overall['total_episodes']}")
        print(f"  Mean reward: {overall['mean_reward']:.4f}")
        print(f"  Mean tracking error: {overall['mean_tracking_error']:.4f}")
        print(f"  Mean survival steps: {overall['mean_survival']:.1f} / {max_steps}")
        print(f"  Survival rate: {overall['survival_rate']:.1%}")
        print(f"  Simulation unstable rate: {overall['simulation_unstable_rate']:.1%}")
        print(f"  Terminations: {overall['termination_counts']}")
        print(f"  Action smoothness (mean delta): {overall['mean_action_smoothness']:.4f}")
        print(f"  Mean joint velocity: {overall['mean_joint_velocity']:.2f}")

        print("\nBy Category:")
        for category, metrics in by_category.items():
            print(f"  {category} ({metrics['n_clips']} clips):")
            print(f"    Reward: {metrics['mean_reward']:.4f}")
            print(f"    Tracking error: {metrics['mean_tracking_error']:.4f}")
            print(f"    Survival: {metrics['mean_survival']:.0f} / {max_steps}")
            print(f"    Survival rate: {metrics['survival_rate']:.1%}")
            print(f"    Simulation unstable: {metrics['simulation_unstable_rate']:.1%}")
            print(f"    Smoothness: {metrics['mean_action_smoothness']:.4f}")

        print(f"{'='*60}\n")

    env.close()

    return {
        "overall": overall,
        "by_category": by_category,
        "per_episode": all_results,
    }


def render_evaluation_video(
    model_path: str,
    output_path: str,
    config_path: str = None,
    policy_id: str = None,
    eval_set: str = "debug_eval",
    clip_name: str = None,
    legacy_root: bool = False,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    max_steps: int = 5000,
) -> None:
    """Render evaluation video for a specific clip."""
    import cv2

    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "default.yaml")

    config = load_config(config_path)
    policy_context = _load_policy_context(config, config_path, policy_id)
    clip_stems = _select_eval_stems(config, config_path, policy_id, eval_set, clip_name, legacy_root)

    env = ELF3TrackingEnv(
        config_path=config_path,
        render_mode="rgb_array",
        policy_id=policy_id,
        motion_stems=clip_stems,
        load_mode=eval_set,
        include_classified_clips=not legacy_root,
        include_root_clips=legacy_root,
        reward_weights=policy_context.get("reward_weights"),
    )
    model = PPO.load(model_path)

    if clip_name:
        clip = env.motion_manager.get_clip(clip_name, prefer_legacy_root=legacy_root)
    else:
        clip = env.motion_manager.sample_clip()

    print(f"Rendering video for clip: {clip.name} ({clip.category})")

    obs, info = env.reset(options={"clip": clip})
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    frames = []
    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        renderer.update_scene(env.data)
        frames.append(renderer.render())

        step += 1
        done = terminated or truncated

        if step % 100 == 0:
            print(f"  Frame {step}...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    renderer.close()
    env.close()

    print(f"Video saved to {output_path} ({len(frames)} frames, {len(frames) / fps:.1f}s)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained ELF3 RL policy")
    parser.add_argument("model_path", type=str, help="Path to trained model (.zip)")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--policy-id", type=str, default=None, choices=POLICY_IDS,
                        help="Skill policy id to evaluate")
    parser.add_argument("--eval-set", type=str, default="benchmark_eval",
                        choices=["benchmark_eval", "debug_eval"])
    parser.add_argument("--legacy-root", action="store_true",
                        help="For debug_eval only, load root-level legacy baseline clips")
    parser.add_argument("--episodes-per-clip", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--render", type=str, default=None,
                        help="Render video to this path (MP4)")
    parser.add_argument("--clip", type=str, default=None,
                        help="Specific clip stem")

    args = parser.parse_args()

    if args.render:
        render_evaluation_video(
            model_path=args.model_path,
            output_path=args.render,
            config_path=args.config,
            policy_id=args.policy_id,
            eval_set=args.eval_set,
            clip_name=args.clip,
            legacy_root=args.legacy_root,
            max_steps=args.max_steps,
        )
    else:
        evaluate_policy(
            model_path=args.model_path,
            config_path=args.config,
            policy_id=args.policy_id,
            eval_set=args.eval_set,
            clip_name=args.clip,
            legacy_root=args.legacy_root,
            n_episodes_per_clip=args.episodes_per_clip,
            max_steps=args.max_steps,
        )
