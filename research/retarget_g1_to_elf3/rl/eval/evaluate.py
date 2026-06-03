"""Evaluate trained policy on reference motions."""

import os
import sys
from pathlib import Path
import yaml
import numpy as np
import mujoco
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

try:
    from ..env.elf3_env import ELF3TrackingEnv
    from ..env.reference_motion import ReferenceMotionManager
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from research.retarget_g1_to_elf3.rl.env.elf3_env import ELF3TrackingEnv
    from research.retarget_g1_to_elf3.rl.env.reference_motion import ReferenceMotionManager


def load_config(config_path: str) -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_single_episode(
    env: ELF3TrackingEnv,
    model: PPO,
    clip,
    max_steps: int = 5000,
    record_trajectory: bool = False
) -> dict:
    """Run a single evaluation episode.

    Args:
        env: Environment instance
        model: Trained PPO model
        clip: Reference motion clip to evaluate on
        max_steps: Maximum episode steps
        record_trajectory: Whether to record full trajectory

    Returns:
        Dict with evaluation metrics
    """
    obs, info = env.reset(options={'clip': clip})
    done = False
    step = 0
    terminated = False
    truncated = False

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

        # Collect metrics
        if 'tracking_error' in info:
            tracking_errors.append(info['tracking_error'])

        joint_velocities.append(np.linalg.norm(env.data.qvel[env.joint_qvel_slice]))
        root_heights.append(env.data.qpos[2])

        if last_action is not None:
            action_deltas.append(np.linalg.norm(action - last_action))
        last_action = action.copy()

        if record_trajectory:
            trajectory.append({
                'qpos': env.data.qpos.copy(),
                'qvel': env.data.qvel.copy(),
                'action': action.copy(),
                'reward': reward,
            })

        done = terminated or truncated

    return {
        'clip_name': clip.name,
        'category': clip.category,
        'n_steps': step,
        'terminated': terminated,
        'truncated': truncated,
        'total_reward': total_reward,
        'mean_reward': total_reward / max(step, 1),
        'mean_tracking_error': np.mean(tracking_errors) if tracking_errors else 0,
        'max_tracking_error': np.max(tracking_errors) if tracking_errors else 0,
        'mean_joint_velocity': np.mean(joint_velocities) if joint_velocities else 0,
        'mean_action_delta': np.mean(action_deltas) if action_deltas else 0,
        'mean_root_height': np.mean(root_heights) if root_heights else 0,
        'min_root_height': np.min(root_heights) if root_heights else 0,
        'trajectory': trajectory if record_trajectory else None,
    }


def evaluate_policy(
    model_path: str,
    config_path: str = None,
    n_episodes_per_clip: int = 3,
    max_steps: int = 5000,
    record_trajectory: bool = False,
    verbose: int = 1
) -> dict:
    """Evaluate trained policy across all reference clips.

    Args:
        model_path: Path to trained PPO model (.zip)
        config_path: Path to config file
        n_episodes_per_clip: Number of episodes per reference clip
        max_steps: Maximum steps per episode
        record_trajectory: Whether to record full trajectories
        verbose: Verbosity level

    Returns:
        Dict with aggregated evaluation metrics
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "default.yaml")

    config = load_config(config_path)

    # Create environment
    env = ELF3TrackingEnv(config_path=config_path)

    # Load model
    model = PPO.load(model_path)

    # Get motion manager
    motion_manager = env.motion_manager

    if verbose > 0:
        print(f"{'='*60}")
        print(f"Evaluating Policy: {model_path}")
        print(f"{'='*60}")
        print(f"Total clips: {len(motion_manager.clips)}")
        print(f"Episodes per clip: {n_episodes_per_clip}")
        print(f"Max steps: {max_steps}")
        print(f"{'='*60}\n")

    all_results = []

    for clip in tqdm(motion_manager.clips, desc="Clips", disable=(verbose < 1)):
        for ep in range(n_episodes_per_clip):
            result = evaluate_single_episode(
                env, model, clip, max_steps, record_trajectory
            )
            result['episode'] = ep
            all_results.append(result)

    # Aggregate results
    by_category = {}
    for cat in motion_manager.CATEGORIES + ['uncategorized']:
        cat_results = [r for r in all_results if r['category'] == cat]
        if cat_results:
            by_category[cat] = {
                'n_clips': len(set(r['clip_name'] for r in cat_results)),
                'mean_reward': np.mean([r['mean_reward'] for r in cat_results]),
                'mean_tracking_error': np.mean([r['mean_tracking_error'] for r in cat_results]),
                'mean_survival': np.mean([r['n_steps'] for r in cat_results]),
                'survival_rate': np.mean([1 if r['truncated'] else 0 for r in cat_results]),
                'mean_action_smoothness': np.mean([r['mean_action_delta'] for r in cat_results]),
            }

    # Overall metrics
    overall = {
        'total_clips': len(motion_manager.clips),
        'total_episodes': len(all_results),
        'mean_reward': np.mean([r['mean_reward'] for r in all_results]),
        'mean_tracking_error': np.mean([r['mean_tracking_error'] for r in all_results]),
        'mean_survival': np.mean([r['n_steps'] for r in all_results]),
        'survival_rate': np.mean([1 if r['truncated'] else 0 for r in all_results]),
        'mean_action_smoothness': np.mean([r['mean_action_delta'] for r in all_results]),
        'mean_joint_velocity': np.mean([r['mean_joint_velocity'] for r in all_results]),
    }

    if verbose > 0:
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"\nOverall:")
        print(f"  Total episodes: {overall['total_episodes']}")
        print(f"  Mean reward: {overall['mean_reward']:.4f}")
        print(f"  Mean tracking error: {overall['mean_tracking_error']:.4f}")
        print(f"  Mean survival steps: {overall['mean_survival']:.1f} / {max_steps}")
        print(f"  Survival rate: {overall['survival_rate']:.1%}")
        print(f"  Action smoothness (mean delta): {overall['mean_action_smoothness']:.4f}")
        print(f"  Mean joint velocity: {overall['mean_joint_velocity']:.2f}")

        print(f"\nBy Category:")
        for cat, metrics in by_category.items():
            print(f"  {cat} ({metrics['n_clips']} clips):")
            print(f"    Reward: {metrics['mean_reward']:.4f}")
            print(f"    Tracking error: {metrics['mean_tracking_error']:.4f}")
            print(f"    Survival: {metrics['mean_survival']:.0f} / {max_steps}")
            print(f"    Survival rate: {metrics['survival_rate']:.1%}")
            print(f"    Smoothness: {metrics['mean_action_smoothness']:.4f}")

        print(f"{'='*60}\n")

    env.close()

    return {
        'overall': overall,
        'by_category': by_category,
        'per_episode': all_results,
    }


def render_evaluation_video(
    model_path: str,
    output_path: str,
    config_path: str = None,
    clip_name: str = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    max_steps: int = 5000
):
    """Render evaluation video for a specific clip.

    Args:
        model_path: Path to trained PPO model
        output_path: Output video path
        config_path: Path to config file
        clip_name: Specific clip to render (default: random)
        width: Video width
        height: Video height
        fps: Video FPS
    """
    import cv2

    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "default.yaml")

    config = load_config(config_path)

    # Create environment with rendering
    env = ELF3TrackingEnv(config_path=config_path, render_mode='rgb_array')

    # Load model
    model = PPO.load(model_path)

    # Get clip
    if clip_name:
        clip = next(c for c in env.motion_manager.clips if c.name == clip_name)
    else:
        clip = env.motion_manager.sample_clip()

    print(f"Rendering video for clip: {clip.name} ({clip.category})")

    obs, info = env.reset(options={'clip': clip})

    # Setup renderer
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    frames = []
    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        renderer.update_scene(env.data)
        frame = renderer.render()
        frames.append(frame)

        step += 1
        done = terminated or truncated

        if step % 100 == 0:
            print(f"  Frame {step}...")

    # Write video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    renderer.close()
    env.close()

    print(f"Video saved to {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained ELF3 RL policy")
    parser.add_argument("model_path", type=str, help="Path to trained model (.zip)")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--episodes-per-clip", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--render", type=str, default=None,
                        help="Render video to this path (MP4)")
    parser.add_argument("--clip", type=str, default=None,
                        help="Specific clip to render")

    args = parser.parse_args()

    if args.render:
        render_evaluation_video(
            model_path=args.model_path,
            output_path=args.render,
            config_path=args.config,
            clip_name=args.clip,
            max_steps=args.max_steps
        )
    else:
        results = evaluate_policy(
            model_path=args.model_path,
            config_path=args.config,
            n_episodes_per_clip=args.episodes_per_clip,
            max_steps=args.max_steps
        )
