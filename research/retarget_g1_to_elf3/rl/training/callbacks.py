"""Custom callbacks for SB3 training."""

import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


class MotionSwitchCallback(BaseCallback):
    """Ensures each motion clip is sampled uniformly during training.

    This callback tracks which clips have been sampled and periodically
    forces sampling of underrepresented clips to ensure balanced training.
    """

    def __init__(
        self,
        env: VecEnv,
        switch_freq: int = 10000,
        verbose: int = 0
    ):
        """Initialize motion switch callback.

        Args:
            env: Vectorized training environment
            switch_freq: How often to check clip distribution (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env = env
        self.switch_freq = switch_freq
        self.clip_counts: Dict[str, int] = {}
        self.last_switch_step = 0

    def _on_training_start(self) -> None:
        """Initialize clip counts."""
        # Get motion manager from first environment
        env_unwrapped = self.env.get_attr('motion_manager')[0]
        for clip in env_unwrapped.clips:
            self.clip_counts[clip.name] = 0

        if self.verbose > 0:
            print(f"MotionSwitchCallback: Tracking {len(self.clip_counts)} clips")

    def _on_step(self) -> bool:
        """Check clip distribution and log statistics."""
        # Update clip counts from environments
        if self.n_calls % 100 == 0:  # Update every 100 steps
            infos = self.locals.get('infos', [])
            for info in infos:
                if 'clip_name' in info:
                    clip_name = info['clip_name']
                    if clip_name in self.clip_counts:
                        self.clip_counts[clip_name] += 1

        # Log distribution periodically
        if self.num_timesteps - self.last_switch_step >= self.switch_freq:
            self.last_switch_step = self.num_timesteps

            if self.verbose > 0 and self.clip_counts:
                # Compute statistics
                counts = list(self.clip_counts.values())
                if counts:
                    mean_count = np.mean(counts)
                    std_count = np.std(counts)
                    min_clip = min(self.clip_counts.items(), key=lambda x: x[1])
                    max_clip = max(self.clip_counts.items(), key=lambda x: x[1])

                    self.logger.record('train/clip_count_mean', mean_count)
                    self.logger.record('train/clip_count_std', std_count)
                    self.logger.record('train/clip_count_min', min_clip[1])
                    self.logger.record('train/clip_count_max', max_clip[1])

                    if self.verbose > 1:
                        print(f"\nClip distribution at step {self.num_timesteps}:")
                        print(f"  Mean: {mean_count:.1f}, Std: {std_count:.1f}")
                        print(f"  Min: {min_clip[0]} ({min_clip[1]})")
                        print(f"  Max: {max_clip[0]} ({max_clip[1]})")

        return True


class DemoCallback(BaseCallback):
    """Periodically save demo videos during training.

    This callback runs the policy on reference motions and saves
    demonstration outputs to visualize training progress.
    """

    def __init__(
        self,
        env: VecEnv,
        demo_dir: str = "demos/",
        save_freq: int = 1000,
        n_demo_steps: int = 500,
        n_clips: int = 3,
        verbose: int = 0
    ):
        """Initialize demo callback.

        Args:
            env: Vectorized training environment
            demo_dir: Directory to save demo videos
            save_freq: How often to save demos (in episodes)
            n_demo_steps: Number of steps to run for each demo
            n_clips: Number of clips to demo
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env = env
        self.demo_dir = Path(demo_dir)
        self.save_freq = save_freq
        self.n_demo_steps = n_demo_steps
        self.n_clips = n_clips
        self.episode_count = 0

        self.demo_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Check if episode ended and save demo if needed."""
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])

        # Count completed episodes
        for done in dones:
            if done:
                self.episode_count += 1

                # Save demo periodically
                if self.episode_count % self.save_freq == 0:
                    self._save_demo()

        return True

    def _save_demo(self):
        """Save demonstration trajectories."""
        if self.verbose > 0:
            print(f"\nSaving demo at episode {self.episode_count}, step {self.num_timesteps}")

        # Get motion manager and sample clips
        env_unwrapped = self.env.get_attr('motion_manager')[0]
        motion_manager = env_unwrapped

        # Sample n_clips random clips
        sampled_clips = []
        for _ in range(self.n_clips):
            clip = motion_manager.sample_clip()
            sampled_clips.append(clip)

        # Run policy on each clip
        demo_data = []
        for clip in sampled_clips:
            # Create single environment with the same config as the training env.
            try:
                from ..env.elf3_env import ELF3TrackingEnv
            except ImportError:
                from research.retarget_g1_to_elf3.rl.env.elf3_env import ELF3TrackingEnv

            config_path = self.env.get_attr('config_path')[0]
            curriculum_phase = self.env.get_attr('curriculum_phase')[0]
            env_single = ELF3TrackingEnv(
                config_path=str(config_path),
                curriculum_phase=curriculum_phase,
            )

            obs, _ = env_single.reset(options={'clip': clip})
            trajectory = [obs]

            for _ in range(self.n_demo_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_single.step(action)
                trajectory.append(obs)

                if terminated or truncated:
                    break

            demo_data.append({
                'clip_name': clip.name,
                'category': clip.category,
                'trajectory': np.array(trajectory),
                'n_steps': len(trajectory)
            })

            env_single.close()

        # Save demo data
        demo_file = self.demo_dir / f"demo_ep{self.episode_count}_step{self.num_timesteps}.npz"
        np.savez(demo_file, **{f"clip_{i}": d for i, d in enumerate(demo_data)})

        if self.verbose > 0:
            print(f"  Saved demo to {demo_file}")
            for i, d in enumerate(demo_data):
                print(f"  Clip {i}: {d['clip_name']} ({d['category']}), {d['n_steps']} steps")


class EvalCallback(BaseCallback):
    """Periodic evaluation callback with detailed metrics."""

    def __init__(
        self,
        eval_env: VecEnv,
        train_env: Optional[VecEnv] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: int = 1
    ):
        """Initialize evaluation callback.

        Args:
            eval_env: Environment for evaluation
            eval_freq: How often to evaluate (in timesteps)
            n_eval_episodes: Number of evaluation episodes
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.train_env = train_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        """Evaluate policy periodically."""
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._evaluate()

        return True

    def _evaluate(self):
        """Run evaluation episodes and log metrics."""
        if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.train_env.obs_rms

        episode_rewards = []
        episode_lengths = []
        tracking_errors = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])
            episode_reward = 0
            episode_length = 0
            episode_tracking_errors = []

            while not bool(done[0]):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)

                episode_reward += reward[0]
                episode_length += 1

                if 'tracking_error' in info[0]:
                    episode_tracking_errors.append(info[0]['tracking_error'])

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if episode_tracking_errors:
                tracking_errors.extend(episode_tracking_errors)

        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        mean_tracking_error = np.mean(tracking_errors) if tracking_errors else 0

        # Log to tensorboard
        self.logger.record('eval/mean_reward', mean_reward)
        self.logger.record('eval/mean_length', mean_length)
        self.logger.record('eval/mean_tracking_error', mean_tracking_error)

        if self.verbose > 0:
            print(f"\nEvaluation at step {self.num_timesteps}:")
            print(f"  Mean reward: {mean_reward:.2f}")
            print(f"  Mean length: {mean_length:.1f}")
            print(f"  Mean tracking error: {mean_tracking_error:.4f}")
