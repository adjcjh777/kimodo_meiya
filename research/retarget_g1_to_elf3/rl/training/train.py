"""Training script for ELF3 RL framework."""

import os
import sys
from pathlib import Path
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

try:
    from ..env.elf3_env import ELF3TrackingEnv
    from .gpu_selector import select_gpus
    from .callbacks import MotionSwitchCallback, EvalCallback
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from research.retarget_g1_to_elf3.rl.env.elf3_env import ELF3TrackingEnv
    from research.retarget_g1_to_elf3.rl.training.gpu_selector import select_gpus
    from research.retarget_g1_to_elf3.rl.training.callbacks import MotionSwitchCallback, EvalCallback


def make_env(rank: int, config_path: str, curriculum_phase: int = 1):
    """Create environment factory for vectorized training.

    Args:
        rank: Environment rank in vectorized setup
        config_path: Path to config file
        curriculum_phase: Curriculum phase (1 or 2)

    Returns:
        Environment factory function
    """
    def _init():
        env = ELF3TrackingEnv(config_path=config_path, curriculum_phase=curriculum_phase)
        return env
    return _init


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _callback_save_freq(freq_timesteps: int, n_envs: int) -> int:
    """Convert desired timesteps to SB3 callback calls."""
    return max(int(freq_timesteps) // max(int(n_envs), 1), 1)


def _make_vec_env(config_path: str, n_envs: int, curriculum_phase: int):
    """Create a vectorized environment, using a subprocess only when useful."""
    env_fns = [make_env(i, config_path, curriculum_phase=curriculum_phase) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method='spawn')


def _make_callbacks(config: dict, env, eval_env, checkpoint_dir: Path, phase: int, n_envs: int):
    """Build callbacks for the current curriculum phase."""
    train_cfg = config['training']
    logging_cfg = config.get('logging', {})
    return [
        CheckpointCallback(
            save_freq=_callback_save_freq(logging_cfg.get('checkpoint_freq', 500000), n_envs),
            save_path=str(checkpoint_dir),
            name_prefix=f"elf3_rl_phase{phase}",
            save_replay_buffer=True,
            save_vecnormalize=True
        ),
        MotionSwitchCallback(
            env=env,
            switch_freq=train_cfg.get('motion_switch_freq', 10000),
            verbose=1
        ),
        EvalCallback(
            eval_env=eval_env,
            train_env=env,
            eval_freq=_callback_save_freq(logging_cfg.get('eval_freq', 100000), n_envs),
            n_eval_episodes=logging_cfg.get('n_eval_episodes', 10),
            verbose=1
        )
    ]


def _find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    return Path.cwd()


def _resolve_output_path(path: str | Path, config_path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return _find_repo_root(Path(config_path).resolve()) / raw


def train(
    config_path: str = None,
    total_timesteps: int = None,
    n_gpus: int = None,
    seed: int = 42,
    resume_from: str = None,
    curriculum_phase: int = None
):
    """Train ELF3 RL policy using PPO with curriculum learning.

    Args:
        config_path: Path to config file (default: rl/config/default.yaml)
        total_timesteps: Override total training timesteps (if None, uses curriculum config)
        n_gpus: Number of GPUs to use
        seed: Random seed
        resume_from: Path to checkpoint to resume from
        curriculum_phase: Force specific curriculum phase (1 or 2). If None, uses auto curriculum.
    """
    # Set default config path
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "default.yaml")

    # Load configuration
    config = load_config(config_path)

    if n_gpus is None:
        n_gpus = int(config.get('gpu', {}).get('n_gpus', 0))

    # Select GPUs
    gpu_ids = select_gpus(n_gpus=n_gpus)
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        print(f"Using GPUs: {gpu_ids}")

    # Set random seed
    np.random.seed(seed)

    # Extract training parameters
    train_cfg = config['training']
    logging_cfg = config.get('logging', {})
    curriculum_cfg = config['curriculum']
    n_envs = train_cfg.get('num_envs', train_cfg.get('n_envs', 1))
    n_steps = train_cfg['n_steps']
    batch_size = train_cfg['batch_size']
    n_epochs = train_cfg['n_epochs']
    learning_rate = train_cfg['learning_rate']
    gamma = train_cfg['gamma']
    gae_lambda = train_cfg['gae_lambda']
    clip_range = train_cfg['clip_range']
    ent_coef = train_cfg['ent_coef']

    # Determine curriculum phase
    use_curriculum = curriculum_cfg.get('enabled', True)
    if curriculum_phase is None:
        # Auto curriculum: start with phase 1
        current_phase = 1
    else:
        current_phase = curriculum_phase

    # Determine total timesteps based on curriculum
    if total_timesteps is None:
        if use_curriculum and curriculum_phase is None:
            # Use sum of both phases
            total_timesteps = curriculum_cfg['phase1']['timesteps'] + curriculum_cfg['phase2']['timesteps']
        else:
            total_timesteps = train_cfg['total_timesteps']

    # Create output directories
    checkpoint_dir = _resolve_output_path(logging_cfg.get('checkpoint_dir', 'checkpoints/elf3_rl'), config_path)
    log_dir = _resolve_output_path(logging_cfg.get('tensorboard_log', 'runs/elf3_rl'), config_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Curriculum enabled: {use_curriculum}")
    print(f"Starting phase: {current_phase}")
    print(f"Environments: {n_envs}")
    print(f"Steps per env: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # Create vectorized environment
    print("Creating vectorized environment...")
    env = _make_vec_env(config_path, n_envs, current_phase)
    eval_env = _make_vec_env(config_path, 1, current_phase)

    # Wrap with VecNormalize
    vec_normalize_path = checkpoint_dir / f"vec_normalize_phase{current_phase}.pkl"
    if resume_from and vec_normalize_path.exists():
        print(f"Loading VecNormalize stats from {vec_normalize_path}")
        env = VecNormalize.load(str(vec_normalize_path), env)
    else:
        env = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Create PPO model
    print("Creating PPO model...")
    network_cfg = config.get('network', {})
    policy_kwargs = {
        'net_arch': network_cfg.get('net_arch', [512, 256]),
        'activation_fn': getattr(__import__('torch').nn, network_cfg.get('activation', 'ReLU'))
    }

    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs,
            seed=seed
        )

    # Create callbacks
    callbacks = _make_callbacks(config, env, eval_env, checkpoint_dir, current_phase, n_envs)

    # Curriculum learning logic
    if use_curriculum and curriculum_phase is None:
        # Two-phase training
        phase1_timesteps = curriculum_cfg['phase1']['timesteps']
        phase2_timesteps = curriculum_cfg['phase2']['timesteps']

        print(f"\n{'='*60}")
        print(f"Phase 1 Training: Simple Actions + Low DR")
        print(f"{'='*60}")
        print(f"Timesteps: {phase1_timesteps:,}")
        print(f"Categories: {curriculum_cfg['phase1']['categories']}")
        print(f"Press Ctrl+C to stop training and save checkpoint.\n")

        try:
            model.learn(
                total_timesteps=phase1_timesteps,
                callback=callbacks,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            _save_final(model, env, checkpoint_dir, vec_normalize_path, log_dir, current_phase, extra_envs=[eval_env])
            return model

        # Save phase 1 model
        phase1_model_path = checkpoint_dir / "phase1_model"
        print(f"\nSaving phase 1 model to {phase1_model_path}")
        model.save(str(phase1_model_path))

        # Transition to phase 2
        print(f"\n{'='*60}")
        print(f"Transitioning to Phase 2: All Actions + High DR")
        print(f"{'='*60}\n")

        current_phase = 2

        # Recreate environments with phase 2
        env.close()
        eval_env.close()
        env = _make_vec_env(config_path, n_envs, 2)
        eval_env = _make_vec_env(config_path, 1, 2)

        # Create new VecNormalize for phase 2 (fresh statistics)
        vec_normalize_path = checkpoint_dir / "vec_normalize_phase2.pkl"
        env = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

        # Load model weights into new environment
        model.set_env(env)

        # Rebuild callbacks for phase 2 so they point at the new envs.
        callbacks = _make_callbacks(config, env, eval_env, checkpoint_dir, current_phase, n_envs)

        print(f"Timesteps: {phase2_timesteps:,}")
        print(f"Categories: {curriculum_cfg['phase2']['categories']}")
        print(f"Press Ctrl+C to stop training and save checkpoint.\n")

        try:
            model.learn(
                total_timesteps=phase1_timesteps + phase2_timesteps,
                callback=callbacks,
                reset_num_timesteps=False,  # Continue from phase 1
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            _save_final(model, env, checkpoint_dir, vec_normalize_path, log_dir, current_phase, extra_envs=[eval_env])
            return model
    else:
        # Single phase training (manual phase or no curriculum)
        print(f"\nStarting training for {total_timesteps:,} timesteps...")
        print("Press Ctrl+C to stop training and save checkpoint.\n")

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")

    # Save final model
    _save_final(model, env, checkpoint_dir, vec_normalize_path, log_dir, current_phase, extra_envs=[eval_env])

    return model


def _save_final(model, env, checkpoint_dir, vec_normalize_path, log_dir, phase, extra_envs=None):
    """Save final model and cleanup."""
    final_model_path = checkpoint_dir / f"final_model_phase{phase}"
    print(f"\nSaving final model to {final_model_path}")
    model.save(str(final_model_path))
    env.save(str(vec_normalize_path))

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Final model: {final_model_path}.zip")
    print(f"VecNormalize stats: {vec_normalize_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    print(f"{'='*60}\n")

    # Close environment
    env.close()
    for extra_env in extra_envs or []:
        extra_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ELF3 RL policy")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps")
    parser.add_argument("--n-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: config gpu.n_gpus)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--curriculum-phase", type=int, default=None,
                        choices=[1, 2],
                        help="Force specific curriculum phase (1 or 2). "
                             "If not set, uses automatic two-phase curriculum.")

    args = parser.parse_args()

    train(
        config_path=args.config,
        total_timesteps=args.timesteps,
        n_gpus=args.n_gpus,
        seed=args.seed,
        resume_from=args.resume,
        curriculum_phase=args.curriculum_phase
    )
