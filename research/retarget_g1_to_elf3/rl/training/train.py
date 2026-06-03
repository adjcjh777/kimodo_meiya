"""Training script for ELF3 RL framework."""

import os
import sys
from copy import deepcopy
from pathlib import Path
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

try:
    from ..env.elf3_env import ELF3TrackingEnv
    from ..env.reference_motion import (
        POLICY_IDS,
        ReferenceMotionManager,
        load_policy_route,
        load_yaml,
        policy_motion_stems,
    )
    from .gpu_selector import select_gpus
    from .callbacks import MotionSwitchCallback, EvalCallback
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
    from research.retarget_g1_to_elf3.rl.training.gpu_selector import select_gpus
    from research.retarget_g1_to_elf3.rl.training.callbacks import MotionSwitchCallback, EvalCallback


DEFAULT_RL_OUTPUT_ROOT = Path("research/retarget_g1_to_elf3/rl")


def make_env(
    rank: int,
    config_path: str,
    curriculum_phase: int = 1,
    policy_id: str | None = None,
    motion_stems: list[str] | None = None,
    reward_weights: dict | None = None,
):
    """Create environment factory for vectorized training.

    Args:
        rank: Environment rank in vectorized setup
        config_path: Path to config file
        curriculum_phase: Curriculum phase (1 or 2)

    Returns:
        Environment factory function
    """
    def _init():
        env = ELF3TrackingEnv(
            config_path=config_path,
            curriculum_phase=curriculum_phase,
            policy_id=policy_id,
            motion_stems=motion_stems,
            load_mode="train",
            include_classified_clips=True,
            include_root_clips=False,
            reward_weights=reward_weights,
        )
        monitored_env = Monitor(env)
        monitored_env.get_clip_names = env.get_clip_names
        return monitored_env
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


def _make_vec_env(
    config_path: str,
    n_envs: int,
    curriculum_phase: int,
    policy_id: str | None = None,
    motion_stems: list[str] | None = None,
    reward_weights: dict | None = None,
):
    """Create a vectorized environment, using a subprocess only when useful."""
    env_fns = [
        make_env(
            i,
            config_path,
            curriculum_phase=curriculum_phase,
            policy_id=policy_id,
            motion_stems=motion_stems,
            reward_weights=reward_weights,
        )
        for i in range(n_envs)
    ]
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
            eval_freq=logging_cfg.get('eval_freq', 100000),
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
    return _resolve_config_path(config.get(key, Path("research/retarget_g1_to_elf3/rl/config") / filename), config_path)


def _load_policy_context(config: dict, config_path: str, policy_id: str | None) -> dict:
    if policy_id is None:
        return {
            "policy_id": None,
            "policy_config": None,
            "all_motion_stems": None,
            "reward_weights": None,
            "route_stats": None,
        }
    if policy_id not in POLICY_IDS:
        raise ValueError(f"Unknown policy_id {policy_id!r}; expected one of {POLICY_IDS}")

    route_path = _policy_file(config, config_path, "policy_route_path", "policy_route.yaml")
    policy_configs_path = _policy_file(config, config_path, "policy_configs_path", "policy_configs.yaml")
    motion_dir = _resolve_config_path(config["motion_dir"], config_path)

    route = load_policy_route(route_path)
    route_stats = ReferenceMotionManager.validate_policy_route(motion_dir, route)
    policy_configs = load_yaml(policy_configs_path).get("policies", {})
    if policy_id not in policy_configs:
        raise ValueError(f"policy_configs.yaml is missing {policy_id}")

    policy_config = deepcopy(policy_configs[policy_id])
    all_motion_stems = policy_motion_stems(route, policy_id)
    phase1_motions = set(policy_config.get("curriculum", {}).get("phase1_motions", []))
    unknown_phase1 = sorted(phase1_motions - set(all_motion_stems))
    if unknown_phase1:
        raise ValueError(f"{policy_id} phase1_motions are not routed to this policy: {unknown_phase1}")

    return {
        "policy_id": policy_id,
        "policy_config": policy_config,
        "all_motion_stems": all_motion_stems,
        "reward_weights": policy_config.get("reward_weights"),
        "route_stats": route_stats,
    }


def _phase_motion_stems(policy_context: dict, phase: int) -> list[str] | None:
    all_motion_stems = policy_context.get("all_motion_stems")
    if all_motion_stems is None:
        return None
    if phase == 1:
        phase1 = policy_context["policy_config"].get("curriculum", {}).get("phase1_motions", [])
        return list(phase1) if phase1 else list(all_motion_stems)
    return list(all_motion_stems)


def _policy_timesteps(policy_context: dict, phase: int | None = None) -> int | None:
    policy_config = policy_context.get("policy_config")
    if policy_config is None:
        return None
    total = int(policy_config["total_timesteps"])
    if phase is None:
        return total
    curriculum = policy_config.get("curriculum", {})
    phase1_ratio = float(curriculum.get("phase1_ratio", 0.4))
    phase1_steps = int(total * phase1_ratio)
    if phase == 1:
        return phase1_steps
    return total - phase1_steps


def _apply_smoke_overrides(train_cfg: dict, total_timesteps: int | None) -> dict:
    train_cfg = dict(train_cfg)
    if total_timesteps is None:
        return train_cfg
    if total_timesteps > 1024:
        return train_cfg

    smoke_steps = max(int(total_timesteps), 2)
    train_cfg["num_envs"] = 1
    train_cfg["n_steps"] = smoke_steps
    train_cfg["batch_size"] = min(max(2, train_cfg.get("batch_size", smoke_steps)), smoke_steps)
    train_cfg["n_epochs"] = min(int(train_cfg.get("n_epochs", 1)), 1)
    return train_cfg


def train(
    config_path: str = None,
    total_timesteps: int = None,
    n_gpus: int = None,
    seed: int = 42,
    resume_from: str = None,
    curriculum_phase: int = None,
    policy_id: str = None,
):
    """Train ELF3 RL policy using PPO with curriculum learning.

    Args:
        config_path: Path to config file (default: rl/config/default.yaml)
        total_timesteps: Override total training timesteps (if None, uses curriculum config)
        n_gpus: Number of GPUs to use
        seed: Random seed
        resume_from: Path to checkpoint to resume from
        curriculum_phase: Force specific curriculum phase (1 or 2). If None, uses auto curriculum.
        policy_id: Optional skill policy id to train.
    """
    # Set default config path
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "default.yaml")

    # Load configuration
    config = load_config(config_path)
    policy_context = _load_policy_context(config, config_path, policy_id)
    timesteps_override = total_timesteps is not None

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
    train_cfg = _apply_smoke_overrides(config['training'], total_timesteps)
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
    target_kl = train_cfg.get('target_kl')

    # Determine curriculum phase
    use_curriculum = curriculum_cfg.get('enabled', True)
    if curriculum_phase is None:
        # Auto curriculum: start with phase 1
        current_phase = 1
    else:
        current_phase = curriculum_phase

    # Determine total timesteps based on curriculum
    policy_total_timesteps = _policy_timesteps(policy_context)
    if total_timesteps is None:
        if policy_total_timesteps is not None:
            total_timesteps = (
                _policy_timesteps(policy_context, current_phase)
                if curriculum_phase is not None
                else policy_total_timesteps
            )
        elif use_curriculum and curriculum_phase is None:
            # Use sum of both phases
            total_timesteps = curriculum_cfg['phase1']['timesteps'] + curriculum_cfg['phase2']['timesteps']
        else:
            total_timesteps = train_cfg['total_timesteps']

    auto_curriculum = use_curriculum and curriculum_phase is None and not timesteps_override

    # Create output directories
    checkpoint_dir = _resolve_output_path(
        logging_cfg.get('checkpoint_dir', DEFAULT_RL_OUTPUT_ROOT / 'checkpoints' / 'elf3_rl'),
        config_path,
    )
    log_dir = _resolve_output_path(
        logging_cfg.get('tensorboard_log', DEFAULT_RL_OUTPUT_ROOT / 'runs' / 'elf3_rl'),
        config_path,
    )
    if policy_id is not None:
        checkpoint_dir = checkpoint_dir / policy_id
        log_dir = log_dir / policy_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    if policy_id is not None:
        print(f"Policy id: {policy_id}")
        print(f"Policy clips: {len(policy_context['all_motion_stems'])}")
        print(f"Route coverage: {policy_context['route_stats']}")
    print(f"Config: {config_path}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Curriculum enabled: {use_curriculum}")
    print(f"Starting phase: {current_phase}")
    print(f"Environments: {n_envs}")
    print(f"Steps per env: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Target KL: {target_kl}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # Create vectorized environment
    print("Creating vectorized environment...")
    phase_motion_stems = _phase_motion_stems(policy_context, current_phase)
    env = _make_vec_env(
        config_path,
        n_envs,
        current_phase,
        policy_id=policy_id,
        motion_stems=phase_motion_stems,
        reward_weights=policy_context.get("reward_weights"),
    )
    eval_env = _make_vec_env(
        config_path,
        1,
        current_phase,
        policy_id=policy_id,
        motion_stems=phase_motion_stems,
        reward_weights=policy_context.get("reward_weights"),
    )

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
    if policy_context.get("policy_config") is not None:
        network_cfg = dict(network_cfg)
        network_cfg["net_arch"] = policy_context["policy_config"].get("net_arch", network_cfg.get("net_arch"))
    policy_kwargs = {
        'net_arch': network_cfg.get('net_arch', [512, 256]),
        'activation_fn': getattr(__import__('torch').nn, network_cfg.get('activation', 'ReLU'))
    }

    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=env)
        model.target_kl = target_kl
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
            target_kl=target_kl,
            verbose=1,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs,
            seed=seed
        )

    # Create callbacks
    callbacks = _make_callbacks(config, env, eval_env, checkpoint_dir, current_phase, n_envs)

    # Curriculum learning logic
    if auto_curriculum:
        # Two-phase training
        if policy_id is not None:
            phase1_timesteps = _policy_timesteps(policy_context, phase=1)
            phase2_timesteps = _policy_timesteps(policy_context, phase=2)
        else:
            phase1_timesteps = curriculum_cfg['phase1']['timesteps']
            phase2_timesteps = curriculum_cfg['phase2']['timesteps']

        print(f"\n{'='*60}")
        print(f"Phase 1 Training: Simple Actions + Low DR")
        print(f"{'='*60}")
        print(f"Timesteps: {phase1_timesteps:,}")
        if policy_id is not None:
            print(f"Motions: {phase_motion_stems}")
        else:
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
            _save_final(model, env, checkpoint_dir, vec_normalize_path, log_dir, current_phase, extra_envs=[eval_env], policy_id=policy_id)
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
        phase_motion_stems = _phase_motion_stems(policy_context, 2)
        env = _make_vec_env(
            config_path,
            n_envs,
            2,
            policy_id=policy_id,
            motion_stems=phase_motion_stems,
            reward_weights=policy_context.get("reward_weights"),
        )
        eval_env = _make_vec_env(
            config_path,
            1,
            2,
            policy_id=policy_id,
            motion_stems=phase_motion_stems,
            reward_weights=policy_context.get("reward_weights"),
        )

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
        if policy_id is not None:
            print(f"Motions: {len(phase_motion_stems)} routed clips")
        else:
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
            _save_final(model, env, checkpoint_dir, vec_normalize_path, log_dir, current_phase, extra_envs=[eval_env], policy_id=policy_id)
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
    _save_final(model, env, checkpoint_dir, vec_normalize_path, log_dir, current_phase, extra_envs=[eval_env], policy_id=policy_id)

    return model


def _save_final(model, env, checkpoint_dir, vec_normalize_path, log_dir, phase, extra_envs=None, policy_id=None):
    """Save final model and cleanup."""
    final_model_path = checkpoint_dir / ("final_model" if policy_id is not None else f"final_model_phase{phase}")
    print(f"\nSaving final model to {final_model_path}")
    model.save(str(final_model_path))
    env.save(str(vec_normalize_path))

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Final model: {final_model_path}.zip")
    print("Format: SB3 native checkpoint ZIP for resume/eval; export ONNX/NPZ for deployment.")
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
    parser.add_argument("--policy-id", type=str, default=None,
                        choices=POLICY_IDS,
                        help="Skill policy id to train")
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
        curriculum_phase=args.curriculum_phase,
        policy_id=args.policy_id,
    )
