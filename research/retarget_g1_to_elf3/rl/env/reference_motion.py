"""Reference motion loading and phase management for RL training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import mujoco
import numpy as np
import yaml


POLICY_IDS = ("locomotion_policy", "posture_balance_policy", "upper_body_policy")
LEGACY_ROOT_CATEGORY = "legacy_root"


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file."""
    with open(Path(path).expanduser(), "r") as f:
        return yaml.safe_load(f) or {}


def load_policy_route(path: str | Path) -> Dict[str, str]:
    """Load motion stem -> policy id route table."""
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"Policy route must be a mapping, got {type(data).__name__}")
    return {str(stem): str(policy_id) for stem, policy_id in data.items()}


def policy_motion_stems(policy_route: Dict[str, str], policy_id: str) -> List[str]:
    """Return sorted motion stems routed to one policy."""
    if policy_id not in POLICY_IDS:
        raise ValueError(f"Unknown policy_id {policy_id!r}; expected one of {POLICY_IDS}")
    return sorted(stem for stem, routed_policy in policy_route.items() if routed_policy == policy_id)


@dataclass
class MotionClip:
    """A single reference motion clip."""

    name: str
    category: str
    qpos: np.ndarray           # (T, 36) full qpos (root 7 + joints 29)
    qvel: np.ndarray           # (T, 35) full qvel (root 6 + joints 29)
    joint_pos: np.ndarray      # (T, 29) joint positions
    joint_vel: np.ndarray      # (T, 29) joint velocities
    body_pos_w: np.ndarray     # (T, 30, 3) body positions
    body_lin_vel_w: np.ndarray # (T, 30, 3) body linear velocities
    foot_contacts: np.ndarray  # (T, 2) left/right foot contact booleans
    fps: float
    duration: float
    n_frames: int
    source_path: Path
    is_legacy_root: bool = False
    joint_names: Optional[np.ndarray] = None
    body_names: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.n_frames


class ReferenceMotionManager:
    """Manages loading and sampling of reference motion clips."""

    CATEGORIES = ["locomotion", "upper_body", "compound", "balance"]
    ALL_CATEGORIES = [*CATEGORIES, LEGACY_ROOT_CATEGORY]
    MODES = {"train", "benchmark_eval", "debug_eval"}

    def __init__(
        self,
        motion_dir: str | Path,
        xml_path: Optional[str] = None,
        mode: str = "train",
        motion_stems: Optional[Sequence[str]] = None,
        include_classified_clips: bool = True,
        include_root_clips: bool = False,
    ):
        """Initialize reference motion manager.

        Args:
            motion_dir: Directory containing classified NPZ subdirectories.
            xml_path: MuJoCo XML path, used for qvel computation if needed.
            mode: Loading mode: train, benchmark_eval, or debug_eval.
            motion_stems: Optional explicit stem allow-list.
            include_classified_clips: Whether to scan classified subdirectories.
            include_root_clips: Whether to scan root-level legacy NPZ files.
        """
        if mode not in self.MODES:
            raise ValueError(f"Unknown reference motion mode {mode!r}; expected {sorted(self.MODES)}")
        if include_root_clips and mode != "debug_eval":
            raise ValueError("Root-level legacy clips may only be loaded in debug_eval mode")

        self.motion_dir = Path(motion_dir)
        self.mode = mode
        self.motion_stems = set(motion_stems) if motion_stems is not None else None
        self.include_classified_clips = include_classified_clips
        self.include_root_clips = include_root_clips
        self.model = mujoco.MjModel.from_xml_path(str(xml_path)) if xml_path else None
        self.clips: List[MotionClip] = []
        self.clips_by_category: Dict[str, List[MotionClip]] = {
            category: [] for category in self.ALL_CATEGORIES
        }

        self._load_all_clips()

        if len(self.clips) == 0:
            suffix = ""
            if self.motion_stems is not None:
                suffix = f" matching stems {sorted(self.motion_stems)}"
            raise ValueError(f"No NPZ files found in {motion_dir}{suffix}")

    @classmethod
    def classified_motion_paths(cls, motion_dir: str | Path) -> List[Path]:
        """Return all classified training motion paths."""
        root = Path(motion_dir)
        paths: List[Path] = []
        for category in cls.CATEGORIES:
            paths.extend(sorted((root / category).glob("*.npz")))
        return paths

    @classmethod
    def root_motion_paths(cls, motion_dir: str | Path) -> List[Path]:
        """Return root-level legacy baseline motion paths."""
        return sorted(Path(motion_dir).glob("*.npz"))

    @classmethod
    def validate_policy_route(
        cls,
        motion_dir: str | Path,
        policy_route: Dict[str, str],
        known_policy_ids: Iterable[str] = POLICY_IDS,
        expected_training_count: int | None = 100,
    ) -> Dict[str, int]:
        """Validate route coverage against the 100 classified training clips."""
        known_policy_ids = set(known_policy_ids)
        classified_paths = cls.classified_motion_paths(motion_dir)
        training_stems = [path.stem for path in classified_paths]
        training_stem_set = set(training_stems)

        duplicates = sorted({stem for stem in training_stems if training_stems.count(stem) > 1})
        missing = sorted(training_stem_set - set(policy_route))
        extra = sorted(set(policy_route) - training_stem_set)
        unknown_policy_ids = sorted({policy for policy in policy_route.values() if policy not in known_policy_ids})

        if duplicates:
            raise ValueError(f"Duplicate classified training motion stems: {duplicates}")
        if expected_training_count is not None and len(training_stems) != expected_training_count:
            raise ValueError(
                f"Expected {expected_training_count} classified training clips, got {len(training_stems)}"
            )
        if missing:
            raise ValueError(f"Policy route is missing {len(missing)} training clips: {missing}")
        if extra:
            raise ValueError(f"Policy route has {len(extra)} non-training clips: {extra}")
        if unknown_policy_ids:
            raise ValueError(f"Policy route uses unknown policy ids: {unknown_policy_ids}")

        return {
            "training_clips": len(training_stems),
            **{
                policy_id: sum(1 for stem in training_stems if policy_route[stem] == policy_id)
                for policy_id in sorted(known_policy_ids)
            },
        }

    def _load_all_clips(self) -> None:
        """Scan configured sources and load NPZ clips."""
        if self.include_classified_clips:
            for category in self.CATEGORIES:
                category_dir = self.motion_dir / category
                if not category_dir.exists():
                    continue

                for npz_file in sorted(category_dir.glob("*.npz")):
                    if not self._should_load_stem(npz_file.stem):
                        continue
                    self._append_clip(npz_file, category, is_legacy_root=False)

        if self.include_root_clips:
            for npz_file in sorted(self.motion_dir.glob("*.npz")):
                if not self._should_load_stem(npz_file.stem):
                    continue
                self._append_clip(npz_file, LEGACY_ROOT_CATEGORY, is_legacy_root=True)

        if self.motion_stems is not None:
            loaded_stems = {clip.name for clip in self.clips}
            missing_stems = sorted(self.motion_stems - loaded_stems)
            if missing_stems:
                raise ValueError(f"Requested motion stems were not loaded: {missing_stems}")

        print(f"Loaded {len(self.clips)} reference motion clips:")
        for category in self.ALL_CATEGORIES:
            n = len(self.clips_by_category[category])
            if n > 0:
                print(f"  {category}: {n} clips")

    def _should_load_stem(self, stem: str) -> bool:
        return self.motion_stems is None or stem in self.motion_stems

    def _append_clip(self, npz_file: Path, category: str, is_legacy_root: bool) -> None:
        try:
            clip = self._load_clip(npz_file, category, is_legacy_root=is_legacy_root)
            self.clips.append(clip)
            self.clips_by_category[category].append(clip)
        except Exception as e:
            print(f"Warning: Failed to load {npz_file}: {e}")

    def _load_clip(self, npz_path: Path, category: str, is_legacy_root: bool = False) -> MotionClip:
        """Load a single motion clip from an NPZ file."""
        data = np.load(npz_path, allow_pickle=False)

        required = [
            "qpos_elf3", "joint_pos", "joint_vel",
            "body_pos_w", "body_lin_vel_w", "foot_contacts", "fps",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"missing required fields: {missing}")

        qpos = data["qpos_elf3"]
        if qpos.ndim == 3:
            qpos = qpos[0]
        if qpos.ndim != 2:
            raise ValueError(f"qpos_elf3 must be 2-D, got {qpos.shape}")

        joint_pos = data["joint_pos"]
        joint_vel = data["joint_vel"]
        body_pos_w = data["body_pos_w"]
        body_lin_vel_w = data["body_lin_vel_w"]
        foot_contacts = self._collapse_foot_contacts(data["foot_contacts"])
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        n_frames = qpos.shape[0]
        duration = n_frames / fps

        joint_names = data.get("joint_names", None)
        body_names = data.get("body_names", None)

        if qpos.shape[1] != 7 + joint_pos.shape[1]:
            raise ValueError(
                f"qpos_elf3 width {qpos.shape[1]} does not match joint_pos width {joint_pos.shape[1]}"
            )
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"joint_vel shape {joint_vel.shape} does not match joint_pos {joint_pos.shape}")
        if body_pos_w.shape[:2] != body_lin_vel_w.shape[:2]:
            raise ValueError(
                f"body_pos_w shape {body_pos_w.shape} does not match body_lin_vel_w {body_lin_vel_w.shape}"
            )

        qvel = self._qvel_from_qpos(qpos, joint_vel, fps)

        return MotionClip(
            name=npz_path.stem,
            category=category,
            qpos=qpos.astype(np.float32),
            qvel=qvel.astype(np.float32),
            joint_pos=joint_pos.astype(np.float32),
            joint_vel=joint_vel.astype(np.float32),
            body_pos_w=body_pos_w.astype(np.float32),
            body_lin_vel_w=body_lin_vel_w.astype(np.float32),
            foot_contacts=foot_contacts,
            fps=fps,
            duration=duration,
            n_frames=n_frames,
            source_path=npz_path,
            is_legacy_root=is_legacy_root,
            joint_names=joint_names,
            body_names=body_names,
        )

    def _qvel_from_qpos(self, qpos: np.ndarray, joint_vel: np.ndarray, fps: float) -> np.ndarray:
        """Compute MuJoCo qvel with the model's nq/nv convention."""
        if self.model is not None:
            qvel = np.zeros((qpos.shape[0], self.model.nv), dtype=np.float64)
            if qpos.shape[0] < 2:
                return qvel.astype(np.float32)
            dt = 1.0 / fps
            qpos64 = qpos.astype(np.float64)
            for frame_idx in range(1, qpos.shape[0]):
                mujoco.mj_differentiatePos(
                    self.model,
                    qvel[frame_idx],
                    dt,
                    qpos64[frame_idx - 1],
                    qpos64[frame_idx],
                )
            qvel[0] = qvel[1]
            return qvel.astype(np.float32)

        qvel = np.zeros((qpos.shape[0], 6 + joint_vel.shape[1]), dtype=np.float32)
        if qpos.shape[0] < 2:
            return qvel

        dt = 1.0 / fps
        qvel[1:, 0:3] = (qpos[1:, 0:3] - qpos[:-1, 0:3]) / dt
        for frame_idx in range(1, qpos.shape[0]):
            prev_inv = np.array(
                [
                    qpos[frame_idx - 1, 3],
                    -qpos[frame_idx - 1, 4],
                    -qpos[frame_idx - 1, 5],
                    -qpos[frame_idx - 1, 6],
                ],
                dtype=np.float64,
            )
            rel = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(rel, qpos[frame_idx, 3:7].astype(np.float64), prev_inv)
            root_ang_vel = np.zeros(3, dtype=np.float64)
            mujoco.mju_quat2Vel(root_ang_vel, rel, dt)
            qvel[frame_idx, 3:6] = root_ang_vel
        qvel[:, 6:] = joint_vel
        qvel[0] = qvel[1]
        return qvel

    @staticmethod
    def _collapse_foot_contacts(foot_contacts: np.ndarray) -> np.ndarray:
        """Convert raw contact points to left/right contacts."""
        contacts = np.asarray(foot_contacts)
        if contacts.ndim != 2:
            raise ValueError(f"foot_contacts must be 2-D, got {contacts.shape}")
        if contacts.shape[1] == 2:
            return contacts.astype(np.float32)
        if contacts.shape[1] == 4:
            left = np.any(contacts[:, :2], axis=1)
            right = np.any(contacts[:, 2:], axis=1)
            return np.stack([left, right], axis=1).astype(np.float32)
        raise ValueError(f"foot_contacts must have 2 or 4 columns, got {contacts.shape[1]}")

    def sample_clip(self) -> MotionClip:
        """Sample a random clip, stratified by category when possible."""
        available_categories = [
            category for category in self.CATEGORIES
            if len(self.clips_by_category[category]) > 0
        ]

        if not available_categories:
            return np.random.choice(self.clips)

        category = np.random.choice(available_categories)
        return np.random.choice(self.clips_by_category[category])

    def get_clip(self, name: str, prefer_legacy_root: bool = False) -> MotionClip:
        """Get a loaded clip by stem."""
        matches = [clip for clip in self.clips if clip.name == name]
        if not matches:
            raise ValueError(f"Unknown reference clip: {name}")
        if prefer_legacy_root:
            legacy_matches = [clip for clip in matches if clip.is_legacy_root]
            if legacy_matches:
                return legacy_matches[0]
        if len(matches) > 1:
            sources = [str(clip.source_path) for clip in matches]
            raise ValueError(f"Ambiguous reference clip {name!r}; matches: {sources}")
        return matches[0]

    def get_reference(
        self,
        clip: MotionClip,
        phase: float,
        frame_offset: int | float = 0,
    ) -> Dict[str, np.ndarray]:
        """Get reference state at phase plus an optional frame offset."""
        phase = phase % 1.0
        n = clip.n_frames

        frame_idx = (phase * (n - 1) + frame_offset) % n
        return self._interpolate_frame(clip, frame_idx)

    def _interpolate_frame(self, clip: MotionClip, frame_idx: float) -> Dict[str, np.ndarray]:
        """Interpolate a clip at a continuous frame index."""
        n = clip.n_frames
        frame_idx = frame_idx % n
        frame_low = int(np.floor(frame_idx))
        frame_high = (frame_low + 1) % n
        alpha = frame_idx - frame_low

        def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
            return a * (1 - t) + b * t

        joint_pos = lerp(clip.joint_pos[frame_low], clip.joint_pos[frame_high], alpha)
        joint_vel = lerp(clip.joint_vel[frame_low], clip.joint_vel[frame_high], alpha)
        body_pos_w = lerp(clip.body_pos_w[frame_low], clip.body_pos_w[frame_high], alpha)
        body_lin_vel_w = lerp(clip.body_lin_vel_w[frame_low], clip.body_lin_vel_w[frame_high], alpha)
        qpos = lerp(clip.qpos[frame_low], clip.qpos[frame_high], alpha)
        qvel = lerp(clip.qvel[frame_low], clip.qvel[frame_high], alpha)

        root_pos = qpos[:3]
        root_quat = qpos[3:7]
        quat_norm = np.linalg.norm(root_quat)
        if quat_norm > 0:
            root_quat = root_quat / quat_norm
            qpos[3:7] = root_quat

        foot_contacts = clip.foot_contacts[frame_low]

        return {
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "body_pos_w": body_pos_w.astype(np.float32),
            "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
            "root_pos": root_pos.astype(np.float32),
            "root_quat": root_quat.astype(np.float32),
            "foot_contacts": foot_contacts.astype(np.float32),
            "qpos": qpos.astype(np.float32),
            "qvel": qvel.astype(np.float32),
        }

    def get_frame(self, clip: MotionClip, frame_idx: int) -> Dict[str, np.ndarray]:
        """Get reference state at a specific frame index."""
        frame_idx = frame_idx % clip.n_frames
        qpos = clip.qpos[frame_idx]
        return {
            "joint_pos": clip.joint_pos[frame_idx],
            "joint_vel": clip.joint_vel[frame_idx],
            "body_pos_w": clip.body_pos_w[frame_idx],
            "body_lin_vel_w": clip.body_lin_vel_w[frame_idx],
            "root_pos": qpos[:3],
            "root_quat": qpos[3:7],
            "foot_contacts": clip.foot_contacts[frame_idx],
            "qpos": qpos,
            "qvel": clip.qvel[frame_idx],
        }
