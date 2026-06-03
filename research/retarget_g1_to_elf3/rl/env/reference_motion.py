"""Reference motion loading and phase management for RL training."""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import mujoco


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
    joint_names: Optional[np.ndarray] = None
    body_names: Optional[np.ndarray] = None

    def __len__(self):
        return self.n_frames


class ReferenceMotionManager:
    """Manages loading and sampling of reference motion clips."""

    CATEGORIES = ["locomotion", "upper_body", "compound", "balance"]

    def __init__(self, motion_dir: str, xml_path: Optional[str] = None):
        """Initialize reference motion manager.

        Args:
            motion_dir: Path to directory containing NPZ files organized by category
            xml_path: Path to MuJoCo XML (used for qvel computation if needed)
        """
        self.motion_dir = Path(motion_dir)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path)) if xml_path else None
        self.clips: List[MotionClip] = []
        self.clips_by_category: Dict[str, List[MotionClip]] = {cat: [] for cat in self.CATEGORIES}

        self._load_all_clips()

        if len(self.clips) == 0:
            raise ValueError(f"No NPZ files found in {motion_dir}")

    def _load_all_clips(self):
        """Scan subdirectories and load all NPZ clips."""
        for category in self.CATEGORIES:
            category_dir = self.motion_dir / category
            if not category_dir.exists():
                continue

            for npz_file in sorted(category_dir.glob("*.npz")):
                try:
                    clip = self._load_clip(npz_file, category)
                    self.clips.append(clip)
                    self.clips_by_category[category].append(clip)
                except Exception as e:
                    print(f"Warning: Failed to load {npz_file}: {e}")

        # Also load clips from root directory (for backward compatibility with existing 10 clips)
        for npz_file in sorted(self.motion_dir.glob("*.npz")):
            name = npz_file.stem
            if any(c.name == name for c in self.clips):
                continue  # skip already loaded
            try:
                clip = self._load_clip(npz_file, "uncategorized")
                self.clips.append(clip)
            except Exception as e:
                print(f"Warning: Failed to load {npz_file}: {e}")

        print(f"Loaded {len(self.clips)} reference motion clips:")
        for category in self.CATEGORIES:
            n = len(self.clips_by_category[category])
            if n > 0:
                print(f"  {category}: {n} clips")

    def _load_clip(self, npz_path: Path, category: str) -> MotionClip:
        """Load a single motion clip from NPZ file."""
        data = np.load(npz_path, allow_pickle=False)

        required = [
            'qpos_elf3', 'joint_pos', 'joint_vel',
            'body_pos_w', 'body_lin_vel_w', 'foot_contacts', 'fps',
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"missing required fields: {missing}")

        qpos = data['qpos_elf3']
        if qpos.ndim == 3:
            qpos = qpos[0]
        if qpos.ndim != 2:
            raise ValueError(f"qpos_elf3 must be 2-D, got {qpos.shape}")

        joint_pos = data['joint_pos']
        joint_vel = data['joint_vel']
        body_pos_w = data['body_pos_w']
        body_lin_vel_w = data['body_lin_vel_w']
        foot_contacts = self._collapse_foot_contacts(data['foot_contacts'])
        fps = float(np.asarray(data['fps']).reshape(-1)[0])
        n_frames = qpos.shape[0]
        duration = n_frames / fps

        joint_names = data.get('joint_names', None)
        body_names = data.get('body_names', None)

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
            mujoco.mju_quat2Vel(qvel[frame_idx, 3:6], rel, dt)
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
        """Sample a random clip, stratified by category.

        Ensures each category is sampled proportionally.
        If a category has no clips, it is skipped.
        """
        available_categories = [
            cat for cat in self.CATEGORIES
            if len(self.clips_by_category[cat]) > 0
        ]

        if not available_categories:
            return np.random.choice(self.clips)

        # Pick a random category first, then a random clip within it
        category = np.random.choice(available_categories)
        clip = np.random.choice(self.clips_by_category[category])
        return clip

    def get_reference(self, clip: MotionClip, phase: float) -> Dict[str, np.ndarray]:
        """Get reference state at a given phase via linear interpolation.

        Args:
            clip: The motion clip to sample from
            phase: Phase in [0, 1], wraps around for looping

        Returns:
            Dict with interpolated joint_pos, joint_vel, body_pos_w, body_lin_vel_w,
            root_pos, root_quat, foot_contacts
        """
        phase = phase % 1.0
        n = clip.n_frames

        # Compute continuous frame index
        frame_idx = phase * (n - 1)
        frame_low = int(np.floor(frame_idx))
        frame_high = min(frame_low + 1, n - 1)
        alpha = frame_idx - frame_low

        # Linear interpolation
        def lerp(a, b, t):
            return a * (1 - t) + b * t

        joint_pos = lerp(clip.joint_pos[frame_low], clip.joint_pos[frame_high], alpha)
        joint_vel = lerp(clip.joint_vel[frame_low], clip.joint_vel[frame_high], alpha)
        body_pos_w = lerp(clip.body_pos_w[frame_low], clip.body_pos_w[frame_high], alpha)
        body_lin_vel_w = lerp(clip.body_lin_vel_w[frame_low], clip.body_lin_vel_w[frame_high], alpha)

        # Extract root position and quaternion from qpos
        qpos = lerp(clip.qpos[frame_low], clip.qpos[frame_high], alpha)
        qvel = lerp(clip.qvel[frame_low], clip.qvel[frame_high], alpha)
        root_pos = qpos[:3]
        root_quat = qpos[3:7]
        quat_norm = np.linalg.norm(root_quat)
        if quat_norm > 0:
            root_quat = root_quat / quat_norm
            qpos[3:7] = root_quat

        # Use nearest neighbor for boolean foot_contacts
        foot_contacts = clip.foot_contacts[frame_low]

        return {
            'joint_pos': joint_pos.astype(np.float32),
            'joint_vel': joint_vel.astype(np.float32),
            'body_pos_w': body_pos_w.astype(np.float32),
            'body_lin_vel_w': body_lin_vel_w.astype(np.float32),
            'root_pos': root_pos.astype(np.float32),
            'root_quat': root_quat.astype(np.float32),
            'foot_contacts': foot_contacts,
            'qpos': qpos.astype(np.float32),
            'qvel': qvel.astype(np.float32),
        }

    def get_frame(self, clip: MotionClip, frame_idx: int) -> Dict[str, np.ndarray]:
        """Get reference state at a specific frame index."""
        frame_idx = frame_idx % clip.n_frames

        return {
            'joint_pos': clip.joint_pos[frame_idx],
            'joint_vel': clip.joint_vel[frame_idx],
            'body_pos_w': clip.body_pos_w[frame_idx],
            'body_lin_vel_w': clip.body_lin_vel_w[frame_idx],
            'foot_contacts': clip.foot_contacts[frame_idx],
            'qpos': clip.qpos[frame_idx],
            'qvel': clip.qvel[frame_idx],
        }
