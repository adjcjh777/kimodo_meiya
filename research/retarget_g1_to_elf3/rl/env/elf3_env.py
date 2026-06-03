"""ELF3 RL Training Environment."""

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from .reference_motion import ReferenceMotionManager, MotionClip
from .rewards import compute_total_reward


def _find_repo_root(start: Path) -> Optional[Path]:
    """Find the nearest parent containing .git."""
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    return None


def _resolve_existing_path(path: str | Path, *, config_path: Path) -> str:
    """Resolve config paths from cwd, config dir, or repository root."""
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return str(raw)

    candidates = [
        Path.cwd() / raw,
        config_path.parent / raw,
    ]
    repo_root = _find_repo_root(config_path.resolve()) or _find_repo_root(Path.cwd().resolve())
    if repo_root is not None:
        candidates.append(repo_root / raw)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((repo_root / raw if repo_root is not None else raw).resolve())


class ELF3TrackingEnv(gym.Env):
    """MuJoCo-based ELF3 tracking environment for RL training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        curriculum_phase: int = 1
    ):
        """Initialize ELF3 tracking environment.

        Args:
            config_path: Path to YAML config file
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            curriculum_phase: Curriculum phase (1 or 2) for training
        """
        super().__init__()

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        config_path = Path(config_path).expanduser().resolve()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path

        self.render_mode = render_mode
        self.curriculum_phase = curriculum_phase

        # Extract config values
        self.xml_path = _resolve_existing_path(self.config['xml_path'], config_path=config_path)
        self.motion_dir = _resolve_existing_path(self.config['motion_dir'], config_path=config_path)
        self.dt = self.config['sim']['dt']
        self.substeps = self.config['sim']['substeps']
        self.control_dt = self.substeps * self.dt
        self.max_steps = self.config['episode']['max_steps']

        # PD controller params
        self.kp = self.config['pd']['kp']
        self.kd = self.config['pd']['kd']
        self.action_scale = float(self.config['action']['scale'])

        # Termination thresholds
        self.fall_height = self.config['termination']['fall_height']
        self.max_pitch = self.config['termination']['max_pitch']
        self.max_roll = self.config['termination']['max_roll']
        self.tracking_error_threshold = self.config['termination']['tracking_error_threshold']
        self.tracking_error_duration = self.config['termination']['tracking_error_duration']
        self.fall_penalty = self.config['termination']['fall_penalty']
        self.timeout_reward = self.config['termination']['timeout_reward']
        self.tracking_fail_penalty = self.config['termination']['tracking_fail_penalty']

        # Reward weights and params
        self.reward_weights = self.config['reward']
        self.reward_params = self.config['reward_params']

        # Action routing
        self.airborne_height_threshold = self.config['action_routing']['airborne_height_threshold']
        self.airborne_penalty_scale = self.config['action_routing']['airborne_penalty_scale']
        self.reward_params = dict(self.reward_params)
        self.reward_params['airborne_penalty_scale'] = self.airborne_penalty_scale

        # Domain randomization (select phase)
        dr_config = self.config['domain_randomization']
        self.dr_enabled = dr_config['enabled']
        phase_key = f'phase{curriculum_phase}'
        if phase_key in dr_config:
            self.dr_root_pos_range = dr_config[phase_key]['root_pos_range']
            self.dr_root_vel_range = dr_config[phase_key]['root_vel_range']
            self.dr_pd_scale_range = dr_config[phase_key]['pd_gain_scale']
            self.dr_friction_range = dr_config[phase_key]['friction_range']
            self.dr_gravity_scale_range = dr_config[phase_key]['gravity_scale']
        else:
            self.dr_enabled = False

        # Curriculum learning
        curriculum_config = self.config['curriculum']
        self.curriculum_enabled = curriculum_config['enabled']
        phase_config = curriculum_config[f'phase{curriculum_phase}']
        self.allowed_categories = phase_config['categories']
        self.category_weights = phase_config['category_weights']
        self.initial_phase_random = phase_config['initial_phase_random']

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt
        self.default_gravity = self.model.opt.gravity.copy()
        self.default_geom_friction = self.model.geom_friction.copy()

        self.n_joints = self.model.nq - 7
        self.n_dofs = self.model.nv - 6
        if self.n_joints != 29 or self.n_dofs != 29:
            raise ValueError(
                f"ELF3TrackingEnv expects 29 actuated joints, got nq={self.model.nq}, nv={self.model.nv}"
            )
        self.joint_qpos_slice = slice(7, 7 + self.n_joints)
        self.joint_qvel_slice = slice(6, 6 + self.n_dofs)
        self.joint_force_slice = slice(6, 6 + self.n_dofs)

        # Reference motion manager
        self.motion_manager = ReferenceMotionManager(self.motion_dir, xml_path=self.xml_path)

        # State variables
        self.current_clip: Optional[MotionClip] = None
        self.phase = 0.0
        self.step_count = 0
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.action_before_last = np.zeros(self.n_joints, dtype=np.float32)
        self.tracking_error_counter = 0

        # End effector and foot indices
        self.body_names = [self.model.body(i).name for i in range(1, self.model.nbody)]
        self.body_name_to_data_index = {name: i for i, name in enumerate(self.body_names)}
        self.body_name_to_model_id = {self.model.body(i).name: i for i in range(1, self.model.nbody)}
        self.end_effector_indices = self._get_end_effector_indices()
        self.foot_indices = self._get_foot_indices()
        self.foot_body_ids = self._get_body_model_ids(['l_ankle_x_link', 'r_ankle_x_link'])
        self.left_foot_geom_ids = self._get_named_geom_indices("l_foot")
        self.right_foot_geom_ids = self._get_named_geom_indices("r_foot")

        # Define spaces
        # Observation: 385 dims
        # - 本体状态 (78): root_height(1) + root_lin_vel(3) + root_ang_vel(3) + gravity_proj(3)
        #                   + phase_sin_cos(2) + action_mean(1) + joint_pos(29) + joint_vel(29)
        #                   + root_quat(4) + root_pos(3)
        # - 参考动作 (67): ref_joint_pos(29) + ref_joint_vel(29) + ref_root_pos(3)
        #                   + ref_root_quat(4) + ref_foot_contact(2)
        # - 历史信息 (58): last_action(29) + action_before_last(29)
        # - 体状态 (180): body_pos(90) + body_vel(90)
        # - 脚部接触 (2): foot_contact(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(385,),
            dtype=np.float32
        )

        # Action: 29 dims (target joint angles)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32
        )

        # Rendering
        self.viewer = None
        self.renderer = None

    def _sample_clip(self) -> MotionClip:
        """Sample a reference motion clip respecting curriculum constraints."""
        if self.allowed_categories:
            # Filter clips by allowed categories
            valid_clips = [
                clip for clip in self.motion_manager.clips
                if clip.category in self.allowed_categories
            ]
            if not valid_clips:
                # Fallback to all clips if no valid ones
                valid_clips = self.motion_manager.clips
        else:
            valid_clips = self.motion_manager.clips

        # Weighted sampling by category
        if self.category_weights:
            # Group by category
            clips_by_category = {}
            for clip in valid_clips:
                if clip.category not in clips_by_category:
                    clips_by_category[clip.category] = []
                clips_by_category[clip.category].append(clip)

            # Sample category first
            categories = list(clips_by_category.keys())
            weights = [self.category_weights.get(cat, 1.0) for cat in categories]
            weights = np.array(weights)
            weights = weights / weights.sum()

            selected_category = np.random.choice(categories, p=weights)
            return np.random.choice(clips_by_category[selected_category])
        else:
            # Uniform random sampling
            return np.random.choice(valid_clips)

    def _get_end_effector_indices(self) -> list:
        """Get body-data indices for end effectors (hands and feet).

        Returns:
            List of body indices in arrays that exclude the world body.
        """
        return self._get_body_data_indices([
            'l_wrist_z_link', 'r_wrist_z_link', 'l_ankle_x_link', 'r_ankle_x_link'
        ])

    def _get_foot_indices(self) -> list:
        """Get body-data indices for feet (left and right).

        Returns:
            List of 2 body indices in arrays that exclude the world body.
        """
        return self._get_body_data_indices(['l_ankle_x_link', 'r_ankle_x_link'])

    def _get_body_data_indices(self, names: list[str]) -> list[int]:
        """Map body names to reference/current body arrays that exclude world."""
        missing = [name for name in names if name not in self.body_name_to_data_index]
        if missing:
            raise ValueError(f"ELF3 XML missing required bodies: {missing}")
        return [self.body_name_to_data_index[name] for name in names]

    def _get_body_model_ids(self, names: list[str]) -> list[int]:
        """Map body names to MuJoCo model body ids."""
        missing = [name for name in names if name not in self.body_name_to_model_id]
        if missing:
            raise ValueError(f"ELF3 XML missing required bodies: {missing}")
        return [self.body_name_to_model_id[name] for name in names]

    def _get_named_geom_indices(self, prefix: str) -> list[int]:
        """Get collision geom ids by name prefix."""
        geom_ids = []
        for geom_id in range(self.model.ngeom):
            name = self.model.geom(geom_id).name or ""
            if name.startswith(prefix) and name.endswith("_collision"):
                geom_ids.append(geom_id)
        return geom_ids

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """Reset environment and sample new reference motion.

        Returns:
            observation: Initial observation
            info: Additional information dict
        """
        super().reset(seed=seed)

        # Sample new reference clip (respecting curriculum constraints)
        requested_clip = options.get('clip') if options else None
        requested_clip_name = options.get('clip_name') if options else None
        if requested_clip is not None:
            self.current_clip = requested_clip
        elif requested_clip_name is not None:
            self.current_clip = next(
                (clip for clip in self.motion_manager.clips if clip.name == requested_clip_name),
                None,
            )
            if self.current_clip is None:
                raise ValueError(f"Unknown reference clip: {requested_clip_name}")
        else:
            self.current_clip = self._sample_clip()
        self.phase = 0.0
        if self.initial_phase_random:
            self.phase = np.random.uniform(0.0, 1.0)
        self.step_count = 0
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.action_before_last = np.zeros(self.n_joints, dtype=np.float32)
        self.tracking_error_counter = 0

        # Initialize from the sampled reference phase.
        ref_state = self.motion_manager.get_reference(self.current_clip, self.phase)
        ref_qpos = ref_state['qpos'].copy()
        ref_qvel = ref_state['qvel'].copy()

        self.model.opt.gravity[:] = self.default_gravity
        self.model.geom_friction[:] = self.default_geom_friction

        # Apply domain randomization to initial state
        if self.dr_enabled:
            # Randomize root position
            root_pos_offset = np.random.uniform(
                self.dr_root_pos_range[0],
                self.dr_root_pos_range[1],
                size=3
            )
            ref_qpos[:3] += root_pos_offset

            # Randomize root velocity
            root_vel_offset = np.random.uniform(
                self.dr_root_vel_range[0],
                self.dr_root_vel_range[1],
                size=6
            )
            ref_qvel[:6] += root_vel_offset

            # Randomize PD gains
            self.kp_scale = np.random.uniform(
                self.dr_pd_scale_range[0],
                self.dr_pd_scale_range[1]
            )
            self.kd_scale = np.random.uniform(
                self.dr_pd_scale_range[0],
                self.dr_pd_scale_range[1]
            )

            # Randomize friction
            friction = np.random.uniform(
                self.dr_friction_range[0],
                self.dr_friction_range[1]
            )
            self.model.geom_friction[0, 0] = friction  # ground plane sliding friction

            # Randomize gravity
            gravity_scale = np.random.uniform(
                self.dr_gravity_scale_range[0],
                self.dr_gravity_scale_range[1]
            )
            self.model.opt.gravity[:] = self.default_gravity
            self.model.opt.gravity[2] = self.default_gravity[2] * gravity_scale
        else:
            self.kp_scale = 1.0
            self.kd_scale = 1.0

        # Set state
        self.data.qpos[:] = ref_qpos
        self.data.qvel[:] = ref_qvel

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray):
        """Execute one control step.

        Args:
            action: Residual joint targets, shape (29,), range [-1, 1]

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode terminated (fall over or tracking fail)
            truncated: Whether episode truncated (time limit)
            info: Additional information
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get reference state at the beginning of this control interval.
        control_ref_state = self.motion_manager.get_reference(self.current_clip, self.phase)
        ref_joint_pos = control_ref_state['joint_pos']
        previous_action = self.last_action.copy()

        # Compute target joint positions (residual control)
        # action is in [-1, 1], scale to [-action_scale, action_scale]
        target_joint_pos = ref_joint_pos + action * self.action_scale

        # PD control to compute torques
        current_joint_pos = self.data.qpos[self.joint_qpos_slice]
        current_joint_vel = self.data.qvel[self.joint_qvel_slice]

        kp = self.kp * self.kp_scale
        kd = self.kd * self.kd_scale

        tau = kp * (target_joint_pos - current_joint_pos) - kd * current_joint_vel

        # Apply torques via qfrc_applied (skip root 6 DOF)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.joint_force_slice] = tau

        # Step simulation for substeps
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        # Update phase
        phase_increment = self.control_dt / self.current_clip.duration
        self.phase = (self.phase + phase_increment) % 1.0
        reward_ref_state = self.motion_manager.get_reference(self.current_clip, self.phase)

        simulation_unstable = not self._is_simulation_finite()
        if simulation_unstable:
            self._restore_finite_state(reward_ref_state)

        # Update step count
        self.step_count += 1

        # Store action history
        self.action_before_last = previous_action
        self.last_action = action.copy()

        # Get new observation
        obs = self._get_obs()

        if simulation_unstable:
            reward = self.tracking_fail_penalty
            terminated = True
            termination_reason = 'simulation_unstable'
        else:
            # Compute reward
            reward = self._compute_reward(action, reward_ref_state, previous_action)

            # Check termination
            terminated, termination_reason = self._check_termination(reward_ref_state)

        # Track tracking error
        current_joint_pos = self.data.qpos[self.joint_qpos_slice]
        tracking_error = np.mean((current_joint_pos - reward_ref_state['joint_pos']) ** 2)
        if tracking_error > self.tracking_error_threshold:
            self.tracking_error_counter += 1
        else:
            self.tracking_error_counter = 0

        truncated = self.step_count >= self.max_steps

        # Add termination penalties/rewards
        if terminated and termination_reason == 'fall':
            reward += self.fall_penalty
        elif terminated and termination_reason == 'tracking_fail':
            reward += self.tracking_fail_penalty
        elif truncated:
            reward += self.timeout_reward

        # Get info
        info = self._get_info()
        info['tracking_error'] = tracking_error
        info['termination_reason'] = termination_reason if terminated else None
        info['simulation_unstable'] = simulation_unstable

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector (385 dims).

        Structure:
        - 本体状态 (78): root_height(1) + root_lin_vel(3) + root_ang_vel(3) + gravity_proj(3)
                          + phase_sin_cos(2) + action_mean(1) + joint_pos(29) + joint_vel(29)
                          + root_quat(4) + root_pos(3)
        - 参考动作 (67): ref_joint_pos(29) + ref_joint_vel(29) + ref_root_pos(3)
                          + ref_root_quat(4) + ref_foot_contact(2)
        - 历史信息 (58): last_action(29) + action_before_last(29)
        - 体状态 (180): body_pos(90) + body_vel(90)
        - 脚部接触 (2): foot_contact(2)

        Returns:
            Observation vector of shape (385,)
        """
        obs_parts = []

        # 本体状态 - Part 1 (10 dims)
        root_height = self.data.qpos[2:3]  # 1
        root_lin_vel = self.data.qvel[0:3]  # 3
        root_ang_vel = self.data.qvel[3:6]  # 3

        # Gravity projection (3 dims)
        root_quat = self.data.qpos[3:7]
        gravity_world = np.array([0, 0, -1])
        gravity_body = self._quat_rotate_inv(root_quat, gravity_world)
        obs_parts.extend([root_height, root_lin_vel, root_ang_vel, gravity_body])

        # Phase encoding (2 dims)
        phase_sin = np.sin(2 * np.pi * self.phase)
        phase_cos = np.cos(2 * np.pi * self.phase)
        obs_parts.append(np.array([phase_sin, phase_cos]))

        # Action mean (1 dim)
        action_mean = np.array([np.mean(self.last_action)])
        obs_parts.append(action_mean)

        # Joint state (58 dims)
        joint_pos = self.data.qpos[self.joint_qpos_slice]  # 29
        joint_vel = self.data.qvel[self.joint_qvel_slice]  # 29
        obs_parts.extend([joint_pos, joint_vel])

        # Root state - Part 2 (7 dims)
        obs_parts.extend([root_quat, self.data.qpos[0:3]])  # quat(4) + pos(3)

        # 参考动作 (67 dims)
        ref_state = self.motion_manager.get_reference(self.current_clip, self.phase)
        ref_joint_pos = ref_state['joint_pos']  # 29
        ref_joint_vel = ref_state['joint_vel']  # 29
        obs_parts.extend([ref_joint_pos, ref_joint_vel])

        ref_root_pos = ref_state['root_pos']  # 3
        ref_root_quat = ref_state['root_quat']  # 4
        obs_parts.extend([ref_root_pos, ref_root_quat])

        ref_foot_contact = ref_state['foot_contacts']  # 2
        obs_parts.append(ref_foot_contact)

        # 历史信息 (58 dims)
        obs_parts.extend([self.last_action, self.action_before_last])  # 29 + 29

        # 体状态 (180 dims)
        body_pos = self.data.xpos[1:31].flatten()  # 30 bodies × 3 = 90
        body_vel = self.data.cvel[1:31, 3:6].flatten()  # linear velocity, 30 bodies × 3 = 90
        obs_parts.extend([body_pos, body_vel])

        # 脚部接触 (2 dims)
        foot_heights = self._foot_bottom_heights()
        foot_contacts = (foot_heights < 0.03).astype(np.float32)
        obs_parts.append(foot_contacts)

        # Concatenate all parts
        obs = np.concatenate(obs_parts).astype(np.float32)

        # Sanity check
        assert obs.shape == (385,), f"Observation shape mismatch: expected (385,), got {obs.shape}"

        return obs

    def _compute_reward(self, action: np.ndarray, ref_state: Dict, previous_action: np.ndarray) -> float:
        """Compute reward for current state.

        Args:
            action: Current action (29,)
            ref_state: Reference state dict from motion manager

        Returns:
            Total reward value
        """
        current_joint_pos = self.data.qpos[self.joint_qpos_slice]
        current_joint_vel = self.data.qvel[self.joint_qvel_slice]
        current_body_pos = self.data.xpos[1:31]  # (30, 3)
        current_root_pos = self.data.qpos[0:3]

        ref_joint_pos = ref_state['joint_pos']
        ref_joint_vel = ref_state['joint_vel']
        ref_body_pos = ref_state['body_pos_w']
        ref_root_pos = ref_state['root_pos']

        # Get torques (qfrc_applied for actuated joints)
        torques = self.data.qfrc_applied[self.joint_force_slice]

        # Check if this action allows airborne feet
        # (auto-detect based on reference motion contact state)
        ref_foot_contacts = ref_state['foot_contacts']
        airborne_action = np.any(ref_foot_contacts < 0.5)

        reward = compute_total_reward(
            current_joint_pos=current_joint_pos,
            current_joint_vel=current_joint_vel,
            current_body_pos=current_body_pos,
            current_root_pos=current_root_pos,
            ref_joint_pos=ref_joint_pos,
            ref_joint_vel=ref_joint_vel,
            ref_body_pos=ref_body_pos,
            ref_root_pos=ref_root_pos,
            current_action=action,
            last_action=previous_action,
            torques=torques,
            weights=self.reward_weights,
            params=self.reward_params,
            end_effector_indices=self.end_effector_indices,
            foot_indices=self.foot_indices,
            current_foot_heights=self._foot_bottom_heights(),
            ref_foot_contacts=ref_foot_contacts,
            airborne_action=airborne_action
        )

        return reward

    def _check_termination(self, ref_state: Dict) -> tuple[bool, str]:
        """Check if episode should terminate.

        Args:
            ref_state: Reference state dict (unused but kept for consistency)

        Returns:
            Tuple of (terminated: bool, reason: str)
            - terminated: Whether episode should end
            - reason: 'fall', 'tracking_fail', or 'none'
        """
        root_height = self.data.qpos[2]
        root_quat = self.data.qpos[3:7]

        # Compute pitch and roll from quaternion
        pitch, roll = self._quat_to_pitch_roll(root_quat)

        # Check fall conditions
        if root_height < self.fall_height:
            return True, 'fall'
        if abs(pitch) > self.max_pitch:
            return True, 'fall'
        if abs(roll) > self.max_roll:
            return True, 'fall'

        # Check tracking failure (sustained high error)
        if self.tracking_error_counter >= self.tracking_error_duration:
            return True, 'tracking_fail'

        return False, 'none'

    def _foot_bottom_heights(self) -> np.ndarray:
        """Return left/right foot bottom heights above the world ground plane."""
        heights = []
        for geom_ids, fallback_body_id in zip(
            [self.left_foot_geom_ids, self.right_foot_geom_ids],
            self.foot_body_ids,
        ):
            if geom_ids:
                heights.append(min(self._geom_bottom_z(geom_id) for geom_id in geom_ids))
            else:
                heights.append(float(self.data.xpos[fallback_body_id, 2]))
        return np.asarray(heights, dtype=np.float32)

    def _geom_bottom_z(self, geom_id: int) -> float:
        """Approximate the lowest world z point of a MuJoCo geom."""
        geom_type = self.model.geom_type[geom_id]
        size = self.model.geom_size[geom_id]
        z = float(self.data.geom_xpos[geom_id, 2])

        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = float(size[0])
            half_len = float(size[1])
            axis_z = float(self.data.geom_xmat[geom_id].reshape(3, 3)[2, 2])
            return z - radius - half_len * abs(axis_z)
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return z - float(size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            xmat = self.data.geom_xmat[geom_id].reshape(3, 3)
            return z - float(np.sum(np.abs(xmat[2]) * size[:3]))
        return z

    def _is_simulation_finite(self) -> bool:
        """Check whether the MuJoCo state remains valid for policy observation."""
        arrays = [
            self.data.qpos,
            self.data.qvel,
            self.data.qacc,
            self.data.xpos,
            self.data.cvel,
        ]
        return all(np.all(np.isfinite(array)) for array in arrays)

    def _restore_finite_state(self, ref_state: Dict) -> None:
        """Recover a finite terminal observation after MuJoCo instability."""
        self.data.qpos[:] = ref_state['qpos']
        self.data.qvel[:] = ref_state['qvel']
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {
            'phase': self.phase,
            'step_count': self.step_count,
            'clip_name': self.current_clip.name if self.current_clip else None,
            'clip_category': self.current_clip.category if self.current_clip else None,
            'control_dt': self.control_dt,
        }

    def _quat_rotate_inv(self, quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Rotate vector by inverse quaternion (conjugate)."""
        # quat = [w, x, y, z]
        # conjugate = [w, -x, -y, -z]
        q_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        return self._quat_rotate(q_conj, vec)

    def _quat_rotate(self, quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion."""
        # quat = [w, x, y, z]
        # vec = [x, y, z]
        q_vec = np.array([0, vec[0], vec[1], vec[2]])
        q_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])

        # q * v * q_conj
        t = 2 * np.cross(quat[1:4], vec)
        result = vec + quat[0] * t + np.cross(quat[1:4], t)
        return result

    def _quat_to_pitch_roll(self, quat: np.ndarray):
        """Extract pitch and roll from quaternion."""
        w, x, y, z = quat

        # Pitch (rotation around y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Roll (rotation around x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        return pitch, roll

    def render(self):
        """Render environment."""
        if self.render_mode is None:
            return None

        if self.viewer is None and self.render_mode == 'human':
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        if self.render_mode == 'rgb_array':
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
