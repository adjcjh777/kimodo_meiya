"""Reward functions for ELF3 RL training."""

import numpy as np
from typing import Dict


MAX_COMPONENT_PENALTY = 100.0
MAX_TOTAL_REWARD = 100.0


def _bounded_exponential_penalty(
    value: float,
    *,
    k: float,
    alpha: float,
    max_penalty: float = MAX_COMPONENT_PENALTY,
) -> float:
    """Return a finite negative exponential penalty."""
    if value <= 0.0:
        return 0.0
    safe_k = max(float(k), 1e-12)
    max_exponent = np.log1p(max_penalty / safe_k)
    exponent = min(float(alpha) * float(value), float(max_exponent))
    penalty = safe_k * (np.exp(exponent) - 1.0)
    return -min(float(penalty), float(max_penalty))


# ==================== Tracking Rewards ====================

def joint_position_reward(
    current_joint_pos: np.ndarray,
    ref_joint_pos: np.ndarray,
    k: float = 5.0
) -> float:
    """Exponential reward for joint position tracking.

    Args:
        current_joint_pos: Current joint positions (29,)
        ref_joint_pos: Reference joint positions (29,)
        k: Exponential decay coefficient

    Returns:
        Reward in [0, 1], where 1 = perfect tracking
    """
    error = np.sum((current_joint_pos - ref_joint_pos) ** 2)
    return np.exp(-k * error)


def joint_velocity_reward(
    current_joint_vel: np.ndarray,
    ref_joint_vel: np.ndarray,
    k: float = 0.1
) -> float:
    """Exponential reward for joint velocity tracking.

    Args:
        current_joint_vel: Current joint velocities (29,)
        ref_joint_vel: Reference joint velocities (29,)
        k: Exponential decay coefficient

    Returns:
        Reward in [0, 1]
    """
    error = np.sum((current_joint_vel - ref_joint_vel) ** 2)
    return np.exp(-k * error)


def end_effector_position_reward(
    current_body_pos: np.ndarray,
    ref_body_pos: np.ndarray,
    body_indices: list,
    k: float = 10.0
) -> float:
    """Exponential reward for end effector (hands, feet) position tracking.

    Args:
        current_body_pos: Current body positions (30, 3)
        ref_body_pos: Reference body positions (30, 3)
        body_indices: List of body indices for end effectors
        k: Exponential decay coefficient

    Returns:
        Reward in [0, 1]
    """
    if not body_indices:
        return 0.0
    current_ee = current_body_pos[body_indices]
    ref_ee = ref_body_pos[body_indices]
    error = np.sum((current_ee - ref_ee) ** 2)
    return np.exp(-k * error)


def root_position_reward(
    current_root_pos: np.ndarray,
    ref_root_pos: np.ndarray,
    k: float = 2.0
) -> float:
    """Exponential reward for root position tracking (xy only).

    Args:
        current_root_pos: Current root position (3,)
        ref_root_pos: Reference root position (3,)
        k: Exponential decay coefficient

    Returns:
        Reward in [0, 1]
    """
    # Only track xy position, not z
    error = np.sum((current_root_pos[:2] - ref_root_pos[:2]) ** 2)
    return np.exp(-k * error)


# ==================== Stability Rewards ====================

def survival_reward() -> float:
    """Fixed reward for being alive.

    Returns:
        Fixed survival reward
    """
    return 0.01


def com_height_reward(
    current_height: float,
    target_height: float = 1.0,
    k: float = 10.0
) -> float:
    """Exponential reward for maintaining center of mass height.

    Args:
        current_height: Current COM height
        target_height: Target COM height
        k: Exponential decay coefficient

    Returns:
        Reward in [0, 1]
    """
    error = (current_height - target_height) ** 2
    return np.exp(-k * error)


def energy_penalty(
    torques: np.ndarray,
    k: float = 0.001
) -> float:
    """Penalty for high energy consumption (torque squared).

    Args:
        torques: Joint torques (29,)
        k: Penalty coefficient

    Returns:
        Negative penalty value (0 to -inf)
    """
    return -k * np.sum(torques ** 2)


def action_smoothness_penalty(
    current_action: np.ndarray,
    last_action: np.ndarray,
    k: float = 0.01
) -> float:
    """Penalty for action discontinuity.

    Args:
        current_action: Current action (29,)
        last_action: Previous action (29,)
        k: Penalty coefficient

    Returns:
        Negative penalty value
    """
    return -k * np.sum((current_action - last_action) ** 2)


# ==================== Foot Penalties ====================

def foot_penetration_penalty(
    foot_heights: np.ndarray,
    threshold: float = -0.02,
    k: float = 1.0,
    alpha: float = 50.0
) -> float:
    """Exponential penalty for foot penetration below ground.

    Args:
        foot_heights: Foot z positions (2,) for left and right foot
        threshold: Penetration threshold (z < threshold is penetration)
        k: Penalty coefficient
        alpha: Exponential growth coefficient

    Returns:
        Negative penalty value
    """
    penetration = np.maximum(threshold - foot_heights, 0.0)
    if np.sum(penetration) == 0:
        return 0.0

    return _bounded_exponential_penalty(np.sum(penetration), k=k, alpha=alpha)


def foot_airborne_penalty(
    current_foot_heights: np.ndarray,
    ref_foot_heights: np.ndarray,
    ref_foot_contacts: np.ndarray | None = None,
    airborne_threshold: float = 0.06,
    ref_threshold: float = 0.04,
    k: float = 0.5,
    alpha: float = 20.0,
    penalty_scale: float = 1.0
) -> float:
    """Penalty for unexpected foot airborne (when reference is on ground).

    Args:
        current_foot_heights: Current foot z positions (2,)
        ref_foot_heights: Reference foot z positions (2,)
        airborne_threshold: Current height > threshold is airborne
        ref_threshold: Reference height < threshold is on ground
        k: Penalty coefficient
        alpha: Exponential growth coefficient
        penalty_scale: Scale factor (0.1 for airborne-allowed actions)

    Returns:
        Negative penalty value
    """
    # Check which feet are unexpectedly airborne
    is_airborne = current_foot_heights > airborne_threshold
    if ref_foot_contacts is not None:
        ref_on_ground = np.asarray(ref_foot_contacts) > 0.5
    else:
        ref_on_ground = ref_foot_heights < ref_threshold
    unexpected_airborne = is_airborne & ref_on_ground

    if not np.any(unexpected_airborne):
        return 0.0

    # Calculate height above threshold for unexpected airborne feet
    height_above = np.maximum(current_foot_heights[unexpected_airborne] - airborne_threshold, 0.0)
    penalty = _bounded_exponential_penalty(np.sum(height_above), k=k, alpha=alpha)
    return penalty * penalty_scale


# ==================== Main Reward Computation ====================

def compute_total_reward(
    current_joint_pos: np.ndarray,
    current_joint_vel: np.ndarray,
    current_body_pos: np.ndarray,
    current_root_pos: np.ndarray,
    ref_joint_pos: np.ndarray,
    ref_joint_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_root_pos: np.ndarray,
    current_action: np.ndarray,
    last_action: np.ndarray,
    torques: np.ndarray,
    weights: Dict[str, float],
    params: Dict[str, float],
    end_effector_indices: list,
    foot_indices: list,
    current_foot_heights: np.ndarray | None = None,
    ref_foot_heights: np.ndarray | None = None,
    ref_foot_contacts: np.ndarray | None = None,
    airborne_action: bool = False
) -> float:
    """Compute total weighted reward.

    Args:
        current_joint_pos: Current joint positions (29,)
        current_joint_vel: Current joint velocities (29,)
        current_body_pos: Current body positions (30, 3)
        current_root_pos: Current root position (3,)
        ref_joint_pos: Reference joint positions (29,)
        ref_joint_vel: Reference joint velocities (29,)
        ref_body_pos: Reference body positions (30, 3)
        ref_root_pos: Reference root position (3,)
        current_action: Current action (29,)
        last_action: Previous action (29,)
        torques: Joint torques (29,)
        weights: Reward component weights
        params: Reward function parameters
        end_effector_indices: List of end effector body indices
        foot_indices: List of foot body indices (left, right)
        current_foot_heights: Optional left/right foot-bottom heights.
        ref_foot_heights: Optional left/right reference foot heights.
        ref_foot_contacts: Optional left/right reference contacts.
        airborne_action: Whether this action allows foot airborne

    Returns:
        Total weighted reward
    """
    # Tracking rewards (60%)
    r_joint_pos = joint_position_reward(
        current_joint_pos, ref_joint_pos, params['joint_pos_k']
    )
    r_joint_vel = joint_velocity_reward(
        current_joint_vel, ref_joint_vel, params['joint_vel_k']
    )
    r_ee_pos = end_effector_position_reward(
        current_body_pos, ref_body_pos, end_effector_indices, params['end_eff_k']
    )
    r_root_pos = root_position_reward(
        current_root_pos, ref_root_pos, params['root_pos_k']
    )

    tracking_reward = (
        weights['joint_position'] * r_joint_pos +
        weights['joint_velocity'] * r_joint_vel +
        weights['end_effector_pos'] * r_ee_pos +
        weights['root_position'] * r_root_pos
    )

    # Stability rewards (30%)
    r_survival = survival_reward()
    r_com_height = com_height_reward(
        current_root_pos[2], params['com_height_target'], params['com_height_k']
    )
    r_energy = energy_penalty(torques, params['energy_k'])
    r_smoothness = action_smoothness_penalty(
        current_action, last_action, params['action_smooth_k']
    )

    stability_reward = (
        weights['survival'] * r_survival +
        weights['com_height'] * r_com_height +
        weights['energy'] * r_energy +
        weights['smoothness'] * r_smoothness
    )

    # Foot penalties (10%)
    if current_foot_heights is None:
        current_foot_heights = current_body_pos[foot_indices, 2] if foot_indices else np.zeros(0)
    if ref_foot_heights is None:
        ref_foot_heights = ref_body_pos[foot_indices, 2] if foot_indices else np.zeros(0)

    p_penetration = foot_penetration_penalty(
        current_foot_heights,
        params['penetration_threshold'],
        params['penetration_k'],
        params['penetration_alpha']
    )

    # Apply action routing: reduce airborne penalty for airborne-allowed actions
    airborne_penalty_scale = params['airborne_penalty_scale'] if airborne_action else 1.0

    p_airborne = foot_airborne_penalty(
        current_foot_heights,
        ref_foot_heights,
        ref_foot_contacts,
        params['airborne_threshold'],
        params['airborne_ref_threshold'],
        params['airborne_k'],
        params['airborne_alpha'],
        airborne_penalty_scale
    )

    foot_penalty = (
        weights['foot_penetration'] * p_penetration +
        weights['foot_airborne'] * p_airborne
    )

    total = tracking_reward + stability_reward + foot_penalty
    total = np.nan_to_num(total, nan=-MAX_TOTAL_REWARD, posinf=MAX_TOTAL_REWARD, neginf=-MAX_TOTAL_REWARD)
    return float(np.clip(total, -MAX_TOTAL_REWARD, MAX_TOTAL_REWARD))
