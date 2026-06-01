#!/usr/bin/env python3
"""Apply per-frame flat-foot and knee-clearance IK constraints to ELF3 qpos.

This is a lightweight post-retargeting repair for motions where the direct
semantic qpos map leaves one foot floating or one knee too close to the floor.
It keeps the source motion fields and rewrites qpos_elf3 plus derived MJLab
fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.retarget_g1_to_elf3.retarget_g1_to_elf3_baseline import (  # noqa: E402
    DEFAULT_ELF3_XML,
    ELF3_BODY_NAMES,
    ELF3_JOINT_NAMES,
    ELF3_QPOS_COLUMNS,
    mjlab_compatible_fields,
)


LOWER_BODY_QPOS_WITH_ROOT = np.array(
    [
        2,  # root_z
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ],
    dtype=np.int64,
)
LOWER_BODY_JOINT_QPOS = LOWER_BODY_QPOS_WITH_ROOT[1:]


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez(tmp_path, **data)
    os.replace(tmp_path, path)


def qpos_bounds(model: mujoco.MjModel, qpos_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(len(qpos_indices), -np.inf, dtype=np.float64)
    hi = np.full(len(qpos_indices), np.inf, dtype=np.float64)
    for jnt_id in range(model.njnt):
        qadr = int(model.jnt_qposadr[jnt_id])
        matches = np.where(qpos_indices == qadr)[0]
        if not len(matches):
            continue
        if model.jnt_limited[jnt_id]:
            lo[matches[0]] = float(model.jnt_range[jnt_id, 0])
            hi[matches[0]] = float(model.jnt_range[jnt_id, 1])
    return lo, hi


def named_ids(model: mujoco.MjModel, obj_type: mujoco.mjtObj, prefix: str) -> list[int]:
    max_id = model.ngeom if obj_type == mujoco.mjtObj.mjOBJ_GEOM else model.nbody
    ids: list[int] = []
    for obj_id in range(max_id):
        name = mujoco.mj_id2name(model, obj_type, obj_id) or ""
        if name.startswith(prefix):
            ids.append(obj_id)
    return ids


def body_id(model: mujoco.MjModel, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if obj_id < 0:
        raise ValueError(f"ELF3 model does not contain body {name!r}")
    return int(obj_id)


def geom_lower_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    center = data.geom_xpos[geom_id]
    size = model.geom_size[geom_id]
    geom_type = int(model.geom_type[geom_id])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
        half_length = float(size[1])
        radius = float(size[0])
        return float(min((center + half_length * axis)[2], (center - half_length * axis)[2]) - radius)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(center[2] - size[0])
    return float(center[2])


def contacts_by_side(foot_contacts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contacts = np.asarray(foot_contacts, dtype=bool)
    if contacts.ndim != 2:
        raise ValueError(f"Expected foot_contacts shape [T, C], got {contacts.shape}")
    if contacts.shape[1] == 2:
        return contacts[:, 0].copy(), contacts[:, 1].copy()
    if contacts.shape[1] == 4:
        return np.any(contacts[:, [0, 1]], axis=1), np.any(contacts[:, [2, 3]], axis=1)
    if contacts.shape[1] == 6:
        return np.any(contacts[:, [0, 1]], axis=1), np.any(contacts[:, [3, 4]], axis=1)
    raise ValueError(f"Unsupported foot_contacts channel count: {contacts.shape[1]}")


def fill_short_contact_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return mask.copy()
    out = mask.copy()
    nframes = len(out)
    idx = 0
    while idx < nframes:
        if out[idx]:
            idx += 1
            continue
        start = idx
        while idx < nframes and not out[idx]:
            idx += 1
        end = idx
        if start > 0 and end < nframes and end - start <= max_gap:
            out[start:end] = True
    return out


def contact_segments(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        end = idx
        if end - start >= min_frames:
            segments.append((start, end))
    return segments


def xy_targets_from_contact_segments(
    xy: np.ndarray,
    contacts: np.ndarray,
    *,
    min_contact_frames: int,
    anchor_frames: int,
) -> np.ndarray:
    targets = np.full_like(xy, np.nan, dtype=np.float64)
    for start, end in contact_segments(contacts, min_contact_frames):
        anchor_end = min(end, start + max(1, anchor_frames))
        targets[start:end] = np.median(xy[start:anchor_end], axis=0)
    return targets


class FlatFootIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        foot_weight: float,
        flat_weight: float,
        xy_weight: float,
        knee_weight: float,
        ref_weight: float,
        smooth_weight: float,
        knee_clearance: float,
        damping: float,
        fd_eps: float,
        lock_root_z: bool,
        solve_root_xy: bool,
        root_xy_ref_multiplier: float,
        hip_roll_ref_multiplier: float,
        hip_yaw_ref_multiplier: float,
        ankle_roll_ref_multiplier: float,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos_indices = LOWER_BODY_JOINT_QPOS if lock_root_z else LOWER_BODY_QPOS_WITH_ROOT
        if solve_root_xy:
            self.qpos_indices = np.concatenate([np.array([0, 1], dtype=np.int64), self.qpos_indices])
        self.foot_weight = foot_weight
        self.flat_weight = flat_weight
        self.xy_weight = xy_weight
        self.knee_weight = knee_weight
        self.ref_weight = ref_weight
        self.smooth_weight = smooth_weight
        self.knee_clearance = knee_clearance
        self.damping = damping
        self.fd_eps = fd_eps

        self.left_foot_geoms = named_ids(model, mujoco.mjtObj.mjOBJ_GEOM, "l_foot")
        self.right_foot_geoms = named_ids(model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot")
        self.left_knee_geoms = named_ids(model, mujoco.mjtObj.mjOBJ_GEOM, "l_knee")
        self.right_knee_geoms = named_ids(model, mujoco.mjtObj.mjOBJ_GEOM, "r_knee")
        self.left_foot_body = body_id(model, "l_ankle_x_link")
        self.right_foot_body = body_id(model, "r_ankle_x_link")

        if not self.left_foot_geoms or not self.right_foot_geoms:
            raise ValueError("Could not find ELF3 foot collision geoms")
        if not self.left_knee_geoms or not self.right_knee_geoms:
            raise ValueError("Could not find ELF3 knee collision geoms")

        self.lower, self.upper = qpos_bounds(model, self.qpos_indices)
        self.ref_scale = np.ones(len(self.qpos_indices), dtype=np.float64)
        root_entries = np.where(self.qpos_indices == 2)[0]
        if len(root_entries):
            self.ref_scale[root_entries[0]] = 0.35
        self.ref_scale[np.isin(self.qpos_indices, [0, 1])] *= root_xy_ref_multiplier
        self.ref_scale[np.isin(self.qpos_indices, [11, 17])] *= hip_roll_ref_multiplier
        self.ref_scale[np.isin(self.qpos_indices, [12, 18])] *= hip_yaw_ref_multiplier
        self.ref_scale[np.isin(self.qpos_indices, [15, 21])] *= ankle_roll_ref_multiplier

    def forward(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def foot_bottoms(self, geom_ids: list[int]) -> np.ndarray:
        return np.array([geom_lower_z(self.model, self.data, geom_id) for geom_id in geom_ids], dtype=np.float64)

    def knee_bottom(self, geom_ids: list[int]) -> float:
        return float(min(geom_lower_z(self.model, self.data, geom_id) for geom_id in geom_ids))

    def residual(
        self,
        x: np.ndarray,
        q_ref: np.ndarray,
        left_xy: np.ndarray,
        right_xy: np.ndarray,
        left_contact: bool,
        right_contact: bool,
        prev_x: np.ndarray | None,
    ) -> np.ndarray:
        qpos = q_ref.copy()
        qpos[self.qpos_indices] = x
        self.forward(qpos)

        left_bottoms = self.foot_bottoms(self.left_foot_geoms)
        right_bottoms = self.foot_bottoms(self.right_foot_geoms)
        left_mat = self.data.xmat[self.left_foot_body].reshape(3, 3)
        right_mat = self.data.xmat[self.right_foot_body].reshape(3, 3)
        left_pos = self.data.xpos[self.left_foot_body]
        right_pos = self.data.xpos[self.right_foot_body]
        left_knee = self.knee_bottom(self.left_knee_geoms)
        right_knee = self.knee_bottom(self.right_knee_geoms)

        parts = []
        if left_contact:
            parts.extend(
                [
                    self.foot_weight * left_bottoms,
                    self.flat_weight * left_mat[:2, 2],
                    self.xy_weight * (left_pos[:2] - left_xy),
                ]
            )
        if right_contact:
            parts.extend(
                [
                    self.foot_weight * right_bottoms,
                    self.flat_weight * right_mat[:2, 2],
                    self.xy_weight * (right_pos[:2] - right_xy),
                ]
            )
        parts.append(self.ref_weight * self.ref_scale * (x - q_ref[self.qpos_indices]))
        parts.append(
            self.knee_weight
            * np.array(
                [
                    max(0.0, self.knee_clearance - left_knee),
                    max(0.0, self.knee_clearance - right_knee),
                ],
                dtype=np.float64,
            )
        )
        if prev_x is not None and self.smooth_weight > 0:
            parts.append(self.smooth_weight * self.ref_scale * (x - prev_x))
        return np.concatenate(parts)

    def solve_frame(
        self,
        q_ref: np.ndarray,
        *,
        prev_x: np.ndarray | None,
        left_xy_target: np.ndarray | None,
        right_xy_target: np.ndarray | None,
        left_contact: bool,
        right_contact: bool,
        max_iters: int,
    ) -> np.ndarray:
        self.forward(q_ref)
        left_xy = self.data.xpos[self.left_foot_body, :2].copy() if left_xy_target is None else left_xy_target
        right_xy = self.data.xpos[self.right_foot_body, :2].copy() if right_xy_target is None else right_xy_target

        x = q_ref[self.qpos_indices].astype(np.float64).copy()
        if prev_x is not None:
            x = 0.5 * x + 0.5 * prev_x
        x = np.clip(x, self.lower, self.upper)

        for _ in range(max_iters):
            residual = self.residual(x, q_ref, left_xy, right_xy, left_contact, right_contact, prev_x)
            jac = np.empty((len(residual), len(x)), dtype=np.float64)
            for col in range(len(x)):
                step = self.fd_eps
                xp = x.copy()
                xm = x.copy()
                xp[col] = min(xp[col] + step, self.upper[col])
                xm[col] = max(xm[col] - step, self.lower[col])
                if xp[col] == xm[col]:
                    jac[:, col] = 0.0
                    continue
                rp = self.residual(xp, q_ref, left_xy, right_xy, left_contact, right_contact, prev_x)
                rm = self.residual(xm, q_ref, left_xy, right_xy, left_contact, right_contact, prev_x)
                jac[:, col] = (rp - rm) / (xp[col] - xm[col])

            lhs = jac.T @ jac + self.damping * np.eye(len(x), dtype=np.float64)
            rhs = -(jac.T @ residual)
            try:
                dx = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            max_step = float(np.max(np.abs(dx)))
            if max_step > 0.12:
                dx *= 0.12 / max_step
            x_next = np.clip(x + dx, self.lower, self.upper)
            if float(np.linalg.norm(x_next - x)) < 1e-5:
                x = x_next
                break
            x = x_next

        q_out = q_ref.copy()
        q_out[self.qpos_indices] = x
        return q_out

    def metrics(self, qpos: np.ndarray) -> dict[str, float]:
        left_foot_min = []
        left_foot_max = []
        right_foot_min = []
        right_foot_max = []
        left_knee = []
        right_knee = []
        left_flat = []
        right_flat = []
        left_xy = []
        right_xy = []
        for row in qpos:
            self.forward(row)
            lfb = self.foot_bottoms(self.left_foot_geoms)
            rfb = self.foot_bottoms(self.right_foot_geoms)
            left_foot_min.append(float(np.min(lfb)))
            left_foot_max.append(float(np.max(lfb)))
            right_foot_min.append(float(np.min(rfb)))
            right_foot_max.append(float(np.max(rfb)))
            left_knee.append(self.knee_bottom(self.left_knee_geoms))
            right_knee.append(self.knee_bottom(self.right_knee_geoms))
            left_flat.append(float(self.data.xmat[self.left_foot_body].reshape(3, 3)[2, 2]))
            right_flat.append(float(self.data.xmat[self.right_foot_body].reshape(3, 3)[2, 2]))
            left_xy.append(self.data.xpos[self.left_foot_body, :2].copy())
            right_xy.append(self.data.xpos[self.right_foot_body, :2].copy())
        left_xy_arr = np.asarray(left_xy)
        right_xy_arr = np.asarray(right_xy)
        if len(left_xy_arr) > 1:
            left_xy_max_step = float(np.max(np.linalg.norm(np.diff(left_xy_arr, axis=0), axis=1)))
            right_xy_max_step = float(np.max(np.linalg.norm(np.diff(right_xy_arr, axis=0), axis=1)))
        else:
            left_xy_max_step = 0.0
            right_xy_max_step = 0.0
        return {
            "left_foot_min_z": float(np.min(left_foot_min)),
            "left_foot_max_z": float(np.max(left_foot_max)),
            "right_foot_min_z": float(np.min(right_foot_min)),
            "right_foot_max_z": float(np.max(right_foot_max)),
            "left_knee_min_z": float(np.min(left_knee)),
            "right_knee_min_z": float(np.min(right_knee)),
            "left_foot_zaxis_z_min": float(np.min(left_flat)),
            "right_foot_zaxis_z_min": float(np.min(right_flat)),
            "left_foot_xy_max_drift": float(np.max(np.linalg.norm(left_xy_arr - left_xy_arr[0], axis=1))),
            "right_foot_xy_max_drift": float(np.max(np.linalg.norm(right_xy_arr - right_xy_arr[0], axis=1))),
            "left_foot_xy_max_step": left_xy_max_step,
            "right_foot_xy_max_step": right_xy_max_step,
        }

    def foot_xy(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.forward(qpos)
        return self.data.xpos[self.left_foot_body, :2].copy(), self.data.xpos[self.right_foot_body, :2].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input ELF3 NPZ containing qpos_elf3")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--elf3-xml", type=Path, default=DEFAULT_ELF3_XML)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--knee-clearance", type=float, default=0.08)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--foot-weight", type=float, default=28.0)
    parser.add_argument("--flat-weight", type=float, default=2.5)
    parser.add_argument("--xy-weight", type=float, default=1.0)
    parser.add_argument("--knee-weight", type=float, default=14.0)
    parser.add_argument("--ref-weight", type=float, default=0.08)
    parser.add_argument("--smooth-weight", type=float, default=0.12)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    parser.add_argument(
        "--hip-roll-ref-multiplier",
        type=float,
        default=1.0,
        help="Extra reference penalty multiplier for l/r hip roll, reducing leg spread.",
    )
    parser.add_argument(
        "--hip-yaw-ref-multiplier",
        type=float,
        default=1.0,
        help="Extra reference penalty multiplier for l/r hip yaw, reducing twisted wide stance.",
    )
    parser.add_argument(
        "--ankle-roll-ref-multiplier",
        type=float,
        default=1.0,
        help="Extra reference penalty multiplier for l/r ankle roll.",
    )
    parser.add_argument(
        "--lock-root-z",
        action="store_true",
        help="Preserve the incoming root_z trajectory and solve only lower-body joints.",
    )
    parser.add_argument(
        "--lock-root-xy",
        action="store_true",
        help="Keep root_x/root_y fixed to the first frame, for stationary squat motions.",
    )
    parser.add_argument(
        "--solve-root-xy",
        action="store_true",
        help="Allow small root_x/root_y IK corrections while pinning stationary feet.",
    )
    parser.add_argument(
        "--root-xy-ref-multiplier",
        type=float,
        default=1.0,
        help="Extra reference penalty multiplier for root_x/root_y when --solve-root-xy is used.",
    )
    parser.add_argument(
        "--stationary-feet",
        action="store_true",
        help="Use the first frame's left/right foot world XY as fixed targets for the whole sequence.",
    )
    parser.add_argument(
        "--contact-aware-feet",
        action="store_true",
        help="Use foot_contacts to pin only contacting feet; non-contact feet are not forced to the ground.",
    )
    parser.add_argument(
        "--contact-gap-fill",
        type=int,
        default=2,
        help="Fill false gaps of at most N frames inside contact segments.",
    )
    parser.add_argument(
        "--min-contact-frames",
        type=int,
        default=3,
        help="Ignore contact segments shorter than this many frames.",
    )
    parser.add_argument(
        "--target-anchor-frames",
        type=int,
        default=3,
        help="Use the median foot XY over the first N frames of a contact segment as the stationary target.",
    )
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    if args.max_iters <= 0:
        raise ValueError("--max-iters must be positive")
    if args.contact_gap_fill < 0:
        raise ValueError("--contact-gap-fill must be non-negative")
    if args.min_contact_frames <= 0:
        raise ValueError("--min-contact-frames must be positive")
    if args.target_anchor_frames <= 0:
        raise ValueError("--target-anchor-frames must be positive")
    if args.lock_root_xy and args.solve_root_xy:
        raise ValueError("--lock-root-xy and --solve-root-xy cannot be used together")
    for name in (
        "root_xy_ref_multiplier",
        "hip_roll_ref_multiplier",
        "hip_yaw_ref_multiplier",
        "ankle_roll_ref_multiplier",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    model = mujoco.MjModel.from_xml_path(str(args.elf3_xml))
    data = load_npz(args.input)
    if "qpos_elf3" not in data:
        raise ValueError(f"{args.input} does not contain qpos_elf3")
    qpos = data["qpos_elf3"].astype(np.float64)
    if qpos.ndim == 3:
        qpos = qpos[0]
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Expected qpos_elf3 shape [T, {model.nq}], got {qpos.shape}")

    solver = FlatFootIK(
        model,
        foot_weight=args.foot_weight,
        flat_weight=args.flat_weight,
        xy_weight=args.xy_weight,
        knee_weight=args.knee_weight,
        ref_weight=args.ref_weight,
        smooth_weight=args.smooth_weight,
        knee_clearance=args.knee_clearance,
        damping=args.damping,
        fd_eps=args.fd_eps,
        lock_root_z=args.lock_root_z,
        solve_root_xy=args.solve_root_xy,
        root_xy_ref_multiplier=args.root_xy_ref_multiplier,
        hip_roll_ref_multiplier=args.hip_roll_ref_multiplier,
        hip_yaw_ref_multiplier=args.hip_yaw_ref_multiplier,
        ankle_roll_ref_multiplier=args.ankle_roll_ref_multiplier,
    )

    before = solver.metrics(qpos)
    if args.lock_root_xy:
        qpos[:, 0] = qpos[0, 0]
        qpos[:, 1] = qpos[0, 1]

    left_contacts = np.ones(qpos.shape[0], dtype=bool)
    right_contacts = np.ones(qpos.shape[0], dtype=bool)
    left_xy_targets = np.full((qpos.shape[0], 2), np.nan, dtype=np.float64)
    right_xy_targets = np.full((qpos.shape[0], 2), np.nan, dtype=np.float64)

    left_xy_by_frame = np.zeros((qpos.shape[0], 2), dtype=np.float64)
    right_xy_by_frame = np.zeros((qpos.shape[0], 2), dtype=np.float64)
    for frame_idx, row in enumerate(qpos):
        left_xy_by_frame[frame_idx], right_xy_by_frame[frame_idx] = solver.foot_xy(row)

    if args.stationary_feet:
        left_xy_targets[:] = left_xy_by_frame[0]
        right_xy_targets[:] = right_xy_by_frame[0]
    elif args.contact_aware_feet:
        if "foot_contacts" not in data:
            raise ValueError(f"{args.input} does not contain foot_contacts required by --contact-aware-feet")
        left_contacts, right_contacts = contacts_by_side(data["foot_contacts"])
        if len(left_contacts) != len(qpos):
            raise ValueError(f"foot_contacts length {len(left_contacts)} does not match qpos length {len(qpos)}")
        left_contacts = fill_short_contact_gaps(left_contacts, args.contact_gap_fill)
        right_contacts = fill_short_contact_gaps(right_contacts, args.contact_gap_fill)
        left_xy_targets = xy_targets_from_contact_segments(
            left_xy_by_frame,
            left_contacts,
            min_contact_frames=args.min_contact_frames,
            anchor_frames=args.target_anchor_frames,
        )
        right_xy_targets = xy_targets_from_contact_segments(
            right_xy_by_frame,
            right_contacts,
            min_contact_frames=args.min_contact_frames,
            anchor_frames=args.target_anchor_frames,
        )
    else:
        left_xy_targets = left_xy_by_frame
        right_xy_targets = right_xy_by_frame

    repaired = np.empty_like(qpos, dtype=np.float32)
    prev_x: np.ndarray | None = None
    for frame_idx, row in enumerate(qpos):
        left_xy_target = None if np.isnan(left_xy_targets[frame_idx]).any() else left_xy_targets[frame_idx]
        right_xy_target = None if np.isnan(right_xy_targets[frame_idx]).any() else right_xy_targets[frame_idx]
        repaired_row = solver.solve_frame(
            row,
            prev_x=prev_x,
            left_xy_target=left_xy_target,
            right_xy_target=right_xy_target,
            left_contact=bool(left_contacts[frame_idx]),
            right_contact=bool(right_contacts[frame_idx]),
            max_iters=args.max_iters,
        )
        repaired[frame_idx] = repaired_row.astype(np.float32)
        prev_x = repaired_row[solver.qpos_indices].copy()
        if (frame_idx + 1) % 30 == 0 or frame_idx == len(qpos) - 1:
            print(f"optimized {frame_idx + 1}/{len(qpos)} frames", flush=True)

    after = solver.metrics(repaired)
    data["qpos_elf3"] = repaired
    data["qpos_columns"] = np.array(ELF3_QPOS_COLUMNS)
    data["qpos_elf3_columns"] = np.array(ELF3_QPOS_COLUMNS)
    data.update(
        mjlab_compatible_fields(
            model,
            repaired,
            fps=args.fps,
            joint_names=ELF3_JOINT_NAMES,
            body_names=ELF3_BODY_NAMES,
        )
    )
    data["flat_foot_constraint"] = json.dumps(
        {
            "method": "per-frame finite-difference damped least-squares IK",
            "qpos_indices": solver.qpos_indices.tolist(),
            "lock_root_z": args.lock_root_z,
            "lock_root_xy": args.lock_root_xy,
            "solve_root_xy": args.solve_root_xy,
            "root_xy_ref_multiplier": args.root_xy_ref_multiplier,
            "stationary_feet": args.stationary_feet,
            "contact_aware_feet": args.contact_aware_feet,
            "contact_gap_fill": args.contact_gap_fill,
            "min_contact_frames": args.min_contact_frames,
            "target_anchor_frames": args.target_anchor_frames,
            "left_contact_frames": int(np.sum(left_contacts)),
            "right_contact_frames": int(np.sum(right_contacts)),
            "knee_clearance": args.knee_clearance,
            "hip_roll_ref_multiplier": args.hip_roll_ref_multiplier,
            "hip_yaw_ref_multiplier": args.hip_yaw_ref_multiplier,
            "ankle_roll_ref_multiplier": args.ankle_roll_ref_multiplier,
            "before": before,
            "after": after,
        },
        ensure_ascii=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(args.output, data)
    print(f"saved {args.output}: qpos_elf3 {repaired.shape}")
    print("before:", json.dumps(before, sort_keys=True))
    print("after: ", json.dumps(after, sort_keys=True))

    if not args.no_csv:
        csv_path = args.output.with_suffix(".csv")
        np.savetxt(csv_path, repaired, delimiter=",")
        print(f"saved {csv_path}: columns={len(ELF3_QPOS_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
