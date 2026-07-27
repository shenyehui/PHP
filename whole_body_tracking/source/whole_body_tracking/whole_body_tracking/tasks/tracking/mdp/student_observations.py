# Copyright (c) 2025, Whole Body Tracking Contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the student policy.

The student policy does NOT receive reference motion as input.
Instead it relies on:
  - depth_image  : simulated depth camera on torso_link (flattened)
  - proprioception : projected_gravity, base_ang_vel, base_lin_vel,
                     joint_pos, joint_vel, last_action
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def student_depth_image_flat(
    env: ManagerBasedEnv,
    sensor_cfg,
    min_distance: float = 0.3,
    max_distance: float = 3.0,
    apply_noise: bool = True,
    depth_offset_range: float = 0.03,
    gaussian_noise_std: float = 0.03,
) -> torch.Tensor:
    """Return the exact, deployment-facing flattened depth tensor.

    The preprocessing contract is deliberately independent of Isaac Lab's
    version-dependent ``mdp.image(normalize=...)`` behavior:

    1. replace non-finite values and clip to the calibrated range;
    2. optionally add PHP's per-frame depth offset and i.i.d. Gaussian noise;
    3. map ``[min_distance, max_distance]`` linearly to ``[-0.5, 0.5]``.
    """
    sensor = env.scene[sensor_cfg.name]
    img = sensor.data.output["distance_to_image_plane"].clone()
    img = torch.nan_to_num(
        img,
        nan=max_distance,
        posinf=max_distance,
        neginf=min_distance,
    ).clamp(min_distance, max_distance)

    if apply_noise:
        offset_shape = (img.shape[0],) + (1,) * (img.ndim - 1)
        depth_offset = torch.empty(offset_shape, device=img.device, dtype=img.dtype).uniform_(
            -depth_offset_range, depth_offset_range
        )
        img = img + depth_offset + gaussian_noise_std * torch.randn_like(img)
        img = img.clamp(min_distance, max_distance)

    img = (img - min_distance) / (max_distance - min_distance) - 0.5
    return img.flatten(1)


def student_projected_gravity(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    from isaaclab.envs.mdp import projected_gravity as _projected_gravity
    return _projected_gravity(env, asset_cfg)


def student_base_lin_vel(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    from isaaclab.envs.mdp import base_lin_vel as _base_lin_vel
    return _base_lin_vel(env, asset_cfg)


def student_base_ang_vel(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    from isaaclab.envs.mdp import base_ang_vel as _base_ang_vel
    return _base_ang_vel(env, asset_cfg)


def student_joint_pos_rel(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    from isaaclab.envs.mdp import joint_pos_rel as _joint_pos_rel
    return _joint_pos_rel(env, asset_cfg)


def student_joint_vel_rel(env: ManagerBasedEnv, asset_cfg) -> torch.Tensor:
    from isaaclab.envs.mdp import joint_vel_rel as _joint_vel_rel
    return _joint_vel_rel(env, asset_cfg)


def student_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    from isaaclab.envs.mdp import last_action as _last_action
    return _last_action(env)


def student_velocity_command(
    env: ManagerBasedEnv,
    command: tuple[float, float] = (1.0, 0.0),
) -> torch.Tensor:
    """Return the externally supplied planar velocity command for every env."""
    return torch.tensor(command, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)


def student_action_label_mask(
    env: ManagerBasedEnv,
    asset_cfg,
    command_name: str = "motion",
    base_pos_threshold: float = 0.5,
    ee_pos_threshold: float | None = 0.5,
    anchor_ori_threshold: float = 0.8,
    ee_body_names: list[str] | None = None,
    body_pos_checks: list[dict] | None = None,
) -> torch.Tensor:
    """Compute a mask that is 1.0 when the student is close enough to the reference
    for distillation to be meaningful, and 0.0 otherwise.

    Mirrors instinctlab's ``student_action_label_mask``:
    - Base position must be within ``base_pos_threshold`` (z-only check)
    - End-effector positions must be within their configured thresholds
      (z-only checks).

    ``ee_pos_threshold``/``ee_body_names`` retain the original single-group
    interface.  ``body_pos_checks`` supports teachers whose feet and wrists
    use different termination thresholds, for example::

        [
            {"threshold": 0.35, "body_names": ["left_foot", "right_foot"]},
            {"threshold": 0.50, "body_names": ["left_wrist", "right_wrist"]},
        ]

    The label mask should describe the teacher's training support, not the
    student's deliberately looser episode termination region.  Outside that
    support the teacher action is extrapolation, so PPO may still act but the
    distillation label is gated off.
    """
    from whole_body_tracking.tasks.tracking.mdp.terminations import (
        bad_anchor_ori,
        bad_anchor_pos_z_only,
        bad_motion_body_pos_z_only,
    )
    base_ood = bad_anchor_pos_z_only(env, command_name, base_pos_threshold)
    ori_ood = bad_anchor_ori(env, asset_cfg, command_name, anchor_ori_threshold)
    ee_ood = torch.zeros_like(base_ood)
    if ee_pos_threshold is not None:
        ee_ood |= bad_motion_body_pos_z_only(
            env,
            command_name,
            ee_pos_threshold,
            body_names=ee_body_names,
        )
    if body_pos_checks is not None:
        for check in body_pos_checks:
            ee_ood |= bad_motion_body_pos_z_only(
                env,
                command_name,
                float(check["threshold"]),
                body_names=check.get("body_names"),
            )
    # Valid = NOT out-of-distribution
    return (~(base_ood | ori_ood | ee_ood)).unsqueeze(-1).float()
