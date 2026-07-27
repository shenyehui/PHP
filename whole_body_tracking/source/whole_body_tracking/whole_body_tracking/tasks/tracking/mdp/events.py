from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
):
    """Randomize a rendered camera pose through the Isaac Lab 2.2 public API.

    The nominal poses are cached on first use so that repeated startup event
    application cannot accumulate calibration error. Position and orientation
    perturbations are expressed in the nominal camera's ``world`` frame.
    """
    sensor = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=env.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if not hasattr(sensor, "_student_nominal_camera_pos_w"):
        sensor._student_nominal_camera_pos_w = sensor.data.pos_w.detach().clone()
        sensor._student_nominal_camera_quat_w = sensor.data.quat_w_world.detach().clone()

    nominal_pos = sensor._student_nominal_camera_pos_w[env_ids]
    nominal_quat = sensor._student_nominal_camera_quat_w[env_ids]
    range_list = [
        offset_pose_ranges.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ]
    ranges = torch.tensor(range_list, device=nominal_pos.device, dtype=nominal_pos.dtype)
    samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), 6),
        device=nominal_pos.device,
    )
    position_offsets = math_utils.quat_apply(nominal_quat, samples[:, :3])
    rotation_offsets = math_utils.quat_from_euler_xyz(
        samples[:, 3], samples[:, 4], samples[:, 5]
    )
    randomized_pos = nominal_pos + position_offsets
    randomized_quat = math_utils.quat_mul(nominal_quat, rotation_offsets)
    sensor.set_world_poses(
        positions=randomized_pos,
        orientations=randomized_quat,
        env_ids=env_ids,
        convention="world",
    )
