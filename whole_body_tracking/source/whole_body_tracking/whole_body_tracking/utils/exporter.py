# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(
    env: ManagerBasedRLEnv,
    run_path: str,
    path: str,
    filename="policy.onnx",
    normalizer_embedded: bool = False,
    normalizer_eps: float | None = None,
) -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
        "normalizer_embedded": normalizer_embedded,
    }
    if normalizer_eps is not None:
        metadata["normalizer_eps"] = normalizer_eps

    depth_sensor = env.scene.sensors.get("depth_camera")
    if depth_sensor is not None:
        depth_cfg = depth_sensor.cfg
        # Isaac Lab ray-caster cameras store the image shape in pattern_cfg and
        # their clipping range directly on the sensor config.  Rendered Camera /
        # TiledCamera (used by the Isaac Lab 2.2 student) instead store height /
        # width on the camera config and the optical clipping range in spawn.
        pattern_cfg = getattr(depth_cfg, "pattern_cfg", None)
        depth_height = getattr(depth_cfg, "height", None)
        depth_width = getattr(depth_cfg, "width", None)
        if pattern_cfg is not None:
            depth_height = getattr(pattern_cfg, "height", depth_height)
            depth_width = getattr(pattern_cfg, "width", depth_width)

        clipping_range = getattr(getattr(depth_cfg, "spawn", None), "clipping_range", None)
        camera_min_distance = getattr(depth_cfg, "min_distance", None)
        camera_max_distance = getattr(depth_cfg, "max_distance", None)
        if clipping_range is not None:
            camera_min_distance = clipping_range[0]
            camera_max_distance = clipping_range[1]

        # Deployment must reproduce the policy preprocessing range, which can
        # be narrower than the camera's optical clipping range (0.3 m vs 0.01 m
        # for the current student).  Prefer the observation term parameters.
        depth_term_cfg = getattr(env.observation_manager.cfg.policy, "depth_image", None)
        depth_params = getattr(depth_term_cfg, "params", {}) or {}
        preprocessing_min = depth_params.get("min_distance", camera_min_distance)
        preprocessing_max = depth_params.get("max_distance", camera_max_distance)
        metadata.update(
            {
                "depth_height": depth_height,
                "depth_width": depth_width,
                "depth_min_distance": preprocessing_min,
                "depth_max_distance": preprocessing_max,
                "depth_camera_clipping_min": camera_min_distance,
                "depth_camera_clipping_max": camera_max_distance,
                "depth_data_type": "distance_to_image_plane",
                "depth_preprocessing": "clip_then_linear_to_minus0.5_plus0.5",
            }
        )

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
