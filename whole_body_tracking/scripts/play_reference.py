"""Replay a tracking reference directly in a registered Isaac Lab task.

This is a geometry/calibration tool, not policy inference: the articulation is
written to each reference frame without physics stepping.  It is therefore
appropriate for checking whether a motion, robot and obstacle share the same
world coordinates before teacher training.
"""

from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay a task's reference motion without a policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Tracking-Climb-Flat-Omni-v0-PLAY",
    help="Registered tracking task used to construct the robot and scene.",
)
parser.add_argument("--motion_file", type=str, required=True, help="Converted MotionLoader NPZ file.")
parser.add_argument("--start_frame", type=int, default=0, help="First reference frame to display.")
parser.add_argument("--frame_step", type=int, default=1, help="Reference frames advanced per render.")
parser.add_argument("--no_loop", action="store_true", help="Exit after reaching the final frame.")
parser.add_argument(
    "--follow_camera",
    action="store_true",
    help="Move the viewer with the robot; the default fixed view is better for box alignment.",
)
parser.add_argument(
    "--depth_view",
    "--depth_camera",
    dest="depth_view",
    action="store_true",
    help=(
        "Attach the Omni student's tiled depth camera and show its metric input "
        "in a live Isaac Sim window."
    ),
)
parser.add_argument(
    "--depth_profile",
    choices=("d455", "legacy"),
    default="d455",
    help="Camera/preprocessing profile used by --depth_view (default: d455).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.headless:
    raise ValueError("Reference visualization requires a non-headless Isaac Sim window.")
if args_cli.frame_step < 1:
    raise ValueError("--frame_step must be at least 1")
if args_cli.depth_view:
    # RTX sensors are disabled unless camera rendering is requested before Kit
    # starts.  This flag is required even though the main viewer is graphical.
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


class _ReferenceDepthViewer:
    """Small live metric-depth window implemented with Isaac Sim's own UI."""

    def __init__(
        self,
        height: int,
        width: int,
        min_distance: float,
        max_distance: float,
        invalid_outside_range: bool,
    ):
        import omni.ui as ui

        self.height = int(height)
        self.width = int(width)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.invalid_outside_range = bool(invalid_outside_range)
        self.frame = 0
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window("Reference Depth Camera", width=640, height=460, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=4):
                self.stats = ui.Label("Depth: waiting for the first rendered frame", height=24)
                with ui.Frame(height=420):
                    ui.ImageWithProvider(self.provider)

    def show(self, depth: torch.Tensor):
        raw_depth_m = depth[0].detach().squeeze(-1).cpu().numpy()
        if self.invalid_outside_range:
            valid = (
                np.isfinite(raw_depth_m)
                & (raw_depth_m >= self.min_distance)
                & (raw_depth_m <= self.max_distance)
            )
            depth_m = np.where(valid, raw_depth_m, 0.0)
            gray = np.zeros_like(depth_m, dtype=np.uint8)
            gray[valid] = np.clip(
                (self.max_distance - depth_m[valid])
                / (self.max_distance - self.min_distance)
                * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8)
            if np.any(valid):
                valid_depth = depth_m[valid]
                stats = (
                    f"valid min {valid_depth.min():.3f}, "
                    f"max {valid_depth.max():.3f}, "
                    f"mean {valid_depth.mean():.3f}, "
                    f"invalid {(~valid).mean() * 100.0:.1f}%"
                )
            else:
                stats = "no valid pixels"
        else:
            depth_m = np.nan_to_num(
                raw_depth_m,
                nan=self.max_distance,
                posinf=self.max_distance,
                neginf=self.min_distance,
            )
            depth_m = np.clip(depth_m, self.min_distance, self.max_distance)
            gray = np.clip(
                (self.max_distance - depth_m)
                / (self.max_distance - self.min_distance)
                * 255.0,
                0.0,
                255.0,
            ).astype(np.uint8)
            stats = (
                f"min {depth_m.min():.3f}, max {depth_m.max():.3f}, "
                f"mean {depth_m.mean():.3f}"
            )
        # Use the exact fixed metric range of the student preprocessing.  Near
        # pixels are white and far pixels are black; no per-frame normalization.
        rgba = np.dstack((gray, gray, gray, np.full_like(gray, 255)))
        self.provider.set_bytes_data(rgba.flatten().data, [self.width, self.height])
        self.stats.text = "Policy depth [m] — near: white, far/invalid: black | " + stats
        if self.frame % 100 == 0:
            print(f"[DEPTH] frame={self.frame} {stats}")
        self.frame += 1

    def close(self):
        self.window.visible = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, _agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.sampling_strategy = "zero"
    if args_cli.depth_view:
        # Use the exact camera class and TiledCamera backend used by Omni
        # student training, without changing the reference-play task itself.
        if args_cli.depth_profile == "d455":
            from whole_body_tracking.tasks.tracking.config.omni.student_env_cfg import (
                OmniD455StudentDepthCameraCfg,
            )

            env_cfg.scene.depth_camera = OmniD455StudentDepthCameraCfg()
        else:
            from whole_body_tracking.tasks.tracking.config.omni.student_env_cfg import (
                OmniStudentDepthCameraCfg,
            )

            env_cfg.scene.depth_camera = OmniStudentDepthCameraCfg()

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    scene = base_env.scene
    robot = scene["robot"]
    motion_command = base_env.command_manager.get_term("motion")
    motion = motion_command.motion

    if motion_command.cfg.body_names[0] != robot.body_names[0]:
        raise RuntimeError(
            "The first tracking body must be the articulation root: {} != {}".format(
                motion_command.cfg.body_names[0], robot.body_names[0]
            )
        )
    if motion.joint_pos.shape[1] != robot.num_joints:
        raise RuntimeError(
            "Motion/robot joint dimension mismatch: {} != {}".format(
                motion.joint_pos.shape[1], robot.num_joints
            )
        )

    fps = float(np.asarray(motion.fps).reshape(-1)[0])
    if fps <= 0.0:
        raise ValueError("Motion FPS must be positive")
    frame_dt = args_cli.frame_step / fps
    frame = min(args_cli.start_frame, motion.time_step_total - 1)
    if frame < 0:
        raise ValueError("--start_frame must be non-negative")

    print("[INFO] Reference frames: {}, FPS: {:.3f}".format(motion.time_step_total, fps))
    print("[INFO] Box size=(1, 1, 1), centre=(0.5, 0.2, 0.5), rotation_wxyz=(1, 0, 0, 0)")
    print("[INFO] The Isaac cuboid spans z=[0, 1] and its bottom is flush with the ground.")
    root_positions = motion.body_pos_w[:, 0]
    root_min = root_positions.amin(dim=0).detach().cpu().tolist()
    root_max = root_positions.amax(dim=0).detach().cpu().tolist()
    print("[INFO] Reference root xyz bounds: min={}, max={}".format(root_min, root_max))

    depth_sensor = None
    depth_viewer = None
    if args_cli.depth_view:
        depth_sensor = scene.sensors.get("depth_camera")
        if depth_sensor is None:
            raise RuntimeError(
                "--depth_view requested, but the scene did not create a 'depth_camera' sensor."
            )
        depth_sensor.reset()
        if args_cli.depth_profile == "d455":
            depth_min_distance = 0.5
            depth_max_distance = 5.0
            invalid_outside_range = True
        else:
            depth_min_distance = 0.3
            depth_max_distance = 3.0
            invalid_outside_range = False
        depth_viewer = _ReferenceDepthViewer(
            height=env_cfg.scene.depth_camera.height,
            width=env_cfg.scene.depth_camera.width,
            min_distance=depth_min_distance,
            max_distance=depth_max_distance,
            invalid_outside_range=invalid_outside_range,
        )
        print(
            f"[INFO] Using the Omni {args_cli.depth_profile} TiledCamera: "
            f"{env_cfg.scene.depth_camera.height}x{env_cfg.scene.depth_camera.width}, "
            f"policy range [{depth_min_distance}, {depth_max_distance}] m."
        )

    # A fixed overview makes an incorrect scene/reference transform obvious.
    base_env.sim.set_camera_view(eye=(2.8, 2.8, 2.2), target=(0.0, 0.0, 0.5))
    next_frame_time = time.perf_counter()
    first_depth_frame = True

    while simulation_app.is_running():
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = motion.body_pos_w[frame, 0]
        root_state[:, :3] += scene.env_origins
        root_state[:, 3:7] = motion.body_quat_w[frame, 0]
        root_state[:, 7:10] = motion.body_lin_vel_w[frame, 0]
        root_state[:, 10:13] = motion.body_ang_vel_w[frame, 0]

        joint_pos = motion.joint_pos[frame].unsqueeze(0)
        joint_vel = motion.joint_vel[frame].unsqueeze(0)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        # Isaac Lab 2.2 tiled render products may contain an initialization
        # frame of zeros.  Warm up RTX after writing the requested first pose.
        render_count = 3 if depth_sensor is not None and first_depth_frame else 1
        for _ in range(render_count):
            base_env.sim.render()
        first_depth_frame = False
        # This tool does not advance physics, so account for the displayed
        # reference-frame interval explicitly.  Otherwise a 30 Hz camera would
        # update at only 0.005 s per displayed frame instead of the replay rate.
        scene.update(frame_dt)
        if depth_viewer is not None:
            depth_viewer.show(depth_sensor.data.output["distance_to_image_plane"])

        if args_cli.follow_camera:
            root = root_state[0, :3].detach().cpu().numpy()
            base_env.sim.set_camera_view(
                eye=(root + np.asarray((-2.2, 2.2, 1.2))).tolist(),
                target=(root + np.asarray((0.0, 0.0, 0.3))).tolist(),
            )

        frame += args_cli.frame_step
        if frame >= motion.time_step_total:
            if args_cli.no_loop:
                break
            frame = 0

        next_frame_time += frame_dt
        remaining = next_frame_time - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            next_frame_time = time.perf_counter()

    if depth_viewer is not None:
        depth_viewer.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

"""

python scripts/play_reference.py \
    --task=Tracking-Climb-Flat-Omni-v0-PLAY \
    --motion_file Datasets/omni_dataset/0729_overbox_1m_2_isaaclab_fps50.npz \
    --depth_view

"""
