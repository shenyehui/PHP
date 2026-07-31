"""Play a depth-encoder student checkpoint in Isaac Lab.

Usage:
python scripts/rsl_rl/play_student.py \
    --motion_file /root/../workspace/PHP/whole_body_tracking/Datasets/Beyond/su/forward_box_climbing_1_omniretarget_scene_interaction_rootz_q05_stand25_trans50_fps50_isaaclab.npz \
    --load_run 20260718_140853 \
    --checkpoint model_19999.pt \
    --num_envs 1 \
    --depth_view

python scripts/rsl_rl/play_student.py \
    --task Tracking-Student-Encoder-Climb-Flat-Omni-v0 \
    --motion_file /root/../workspace/PHP/whole_body_tracking/Datasets/omni_dataset/overbox_1m_isaaclab_fps50.npz \
    --load_run 20260725_020142 \
    --checkpoint model_19999.pt \
    --num_envs 1 \
    --depth_view

"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Play a distilled depth student checkpoint.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--motion_file", type=str, required=True, help="Path to motion .npz file")
parser.add_argument(
    "--task",
    type=str,
    default="Tracking-Student-Encoder-Climb-Flat-G1-v0",
    help="Student task to play.",
)
parser.add_argument("--export_onnx", dest="export_onnx", action="store_true", default=True)
parser.add_argument("--no_export_onnx", dest="export_onnx", action="store_false")
parser.add_argument("--depth_view", action="store_true", help="Show the policy depth input synchronously.")
parser.add_argument(
    "--depth_ray_debug",
    action="store_true",
    help="Deprecated on Isaac Lab 2.2: tiled depth cameras have no ray-hit markers.",
)
parser.add_argument(
    "--random_start",
    action="store_true",
    help="Use uniform random reference phases instead of phase 0.",
)
parser.add_argument("--command_vx", type=float, default=1.0)
parser.add_argument("--command_vy", type=float, default=0.0)
parser.add_argument(
    "--camera_backend",
    type=str,
    choices=("tiled", "standard"),
    default="tiled",
    help=(
        "Depth camera backend. 'standard' is a single-environment diagnostic "
        "that bypasses Isaac Lab 2.2 TiledCamera rendering."
    ),
)

cli_args.add_rsl_rl_args(parser)  # adds --load_run, --checkpoint, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# The policy always consumes rendered depth, including headless play without
# video or the OpenCV depth window.
args_cli.enable_cameras = True
if args_cli.depth_ray_debug:
    print("[WARN] --depth_ray_debug is unavailable with Isaac Lab 2.2 TiledCamera; use --depth_view instead.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata

TASK = args_cli.task


def _export_student_onnx(policy, normalizer, sample_obs, export_dir, filename="policy.onnx"):
    """Export the full student inference pipeline to ONNX:

    observation → normalizer → depth encoder + proprio → actor MLP → actions
    """
    import torch.nn as nn

    device = next(policy.parameters()).device

    # Build a traceable wrapper that calls the student's full forward pipeline
    class StudentExportWrapper(nn.Module):
        def __init__(self, policy, normalizer):
            super().__init__()
            self.policy = policy
            self.normalizer = normalizer

        def forward(self, obs):
            if self.normalizer is not None:
                obs = self.normalizer(obs)
            return self.policy.act_inference(obs)

    wrapper = StudentExportWrapper(policy, normalizer)
    wrapper.eval()
    wrapper.to(device)

    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, filename)

    torch.onnx.export(
        wrapper,
        sample_obs.to(device),
        out_path,
        export_params=True,
        opset_version=11,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    print(f"[INFO] ONNX exported to {out_path}")


class _DepthViewer:
    """Display policy depth in Isaac Sim UI, with OpenCV/file fallbacks."""

    def __init__(
        self,
        height: int,
        width: int,
        min_distance: float,
        max_distance: float,
        preprocessing: str,
        fallback_path: str,
        fallback_interval: int = 5,
        prefer_isaac_ui: bool = True,
    ):
        self.height = height
        self.width = width
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.preprocessing = preprocessing
        if preprocessing not in {
            "clip_then_linear_to_minus0.5_plus0.5",
            "range_mask_then_divide_by_max",
        }:
            raise ValueError(f"Unsupported depth preprocessing: {preprocessing!r}")
        self.window_name = "student policy depth input"
        self.fallback_path = fallback_path
        self.fallback_interval = max(int(fallback_interval), 1)
        self.frame = 0
        self.has_window = False
        self.isaac_ui_window = None
        self.isaac_ui_provider = None
        self.isaac_ui_stats = None
        self.cv2 = None

        # Use Kit's own UI in graphical Isaac Sim.  This is independent of the
        # opencv-python wheel and therefore works in containers that only have
        # opencv-python-headless installed.
        if prefer_isaac_ui:
            try:
                import omni.ui as ui

                self.isaac_ui_provider = ui.ByteImageProvider()
                self.isaac_ui_window = ui.Window(
                    "Student Policy Depth",
                    width=640,
                    height=460,
                    visible=True,
                )
                with self.isaac_ui_window.frame:
                    with ui.VStack(spacing=4):
                        self.isaac_ui_stats = ui.Label(
                            "Policy depth: waiting for first frame",
                            height=24,
                        )
                        with ui.Frame(height=420):
                            ui.ImageWithProvider(self.isaac_ui_provider)
                print("[INFO] Live depth is shown in the Isaac Sim 'Student Policy Depth' window.")
                return
            except Exception as exc:
                self.isaac_ui_window = None
                self.isaac_ui_provider = None
                self.isaac_ui_stats = None
                print(f"[WARN] Isaac Sim depth window could not be created: {exc}")

        # Secondary fallback for environments with a functional OpenCV GUI.
        try:
            import cv2

            self.cv2 = cv2
        except ImportError:
            self.cv2 = None
        if self.cv2 is not None and self._highgui_display_available():
            try:
                self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
                self.has_window = True
            except self.cv2.error:
                self.has_window = False
        if not self.has_window and self.isaac_ui_provider is None:
            # opencv-python-headless deliberately has no GTK/Qt HighGUI
            # implementation.  A Qt build can also abort the entire Python
            # process (rather than raise cv2.error) if DISPLAY points to a
            # missing local X socket, so do not call namedWindow in that case.
            print(
                "[WARN] OpenCV HighGUI/display is unavailable; depth preview will be "
                f"updated at: {self.fallback_path}"
            )

    def _highgui_display_available(self) -> bool:
        if self.cv2 is None:
            return False
        build_info = self.cv2.getBuildInformation()
        gui_line = next((line for line in build_info.splitlines() if line.strip().startswith("GUI:")), "")
        if "NONE" in gui_line.upper():
            return False

        # Local X11 displays use /tmp/.X11-unix/X<N>.  Checking the socket
        # avoids a fatal Qt abort when DISPLAY is set but not mounted into the
        # container.  Non-local/X-forwarded displays are left to HighGUI.
        display = os.environ.get("DISPLAY", "")
        if display.startswith(":"):
            display_number = display[1:].split(".", 1)[0]
            return display_number.isdigit() and os.path.exists(f"/tmp/.X11-unix/X{display_number}")
        if display:
            return True

        wayland_display = os.environ.get("WAYLAND_DISPLAY", "")
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "")
        return bool(wayland_display and runtime_dir and os.path.exists(os.path.join(runtime_dir, wayland_display)))

    def show(self, obs: torch.Tensor):
        import numpy as np

        depth_normalized = obs[0, : self.height * self.width].reshape(
            self.height, self.width
        )
        depth_normalized = depth_normalized.detach().cpu().numpy()
        if self.preprocessing == "range_mask_then_divide_by_max":
            # Zero is the D455 contract's invalid-pixel sentinel.  Valid values
            # occupy [min/max, 1], so the mask is unambiguous.
            valid = depth_normalized > 0.0
            depth_m = depth_normalized * self.max_distance
            gray = np.zeros_like(depth_normalized, dtype=np.uint8)
            gray[valid] = np.clip(
                (self.max_distance - depth_m[valid])
                / (self.max_distance - self.min_distance)
                * 255.0,
                0,
                255,
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
                stats = "no valid depth pixels (invalid 100.0%)"
        else:
            valid = np.ones_like(depth_normalized, dtype=bool)
            depth_m = (depth_normalized + 0.5) * (
                self.max_distance - self.min_distance
            ) + self.min_distance
            gray = np.clip(
                (self.max_distance - depth_m)
                / (self.max_distance - self.min_distance)
                * 255.0,
                0,
                255,
            ).astype(np.uint8)
            stats = (
                f"min {depth_m.min():.3f}, max {depth_m.max():.3f}, "
                f"mean {depth_m.mean():.3f}"
            )
        # Fixed metric scale: near pixels are bright and far pixels are dark.
        # Fixed scaling (instead of per-frame normalization) preserves actual
        # distance changes and makes a constant/invalid frame immediately clear.

        if self.isaac_ui_provider is not None:
            rgba = np.dstack((gray, gray, gray, np.full_like(gray, 255)))
            self.isaac_ui_provider.set_bytes_data(rgba.flatten().data, [self.width, self.height])
            if self.isaac_ui_stats is not None:
                self.isaac_ui_stats.text = (
                    "Policy depth [m] — near: white, far/invalid: black | "
                    + stats
                )
            self.frame += 1
            return

        if self.has_window:
            preview = self.cv2.resize(
                gray,
                (self.width * 6, self.height * 6),
                interpolation=self.cv2.INTER_NEAREST,
            )
            try:
                self.cv2.imshow(self.window_name, preview)
                self.cv2.waitKey(1)
            except self.cv2.error:
                # A build may expose namedWindow but still fail at imshow when
                # DISPLAY/Wayland is unavailable.  Switch to file fallback.
                self.has_window = False
                print(
                    "[WARN] OpenCV cannot display a window; depth preview will be "
                    f"updated at: {self.fallback_path}"
                )
        if not self.has_window and self.frame % self.fallback_interval == 0:
            if self.cv2 is not None:
                self.cv2.imwrite(self.fallback_path, gray)
        self.frame += 1

    def close(self):
        if self.isaac_ui_window is not None:
            self.isaac_ui_window.visible = False
        if self.has_window:
            try:
                self.cv2.destroyWindow(self.window_name)
            except self.cv2.error:
                pass


def _refresh_depth_observation(env):
    """Render a valid TiledCamera frame before export and the first action.

    Isaac Lab 2.2 initializes TiledCamera output buffers with zeros.  The first
    observation may therefore precede the first RTX render and map entirely to
    -0.5.  Rendering first and invalidating the sensor cache prevents the play
    policy and parity files from consuming that initialization buffer.
    """
    base_env = env.unwrapped
    depth_sensor = base_env.scene.sensors.get("depth_camera")
    if depth_sensor is None:
        return env.get_observations()

    # Render a few times to allow the RTX/replicator texture to become valid,
    # without advancing physics or changing the reference phase.
    depth_sensor.reset()
    for _ in range(3):
        base_env.sim.render()
    raw_depth = depth_sensor.data.output["distance_to_image_plane"]

    finite_depth = raw_depth[torch.isfinite(raw_depth)]
    if finite_depth.numel() > 0:
        depth_min = finite_depth.min().item()
        depth_max = finite_depth.max().item()
        print(
            f"[DEBUG] rendered depth meters range=[{depth_min:.3f}, "
            f"{depth_max:.3f}], mean={finite_depth.mean().item():.3f}"
        )
        if depth_max - depth_min < 1.0e-5:
            print(
                "[WARN] Rendered depth is constant after camera warm-up. "
                "Do not use the parity files for sim2sim until the camera is checked."
            )
    else:
        print("[WARN] Rendered depth contains no finite pixels.")

    # Recompute all observation groups so the policy tensor contains the newly
    # rendered depth rather than the observation cached by env initialization.
    return env.get_observations()


def _use_standard_depth_camera(env_cfg):
    """Replace TiledCameraCfg with CameraCfg while preserving calibration.

    A standard rendered Camera is too expensive for thousands of training
    environments, but is the most direct diagnostic for a one-environment play
    run when the tiled render product returns a constant zero depth buffer.
    """
    from isaaclab.sensors import CameraCfg

    tiled_cfg = env_cfg.scene.depth_camera
    env_cfg.scene.depth_camera = CameraCfg(
        prim_path=tiled_cfg.prim_path,
        update_period=tiled_cfg.update_period,
        history_length=tiled_cfg.history_length,
        debug_vis=False,
        offset=CameraCfg.OffsetCfg(
            pos=tiled_cfg.offset.pos,
            rot=tiled_cfg.offset.rot,
            convention=tiled_cfg.offset.convention,
        ),
        spawn=tiled_cfg.spawn,
        depth_clipping_behavior=tiled_cfg.depth_clipping_behavior,
        data_types=list(tiled_cfg.data_types),
        width=tiled_cfg.width,
        height=tiled_cfg.height,
        update_latest_camera_pose=True,
    )


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    print(f"[INFO] Loading student from run: {args_cli.load_run}")
    print(f"[INFO] Motion file: {args_cli.motion_file}")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.sampling_strategy = "uniform" if args_cli.random_start else "zero"
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.depth_image.params["apply_noise"] = False
    env_cfg.observations.policy.velocity_command.params["command"] = (
        args_cli.command_vx,
        args_cli.command_vy,
    )
    depth_params = env_cfg.observations.policy.depth_image.params
    depth_min_distance = float(depth_params.get("min_distance", 0.3))
    depth_max_distance = float(depth_params.get("max_distance", 3.0))
    depth_preprocessing = depth_params.get(
        "preprocessing",
        "clip_then_linear_to_minus0.5_plus0.5",
    )
    env_cfg.events.randomize_camera_rays = None
    if args_cli.camera_backend == "standard":
        if args_cli.num_envs != 1:
            raise ValueError("--camera_backend standard requires --num_envs 1")
        _use_standard_depth_camera(env_cfg)
        print("[INFO] Using standard Camera depth backend for single-environment diagnosis.")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = args_cli.checkpoint or "model_.*.pt"
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, checkpoint)
    print(f"[INFO] Checkpoint: {resume_path}")

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_dir, exist_ok=True)

    obs, _ = _refresh_depth_observation(env)

    if args_cli.export_onnx:
        print(f"[INFO] Exporting ONNX to: {export_dir}")
        try:
            _export_student_onnx(
                ppo_runner.alg.policy,
                ppo_runner.obs_normalizer,
                obs[:1],  # batch size 1
                export_dir,
                filename="policy.onnx",
            )
            print("[INFO] ONNX export done!")
            # Attach PD metadata (joint stiffness, damping, etc.) for deployment
            attach_onnx_metadata(
                env.unwrapped,
                resume_path,
                export_dir,
                filename="policy.onnx",
                normalizer_embedded=True,
                normalizer_eps=getattr(ppo_runner.obs_normalizer, "eps", None),
            )
            print("[INFO] PD metadata attached")
        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}")

    # Save raw and policy-input observations separately for sim2sim parity.
    import numpy as np
    obs0 = obs[0].cpu().numpy()
    policy_obs0 = ppo_runner.obs_normalizer(obs[:1])[0].cpu().numpy()
    with torch.inference_mode():
        action0 = policy(obs[:1])[0].cpu().numpy()
    np.save(os.path.join(export_dir, "isaac_obs_raw.npy"), obs0)
    np.save(os.path.join(export_dir, "isaac_obs_policy_input.npy"), policy_obs0)
    np.save(os.path.join(export_dir, "isaac_action.npy"), action0)
    depth_height = ppo_runner.alg.policy._depth_height
    depth_width = ppo_runner.alg.policy._depth_width
    depth_size = depth_height * depth_width
    print(f"[DEBUG] Saved Isaac parity observations, shape={obs0.shape}")
    print(
        f"[DEBUG] depth normalized range=[{obs0[:depth_size].min():.3f}, "
        f"{obs0[:depth_size].max():.3f}], mean={obs0[:depth_size].mean():.3f}"
    )

    # Also save root state: position + quaternion + joint positions
    root_state = env.unwrapped.scene["robot"].data.root_state_w[0].cpu().numpy()
    joint_pos = env.unwrapped.scene["robot"].data.joint_pos[0].cpu().numpy()
    np.save(os.path.join(export_dir, "isaac_root_state.npy"), root_state)
    np.save(os.path.join(export_dir, "isaac_joint_pos.npy"), joint_pos)
    print(f"[DEBUG] root_state (pos+quat): {root_state[:7]}")
    print(f"[DEBUG] joint_pos[:5]: {joint_pos[:5]}, range=[{joint_pos.min():.3f}, {joint_pos.max():.3f}]")

    timestep = 0
    depth_viewer = None
    if args_cli.depth_view:
        depth_viewer = _DepthViewer(
            depth_height,
            depth_width,
            depth_min_distance,
            depth_max_distance,
            depth_preprocessing,
            fallback_path=os.path.join(export_dir, "depth_preview.png"),
            prefer_isaac_ui=not args_cli.headless,
        )
        depth_viewer.show(obs)

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if depth_viewer is not None:
            depth_viewer.show(obs)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    if depth_viewer is not None:
        depth_viewer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
