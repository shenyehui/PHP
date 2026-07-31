# Copyright (c) 2025, Whole Body Tracking Contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train a student policy via teacher-student distillation with RSL-RL.

Usage examples:
  # Single teacher
  python scripts/rsl_rl/train_student.py \\
      --task=Tracking-Student-Encoder-Climb-Flat-G1-v0 \\
      --registry_name /path/to/motion.npz \\
      --teacher_logdirs /path/to/teacher_checkpoint_dir \\
      --headless

  # Multiple teachers
  python scripts/rsl_rl/train_student.py \\
      --task=Tracking-Student-Encoder-Climb-Flat-G1-v0 \\
      --registry_name /path/to/motion.npz \\
      --teacher_logdirs /path/to/teacher_A,/path/to/teacher_B \\
      --teacher_mode multi_random \\
      --headless

  # Resume from checkpoint
  python scripts/rsl_rl/train_student.py \\
      --task=Tracking-Student-Encoder-Climb-Flat-G1-v0 \\
      --registry_name /path/to/motion.npz \\
      --teacher_logdirs /path/to/teacher \\
      --resume /path/to/student_checkpoint_dir \\
      --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import warnings

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The function 'quat_rotate_inverse' will be deprecated")

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a student policy via distillation.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Tracking-Student-Encoder-Climb-Flat-G1-v0",
    help="Name of the task (the encoder task is the PHP paper-aligned default).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, required=True, help="Path to motion .npz file for training.")

# Student-specific arguments
parser.add_argument(
    "--teacher_logdirs",
    type=str,
    required=True,
    help=(
        "Comma-separated list of teacher checkpoint directories. "
        "Each directory should contain model_*.pt files. "
        "Example: '/path/t1,/path/t2'"
    ),
)
parser.add_argument("--command_vx", type=float, default=1.0, help="Student planar velocity command x [m/s].")
parser.add_argument("--command_vy", type=float, default=0.0, help="Student planar velocity command y [m/s].")
parser.add_argument(
    "--teacher_mode",
    type=str,
    default="single",
    choices=["single", "multi_random", "multi_best"],
    help="How to use multiple teachers. 'single' uses only the first. "
    "'multi_random' randomly picks one per step. "
    "'multi_best' picks the closest teacher action.",
)
parser.add_argument(
    "--distill_only_iterations",
    type=int,
    default=None,
        help="Number of initial iterations with distillation only (no PPO). "
        "The PHP-aligned default is 0.",
)
parser.add_argument(
    "--teacher_act_prob",
    type=float,
    default=0.0,
    help="DAgger: probability of using teacher action instead of student during rollout.",
)

# append optional torchrun arguments
cli_args.add_distributed_args(parser)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# Student policy needs depth camera, so always enable cameras
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _audit_training_depth(env, min_policy_distance: float = 0.3) -> None:
    """Fail fast when the rendered student depth is constant or self-occluded."""
    base_env = env.unwrapped
    depth_sensor = base_env.scene.sensors.get("depth_camera")
    if depth_sensor is None:
        raise RuntimeError("Student task has no 'depth_camera' sensor.")

    depth_sensor.reset()
    for _ in range(3):
        base_env.sim.render()
    raw_depth = depth_sensor.data.output["distance_to_image_plane"]
    sample_count = min(int(raw_depth.shape[0]), 32)
    sampled_depth = raw_depth[:sample_count]
    finite_mask = torch.isfinite(sampled_depth)
    finite_depth = sampled_depth[finite_mask]
    if finite_depth.numel() == 0:
        raise RuntimeError("Depth camera health check failed: no finite pixels.")

    depth_min = finite_depth.min().item()
    depth_max = finite_depth.max().item()
    depth_mean = finite_depth.mean().item()
    per_env_scene_fraction = (
        ((sampled_depth > min_policy_distance) & finite_mask).flatten(1).float().mean(dim=1)
    )
    worst_scene_fraction = per_env_scene_fraction.min().item()
    print(
        f"[INFO] Depth camera startup audit ({sample_count} envs): "
        f"range=[{depth_min:.3f}, {depth_max:.3f}] m, mean={depth_mean:.3f} m, "
        f"worst_pixels_beyond_{min_policy_distance:.1f}m={worst_scene_fraction:.3f}"
    )
    if depth_max - depth_min < 1.0e-5 or worst_scene_fraction < 0.01:
        raise RuntimeError(
            "Depth camera health check failed: the image is constant or almost entirely "
            f"closer than the policy minimum ({min_policy_distance:.1f} m). Refusing to "
            "train a student with an invalid/self-occluded depth input."
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train a student policy with teacher-student distillation."""
    # override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Preserve the original single-process settings, or assign one CUDA device,
    # a distinct seed and a share of the environments to each torchrun rank.
    dist = cli_args.configure_distributed_training(
        env_cfg, agent_cfg, args_cli, app_launcher
    )
    if dist.enabled:
        print(
            f"[INFO][rank {dist.global_rank}/{dist.world_size}] Distributed student training: "
            f"device=cuda:{dist.local_rank}, seed={agent_cfg.seed}, "
            f"local_envs={dist.local_num_envs}, global_envs={dist.global_num_envs}"
        )

    # ---- Apply student-specific CLI overrides ----
    agent_cfg.algorithm.teacher_mode = args_cli.teacher_mode
    agent_cfg.algorithm.teacher_logdirs = [
        d.strip() for d in args_cli.teacher_logdirs.split(",") if d.strip()
    ]
    if args_cli.distill_only_iterations is not None:
        agent_cfg.algorithm.distill_only_iterations = args_cli.distill_only_iterations
    agent_cfg.algorithm.teacher_act_prob = args_cli.teacher_act_prob

    # ---- Load resume checkpoint if provided ----
    # --load_run is handled by cli_args.update_rsl_rl_cfg automatically
    if agent_cfg.load_run is not None:
        agent_cfg.resume = True

    # ---- Print config summary ----
    if dist.is_main_process:
        print("[INFO] Student training configuration:")
        print(f"  Task: {args_cli.task}")
        print(f"  Teacher mode: {agent_cfg.algorithm.teacher_mode}")
        print(f"  Teacher dirs: {agent_cfg.algorithm.teacher_logdirs}")
        print(f"  Local/global envs: {dist.local_num_envs}/{dist.global_num_envs}")
        print(f"  Max iterations: {agent_cfg.max_iterations}")
        print(f"  Motion registry: {args_cli.registry_name}")

    # ---- Create environment ----
    # Set the motion file from registry_name before creating the env
    env_cfg.commands.motion.motion_file = args_cli.registry_name
    if hasattr(env_cfg.observations.policy, "velocity_command"):
        env_cfg.observations.policy.velocity_command.params["command"] = (
            args_cli.command_vx,
            args_cli.command_vy,
        )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Catch camera calibration/rendering failures before a multi-hour run.
    _audit_training_depth(
        env,
        min_policy_distance=env_cfg.observations.policy.depth_image.params["min_distance"],
    )

    motion_command = env.unwrapped.command_manager.get_term("motion")
    robot_data = env.unwrapped.scene["robot"].data
    default_joint_pos = getattr(
        robot_data,
        "default_joint_pos_nominal",
        robot_data.default_joint_pos[0],
    )
    phase0_joint_delta = torch.linalg.vector_norm(
        motion_command.motion.joint_pos[0] - default_joint_pos
    ).item()
    phase0_joint_speed = torch.linalg.vector_norm(motion_command.motion.joint_vel[0]).item()
    phase0_root_speed = torch.linalg.vector_norm(
        motion_command.motion.body_lin_vel_w[0, 0]
    ).item()
    if dist.is_main_process:
        print(
            "[INFO] Motion phase-0 startup audit: "
            f"joint_delta_from_default={phase0_joint_delta:.3f} rad, "
            f"joint_speed={phase0_joint_speed:.3f} rad/s, "
            f"root_speed={phase0_root_speed:.3f} m/s"
        )
        if phase0_joint_delta > 0.5 or phase0_joint_speed > 0.5 or phase0_root_speed > 0.2:
            print(
                "[WARN] Reference phase 0 is not a nominal static standing state. "
                "For deployment, prepend a standing/zero-velocity segment and a smooth "
                "transition, or use a stand controller before policy handover."
            )
    action_scale = env.unwrapped.action_manager.get_term("joint_pos")._scale[0]
    if dist.is_main_process and not torch.allclose(action_scale, torch.ones_like(action_scale)):
        print(
            "[WARN] This teacher/student pair uses the repository's per-joint action scale, "
            "not PHP's all-ones expert action scale. Keep it unchanged for compatibility "
            "with existing teachers; retrain both teacher and student to reproduce that "
            "paper setting exactly."
        )

    # ---- Create runner ----
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    # Determine log directory
    log_root = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Convert config to dict for runner
    train_cfg = {
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval": agent_cfg.save_interval,
        "empirical_normalization": agent_cfg.empirical_normalization,
        "policy": {
            "class_name": agent_cfg.policy.class_name,
            **agent_cfg.policy.__dict__,
        },
        "algorithm": {
            "class_name": agent_cfg.algorithm.class_name,
            **agent_cfg.algorithm.__dict__,
        },
        "logger": "tensorboard",
        "load_run": agent_cfg.load_run,
        "resume": agent_cfg.resume,
    }

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(
            log_root,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )
        if dist.is_main_process:
            print(f"[INFO] Resuming student checkpoint from: {resume_path}")
        runner.load(resume_path)

    # ---- Load teacher models ----
    # The runner's algorithm has a load_teachers method; we need to call it
    # after environment creation (so we know observation dimensions).
    obs, extras = env.get_observations()
    teacher_obs_dim = extras["observations"]["teacher"].shape[1]
    student_act_dim = env.num_actions

    runner.alg.load_teachers(teacher_obs_dim, student_act_dim)

    # Each teacher owns and applies the normalizer stored in its checkpoint.
    # Sharing the first teacher's statistics across an ensemble silently
    # corrupts the other teachers' action labels.

    # ---- Save config ----
    if dist.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        if dist.enabled:
            dump_yaml(
                os.path.join(log_dir, "params", "distributed.yaml"),
                {
                    "world_size": dist.world_size,
                    "num_envs_mode": args_cli.distributed_num_envs_mode,
                    "local_num_envs": dist.local_num_envs,
                    "global_num_envs": dist.global_num_envs,
                    "rank_seed_rule": "base_seed + global_rank",
                },
            )

    # ---- Train ----
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    finally:
        # No distributed collectives occur after runner.learn() returns.
        env.close()
        cli_args.shutdown_distributed()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # Also covers failures after the runner initialized NCCL but before learn().
        cli_args.shutdown_distributed()
        # close sim app even when one torchrun rank fails
        simulation_app.close()

"""
python scripts/rsl_rl/train_student.py \
    --task Tracking-Student-Encoder-Climb-Flat-G1-v0 \
    --registry_name /root/../workspace/PHP/whole_body_tracking/Datasets/Beyond/forward_box_climbing_1_holosoma_scene_interaction_rootz_q05_fps50_isaaclab.npz \
    --teacher_logdirs /root/../workspace/PHP/whole_body_tracking/logs/rsl_rl/g1_flat/2026-06-30_01-58-01 \
    --teacher_mode single \
    --command_vx 1.0 \
    --command_vy 0.0 \
    --num_envs 1048 \
    --headless

python scripts/rsl_rl/train_student.py \
    --task Tracking-Student-Encoder-Climb-Flat-G1-v0 \
    --registry_name /root/../workspace/PHP/whole_body_tracking/Datasets/Beyond/su/forward_box_climbing_1_omniretarget_scene_interaction_rootz_q05_stand25_trans50_fps50_isaaclab.npz \
    --teacher_logdirs /root/../workspace/PHP/whole_body_tracking/logs/rsl_rl/g1_flat/2026-07-17_15-30-51 \
    --teacher_mode single \
    --command_vx 1.0 \
    --command_vy 0.0 \
    --num_envs 1048 \
    --headless


python scripts/rsl_rl/train_student.py \
  --task Tracking-Student-Encoder-Climb-Flat-Omni-v0 \
  --registry_name Datasets/omni_dataset/overbox_1m_isaaclab_fps50.npz \
  --teacher_logdirs logs/rsl_rl/omni_climb_teacher/2026-07-24_02-54-51 \
  --teacher_mode single \
  --command_vx 1.0 \
  --command_vy 0.0 \
  --num_envs 1024 \
  --max_iterations 50000 \
  --headless

  python scripts/rsl_rl/train_student.py \
  --task Tracking-Student-Encoder-Climb-Flat-Omni-D455-v0 \
  --registry_name Datasets/omni_dataset/0729_overbox_1m_2_isaaclab_fps50.npz \
  --teacher_logdirs logs/rsl_rl/omni_climb_teacher/2026-07-29_02-42-00 \
  --teacher_mode single \
  --command_vx 1.0 \
  --command_vy 0.0 \
  --num_envs 2048 \
  --max_iterations 50000 \
  --headless

"""
