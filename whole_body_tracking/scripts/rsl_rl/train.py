# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
# 顶部添加：屏蔽 Isaac Lab 弃用警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The function 'quat_rotate_inverse' will be deprecated")

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")

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
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# python scripts/rsl_rl/train.py --task=Tracking-Climb-Flat-G1-v0 --registry_name /root/../workspace/whole_body_tracking/Datasets/Beyond/forward_box_climbing_1_holosoma_scene_interaction_rootz_q05_fps50_isaaclab.npz --headless --logger tensorboard
# python scripts/rsl_rl/play.py --task=Tracking-Climb-Flat-G1-v0 --num_envs=2 --motion_file /root/../workspace/whole_body_tracking/Datasets/Beyond/forward_box_climbing_1_holosoma_scene_interaction_rootz_q05_fps50_isaaclab.npz

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
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
            f"[INFO][rank {dist.global_rank}/{dist.world_size}] Distributed teacher training: "
            f"device=cuda:{dist.local_rank}, seed={agent_cfg.seed}, "
            f"local_envs={dist.local_num_envs}, global_envs={dist.global_num_envs}"
        )

    # load the motion file from the wandb registry
    # registry_name = args_cli.registry_name
    # if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
    #     registry_name += ":latest"
    # import pathlib

    # import wandb

    # api = wandb.Api()
    # artifact = api.artifact(registry_name)
    # env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    env_cfg.commands.motion.motion_file = args_cli.registry_name
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if dist.is_main_process:
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    record_video = args_cli.video and dist.is_main_process
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if record_video else None)
    # wrap for video recording
    if record_video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=args_cli.registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if dist.is_main_process:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    if dist.is_main_process:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
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

    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
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

""""
# python scripts/rsl_rl/train.py --task=Tracking-Climb-Flat-G1-v0  --seed 42 --registry_name  /root/../workspace/PHP/whole_body_tracking/Datasets/Beyond/su/forward_box_climbing_1_omniretarget_scene_interaction_rootz_q05_stand25_trans50_fps50_isaaclab.npz --headless --logger tensorboard

# python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --registry_name /root/../workspace/PHP/whole_body_tracking/Datasets/cmu/85_01_stageii_gmr.npz --headless --logger tensorboard   --run_name 85_01_wo_state

python scripts/rsl_rl/train.py \
  --task=Tracking-Climb-Flat-Omni-v0 \
  --registry_name Datasets/omni_dataset/overbox_1m_isaaclab_fps50.npz \
  --num_envs 1 \
  --max_iterations 100 \
  --logger tensorboard

python scripts/rsl_rl/train.py \
  --task=Tracking-Climb-Flat-Omni-v0 \
  --seed 42 \
  --registry_name Datasets/omni_dataset/0729_overbox_1m_2_isaaclab_fps50.npz \
  --num_envs 4096 \
  --headless \
  --logger tensorboard

python scripts/rsl_rl/train.py \
  --task=Tracking-Climb-Flat-Omni-Wo-State-Estimation-v0 \
  --seed 42 \
  --registry_name Datasets/omni_dataset/0729_overbox_1m_2_isaaclab_fps50.npz \
  --num_envs 4096 \
  --headless \
  --logger tensorboard
"""
