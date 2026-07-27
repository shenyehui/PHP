from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


@dataclass(frozen=True)
class DistributedContext:
    """Resolved torchrun rank information and environment allocation."""

    enabled: bool
    world_size: int
    global_rank: int
    local_rank: int
    local_num_envs: int
    global_num_envs: int

    @property
    def is_main_process(self) -> bool:
        return self.global_rank == 0


def add_distributed_args(parser: argparse.ArgumentParser) -> None:
    """Add opt-in distributed-training arguments shared by teacher and student."""
    arg_group = parser.add_argument_group(
        "distributed training",
        description="Optional torchrun-based multi-GPU training.",
    )
    arg_group.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help=(
            "Enable synchronous multi-GPU training. Launch this script with "
            "'python -m torch.distributed.run' (torchrun). If omitted, the "
            "existing single-process training path is unchanged."
        ),
    )
    arg_group.add_argument(
        "--distributed_num_envs_mode",
        "--distributed-num-envs-mode",
        dest="distributed_num_envs_mode",
        choices=("total", "per_gpu"),
        default="total",
        help=(
            "Meaning of --num_envs in distributed mode. 'total' (default) "
            "splits the requested global total evenly over all ranks; 'per_gpu' "
            "creates that many environments on every GPU."
        ),
    )


def configure_distributed_training(
    env_cfg,
    agent_cfg,
    args_cli: argparse.Namespace,
    app_launcher,
) -> DistributedContext:
    """Apply rank-local device, seed and environment-count settings.

    The non-distributed branch deliberately mirrors the original training
    scripts. Distributed mode is opt-in and requires a torchrun world.
    """

    distributed_requested = bool(getattr(args_cli, "distributed", False))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}.")
    if not 0 <= global_rank < world_size:
        raise ValueError(
            f"RANK must be in [0, WORLD_SIZE), got RANK={global_rank}, WORLD_SIZE={world_size}."
        )
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}.")

    if distributed_requested and world_size == 1:
        raise RuntimeError(
            "--distributed was requested, but WORLD_SIZE=1. Launch with torchrun, for example: "
            "python -m torch.distributed.run --standalone --nnodes=1 "
            "--nproc_per_node=4 scripts/rsl_rl/train.py ... --distributed"
        )
    if not distributed_requested and world_size > 1:
        raise RuntimeError(
            f"Detected a torchrun world with WORLD_SIZE={world_size}, but --distributed was not set. "
            "Add --distributed, or launch the script normally without torchrun."
        )

    launcher_local_rank = int(getattr(app_launcher, "local_rank", local_rank))
    launcher_global_rank = int(getattr(app_launcher, "global_rank", global_rank))
    if distributed_requested and (
        launcher_local_rank != local_rank or launcher_global_rank != global_rank
    ):
        raise RuntimeError(
            "Isaac Lab AppLauncher rank does not match torchrun: "
            f"launcher=({launcher_global_rank}, {launcher_local_rank}), "
            f"torchrun=({global_rank}, {local_rank})."
        )

    requested_num_envs = (
        int(args_cli.num_envs)
        if args_cli.num_envs is not None
        else int(env_cfg.scene.num_envs)
    )
    if requested_num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {requested_num_envs}.")

    if distributed_requested:
        num_envs_mode = args_cli.distributed_num_envs_mode
        if num_envs_mode == "total":
            if requested_num_envs % world_size != 0:
                raise ValueError(
                    "With --distributed_num_envs_mode=total, --num_envs must be "
                    f"divisible by WORLD_SIZE. Got {requested_num_envs} environments "
                    f"and WORLD_SIZE={world_size}."
                )
            local_num_envs = requested_num_envs // world_size
            global_num_envs = requested_num_envs
        else:
            local_num_envs = requested_num_envs
            global_num_envs = requested_num_envs * world_size

        device = f"cuda:{local_rank}"
        base_seed = int(agent_cfg.seed if agent_cfg.seed is not None else 0)
        rank_seed = base_seed + global_rank

        env_cfg.scene.num_envs = local_num_envs
        env_cfg.sim.device = device
        env_cfg.seed = rank_seed
        agent_cfg.device = device
        agent_cfg.seed = rank_seed
    else:
        local_num_envs = requested_num_envs
        global_num_envs = requested_num_envs
        env_cfg.scene.num_envs = local_num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = (
            args_cli.device if args_cli.device is not None else env_cfg.sim.device
        )

    local_rollout_size = local_num_envs * int(agent_cfg.num_steps_per_env)
    num_mini_batches = int(agent_cfg.algorithm.num_mini_batches)
    if local_rollout_size < num_mini_batches:
        raise ValueError(
            "The rank-local rollout is smaller than the number of mini-batches: "
            f"{local_num_envs} envs * {agent_cfg.num_steps_per_env} steps = "
            f"{local_rollout_size}, num_mini_batches={num_mini_batches}. "
            "Increase --num_envs or reduce num_mini_batches."
        )

    return DistributedContext(
        enabled=distributed_requested,
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        local_num_envs=local_num_envs,
        global_num_envs=global_num_envs,
    )


def shutdown_distributed() -> None:
    """Release the process group after the last synchronized update."""

    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--wandb_path", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
