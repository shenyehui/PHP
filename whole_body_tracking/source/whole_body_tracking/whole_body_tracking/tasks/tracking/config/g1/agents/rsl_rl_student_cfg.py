# Copyright (c) 2025, Whole Body Tracking Contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration for the PHP encoder student training route."""

import os

from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Encoder policy
# ---------------------------------------------------------------------------

@configclass
class EncoderPolicyCfg:
    """Depth encoder and MLP architecture for the student."""

    class_name: str = "StudentEncoderPolicy"

    init_noise_std: float = 0.01
    actor_hidden_dims: list = [2048, 1024, 512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "elu"

    # Conv2D encoder config
    encoder_config: dict = {
        "input_channels": 1,
        "input_height": 58,
        "input_width": 87,
        "channels": [16, 32, 32],
        "kernel_sizes": [5, 3, 3],
        "strides": [2, 2, 1],
        "paddings": [2, 1, 1],
        "hidden_sizes": [],
        "output_size": 32,
        "activation": "elu",
        "use_maxpool": False,
        "global_average_pool": True,
    }

    # Observation component layout: tells the policy where depth sits in the flat obs
    # These will be auto-populated from the environment's observation config.
    obs_components: dict | None = None


# ---------------------------------------------------------------------------
# DistillPPO algorithm
# ---------------------------------------------------------------------------

@configclass
class DistillPPOAlgorithmCfg:
    """DistillPPO algorithm configuration."""

    class_name: str = "DistillPPO"

    # --- Teacher settings ---------------------------------------------------
    teacher_mode: str = "single"  # "single" | "multi_random" | "multi_best"
    teacher_logdirs: list = []  # list of paths to teacher checkpoint dirs
    teacher_policy_cfg: dict = {
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    }
    teacher_act_prob: float = 0.0  # DAgger: probability of using teacher action

    # --- Distillation loss --------------------------------------------------
    distill_loss_coef: float = 1.0
    distill_loss_type: str = "mse"  # "mse" | "huber" | "smooth_l1"
    distill_only_iterations: int = 0
    distill_ppo_weight_schedule: bool = True
    distill_ppo_schedule_iterations: int = 10000
    distill_loss_schedule_scale: float = 10.0
    distill_weight_min: float = 0.1
    disable_teacher_act_when_ppo: bool = True
    teacher_act_disable_ppo_weight_threshold: float = 0.1

    # --- PPO settings -------------------------------------------------------
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.001  # PHP paper: 0.001 (critical for preventing mode collapse)
    num_learning_epochs: int = 2  # PHP paper: 2
    num_mini_batches: int = 96  # PHP paper: 96 (smaller updates = more stable)
    learning_rate: float = 3.0e-4
    schedule: str = "adaptive"  # PHP paper: adaptive KL-based early stopping
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = True
    weight_decay: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@configclass
class G1StudentEncoderPPORunnerCfg:
    """Runner for the single supported encoder student."""

    num_steps_per_env: int = 24
    max_iterations: int = 20000
    save_interval: int = 500
    experiment_name: str = "g1_student_encoder"
    empirical_normalization: bool = False

    seed: int = 42
    device: str = "cuda:0"
    load_run: str | None = None
    load_checkpoint: str | None = None
    resume: bool = False
    logger: str = "tensorboard"
    log_project_name: str | None = None
    wandb_project: str | None = None
    neptune_project: str | None = None

    policy: EncoderPolicyCfg = EncoderPolicyCfg()
    algorithm: DistillPPOAlgorithmCfg = DistillPPOAlgorithmCfg()

    def __post_init__(self):
        self.resume = self.load_run is not None
        self.run_name = (
            f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}"
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else ""
        )
