"""Gym registrations for Omni teacher and depth-student tasks."""

import gymnasium as gym

from . import agents, flat_env_cfg, student_env_cfg


gym.register(
    id="Tracking-Climb-Flat-Omni-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniClimbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniClimbPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Climb-Flat-Omni-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniClimbPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OmniClimbPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Climb-Flat-Omni-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniClimbWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "OmniClimbWoStateEstimationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Climb-Flat-Omni-Wo-State-Estimation-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.OmniClimbWoStateEstimationPlayEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "OmniClimbWoStateEstimationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Student-Encoder-Climb-Flat-Omni-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": student_env_cfg.OmniStudentEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_student_cfg:OmniStudentEncoderPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Student-Encoder-Climb-Flat-Omni-D455-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": student_env_cfg.OmniD455StudentEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_student_cfg:"
            "OmniD455StudentEncoderPPORunnerCfg"
        ),
    },
)
