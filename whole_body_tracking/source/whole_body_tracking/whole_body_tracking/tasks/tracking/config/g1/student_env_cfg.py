# Copyright (c) 2025, Whole Body Tracking Contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Student environment configuration for teacher-student distillation.

The student policy observes only depth images + proprioception (no reference motion).
The teacher / critic still receives full privileged observations for distillation targets.
"""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as tracking_mdp
import whole_body_tracking.tasks.tracking.mdp.student_observations as student_obs
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    MySceneCfg,
    TrackingEnvCfg,
)

# ---------------------------------------------------------------------------
# Depth camera helper
# ---------------------------------------------------------------------------

@configclass
class StudentDepthCameraCfg(TiledCameraCfg):
    """Isaac Lab 2.2 tiled depth camera mounted on the robot torso.

    Isaac Lab 2.2's ray caster supports only one static mesh, so it cannot see
    both the ground and the per-environment climbing box. TiledCamera is the
    2.2-compatible batched renderer and observes the complete scene.
    """
    prim_path = "{ENV_REGEX_NS}/Robot/torso_link/depth_camera"
    offset = TiledCameraCfg.OffsetCfg(
        # Exact d435_joint calibration from this project's
        # g1_29dof_mode_15_spherehand.urdf.  The previous approximate offset
        # came from a different G1 mesh and placed the optical center about
        # 1.3 cm inside this robot's head shell, producing a constant 1.5 cm
        # self-hit over the entire depth image.
        pos=(0.0576235, 0.01753, 0.42987),
        # URDF d435_joint rpy=(0, 0.8307767239493009, 0), i.e. 47.6 deg down.
        rot=(0.9149596678498247, 0.0, 0.40354529635239006, 0.0),
        convention="world",
    )
    spawn = sim_utils.PinholeCameraCfg(
        focal_length=1.0,
        horizontal_aperture=2 * math.tan(math.radians(87) / 2),
        vertical_aperture=2 * math.tan(math.radians(58) / 2),
        # Match instinctlab's 5 cm minimum ray distance so residual robot
        # surfaces immediately around the lens do not occlude the scene.
        clipping_range=(0.05, 3.0),
    )
    height = 58
    width = 87
    data_types = ["distance_to_image_plane"]
    update_period = 1 / 30
    debug_vis = False
    depth_clipping_behavior = "max"


# ---------------------------------------------------------------------------
# Observations – student (policy) and teacher / critic (privileged)
# ---------------------------------------------------------------------------

@configclass
class StudentObservationsCfg:
    """Observation specifications for the student distillation environment.

    ``policy`` – student input (depth + proprio, NO reference motion)
    ``teacher`` – teacher input (full reference motion + proprio)
    ``critic`` – critic input (privileged, same as teacher or richer)
    """

    @configclass
    class PolicyCfg(ObsGroupCfg):
        """Student policy observations: depth image + proprioception only."""

        depth_image = ObsTermCfg(
            func=student_obs.student_depth_image_flat,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "min_distance": 0.3,
                "max_distance": 3.0,
                "apply_noise": True,
                "depth_offset_range": 0.03,
                "gaussian_noise_std": 0.03,
            },
        )

        projected_gravity = ObsTermCfg(
            func=student_obs.student_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTermCfg(
            func=student_obs.student_base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_pos = ObsTermCfg(
            func=student_obs.student_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTermCfg(
            func=student_obs.student_joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTermCfg(
            func=student_obs.student_last_action,
        )
        velocity_command = ObsTermCfg(
            func=student_obs.student_velocity_command,
            params={"command": (1.0, 0.0)},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True  # all terms are flat 1D now


    @configclass
    class TeacherCfg(ObsGroupCfg):
        """Teacher observations: full reference motion + proprioception.

        This is identical to what Tracking-Climb-Flat-G1-v0 teachers received
        during their training.
        """

        command = ObsTermCfg(func=tracking_mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTermCfg(
            func=tracking_mdp.motion_anchor_pos_b, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObsTermCfg(
            func=tracking_mdp.motion_anchor_ori_b, params={"command_name": "motion"}
        )
        base_lin_vel = ObsTermCfg(func=tracking_mdp.base_lin_vel)
        base_ang_vel = ObsTermCfg(func=tracking_mdp.base_ang_vel)
        joint_pos = ObsTermCfg(func=tracking_mdp.joint_pos_rel)
        joint_vel = ObsTermCfg(func=tracking_mdp.joint_vel_rel)
        actions = ObsTermCfg(func=tracking_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroupCfg):
        """Critic (privileged) observations: same as teacher for now.

        Can be extended with additional privileged information if desired.
        """

        command = ObsTermCfg(func=tracking_mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTermCfg(func=tracking_mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTermCfg(func=tracking_mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTermCfg(func=tracking_mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTermCfg(func=tracking_mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTermCfg(func=tracking_mdp.base_lin_vel)
        base_ang_vel = ObsTermCfg(func=tracking_mdp.base_ang_vel)
        joint_pos = ObsTermCfg(func=tracking_mdp.joint_pos_rel)
        joint_vel = ObsTermCfg(func=tracking_mdp.joint_vel_rel)
        actions = ObsTermCfg(func=tracking_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()
    critic: CriticCfg = CriticCfg()

    @configclass
    class ActionLabelMaskObsCfg(ObsGroupCfg):
        """Mask that gates distillation: 1.0 when student is close to reference, 0.0 otherwise.

        Matches instinctlab's ``student_action_label_mask`` pattern.
        When the student is too far from the reference motion, distillation targets
        become meaningless — this mask prevents the student from trying to imitate
        teacher actions from a completely different state.
        """
        action_label_mask = ObsTermCfg(
            func=student_obs.student_action_label_mask,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                # Match the actual teacher termination configuration.  The
                # student itself is relaxed from 0.5 to 1.0 below.
                "base_pos_threshold": 0.25,
                "ee_pos_threshold": 0.25,
                "anchor_ori_threshold": 0.8,
                "ee_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    action_label_mask: ActionLabelMaskObsCfg = ActionLabelMaskObsCfg()


# ---------------------------------------------------------------------------
# Scene – same as G1BoxClimbingEnvCfg but with additional depth camera
# ---------------------------------------------------------------------------

@configclass
class StudentSceneCfg(MySceneCfg):
    """Scene with flat terrain, box obstacle, G1 robot, and depth camera."""

    # Add a simplified depth camera for the student
    depth_camera: TiledCameraCfg = StudentDepthCameraCfg()

    def __post_init__(self):
        super().__post_init__()
        # Add the box obstacle (same as G1BoxClimbingEnvCfg)
        self.box = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ObstacleBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 0.8, 0.608),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.3664, -0.672, 0.304),
                rot=(0.9983759397857438, 0.0, 0.0, 0.056969139513712616),
            ),
        )


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

@configclass
class G1StudentEnvCfg(TrackingEnvCfg):
    """Student environment for distillation from teacher(s)."""

    # Override scene to include depth camera + box
    scene: StudentSceneCfg = StudentSceneCfg(num_envs=4096, env_spacing=8)

    # Override observations to split policy / teacher / critic
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        # Call parent to set up robot, actions, commands, etc.
        super().__post_init__()

        # Robot configuration (same as G1BoxClimbingEnvCfg)
        from whole_body_tracking.robots.g1_spherehand import G1_CYLINDER_CFG, G1_ACTION_SCALE
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        # Student training samples reference phases uniformly.  The command
        # term already writes the sampled reference state on reset; adding a
        # second reset event used to sample and write twice.
        self.commands.motion.sampling_strategy = "uniform"

        self.events.randomize_camera_rays = EventTermCfg(
            func=tracking_mdp.randomize_camera_pose,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("depth_camera"),
                "offset_pose_ranges": {
                    "x": (-0.025, 0.025),
                    "y": (-0.025, 0.025),
                    "z": (-0.025, 0.025),
                    "roll": (-math.radians(2.5), math.radians(2.5)),
                    "pitch": (-math.radians(2.5), math.radians(2.5)),
                    "yaw": (-math.radians(2.5), math.radians(2.5)),
                },
            },
        )

        # ---- Disable random pushes (对标 instinctlab: self.events.push_robot = None) ----
        # 学生策略没有 ref motion，被推后无法恢复，蒸馏目标也不包含推力补偿
        self.events.push_robot = None

        # ---- Relaxed terminations for student (no ref motion in policy obs) ----
        # 初始收紧（0.5），随训练逐步放松（1.0），对标 instinctlab
        self._student_init_termination_thresholds()

    # ------------------------------------------------------------------
    # Student-specific helper: adaptive termination thresholds
    # ------------------------------------------------------------------
    def _student_init_termination_thresholds(self):
        """Set relaxed termination thresholds and register a schedule callback.

        Students don't have access to reference motion in their policy
        observations, so tracking is inherently less precise than the
        teacher.  We start with tight thresholds and loosen them over
        the course of training.
        """
        # --- static relaxation (initial values) ---
        # PHP student threshold starts at 0.5 m and relaxes to 1.0 m.
        self.terminations.anchor_pos.params["threshold"] = 0.5
        self.terminations.ee_body_pos.params["threshold"] = 0.5

        # --- out_of_border (对标 instinctlab perceptive_env_cfg.py L646-650) ---
        self.terminations.out_of_border = DoneTermCfg(
            func=_student_terrain_out_of_bounds,
            time_out=True,
            params={"max_distance": 10.0},
        )

        # --- adaptive schedule ---
        # 对标 instinctlab: 从 0.5 逐步放松到 1.0（240k 步）
        # instinctlab 还有 base_pg_too_far（静态阈值），此处暂不添加
        from isaaclab.managers import CurriculumTermCfg as CurrTerm
        self.curriculum.termination_schedule = CurrTerm(
            func=student_termination_schedule,
            params={
                "start_threshold": 0.5,
                "end_threshold": 1.0,
                "schedule_steps": 10000 * 24,
            },
        )


def student_termination_schedule(
    env,
    env_ids,
    start_threshold: float = 0.5,
    end_threshold: float = 1.0,
    schedule_steps: int = 240000,
):
    """Module-level function so Hydra can serialise / deserialise it."""
    if len(env_ids) == 0:
        return None
    progress = min(float(env.common_step_counter) / max(float(schedule_steps), 1.0), 1.0)
    threshold = start_threshold + (end_threshold - start_threshold) * progress
    for term_name in ("anchor_pos", "ee_body_pos"):
        term_cfg = getattr(env.cfg.terminations, term_name, None)
        if term_cfg is not None:
            term_cfg.params["threshold"] = threshold
        if hasattr(env, "termination_manager") and term_name in env.termination_manager.active_terms:
            live_term_cfg = env.termination_manager.get_term_cfg(term_name)
            live_term_cfg.params["threshold"] = threshold
            env.termination_manager.set_term_cfg(term_name, live_term_cfg)
    return {"termination_threshold": threshold}


def _student_terrain_out_of_bounds(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float = 5.0,
) -> torch.Tensor:
    """Terminate if robot base drifts too far from its own environment origin.
    Uses env-relative position (not absolute world), suitable for flat terrain.
    """
    asset = env.scene[asset_cfg.name]
    # root_pos_w includes env_origins offset; subtract to get env-relative position
    root_pos_env = asset.data.root_pos_w - env.scene.env_origins
    dist_xy = torch.norm(root_pos_env[:, :2], dim=-1)
    return dist_xy > max_distance
