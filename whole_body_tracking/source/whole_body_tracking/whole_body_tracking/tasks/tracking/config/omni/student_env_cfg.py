"""Depth-student distillation environment for the Omni 1 m box climb."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as tracking_mdp
import whole_body_tracking.tasks.tracking.mdp.student_observations as student_obs
from whole_body_tracking.robots.omni import (
    OMNI_FOOT_BODY_NAMES,
    OMNI_WRIST_BODY_NAMES,
)
from whole_body_tracking.tasks.tracking.config.omni.flat_env_cfg import (
    OMNI_ANCHOR_TERMINATION_THRESHOLD,
    OMNI_FOOT_TERMINATION_THRESHOLD,
    OMNI_WRIST_TERMINATION_THRESHOLD,
    OmniClimbRewardsCfg,
    OmniTerminationsCfg,
    OmniTrackingEnvCfg,
    configure_omni_tracking,
    make_omni_box_cfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import MySceneCfg


@configclass
class OmniStudentDepthCameraCfg(TiledCameraCfg):
    """Legacy depth profile mounted at the Omni camera URDF joint.

    ``d435_joint`` in omni_29dof_ballhand_c_camera.urdf is fixed to
    ``waist_pitch_link`` at xyz=(0.068668, 0.0475, 0.231012) and
    rpy=(-2.513273708282232, 0, -pi/2).  The empty d435_link is an ROS optical
    frame (+Z forward), hence ``convention='ros'``.
    """

    prim_path = "{ENV_REGEX_NS}/Robot/waist_pitch_link/depth_camera"
    offset = TiledCameraCfg.OffsetCfg(
        pos=(0.068668, 0.0475, 0.231012),
        rot=(
            0.21850815162985124,
            -0.6724984666683678,
            0.6724984666683680,
            -0.21850815162985135,
        ),
        convention="ros",
    )
    spawn = sim_utils.PinholeCameraCfg(
        focal_length=1.0,
        horizontal_aperture=2 * math.tan(math.radians(87.0) / 2),
        vertical_aperture=2 * math.tan(math.radians(58.0) / 2),
        clipping_range=(0.05, 3.0),
    )
    height = 58
    width = 87
    data_types = ["distance_to_image_plane"]
    update_period = 1 / 30
    debug_vis = False
    depth_clipping_behavior = "max"


@configclass
class OmniActionLabelMaskObsCfg(ObservationGroupCfg):
    action_label_mask = ObservationTermCfg(
        func=student_obs.student_action_label_mask,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            # Gate labels using the exact support of the teacher checkpoint
            # trained by OmniClimbEnvCfg.  Student episode termination remains
            # intentionally looser (0.5 -> 1.0 m) below.
            "base_pos_threshold": OMNI_ANCHOR_TERMINATION_THRESHOLD,
            "ee_pos_threshold": None,
            "anchor_ori_threshold": 0.8,
            "body_pos_checks": [
                {
                    "threshold": OMNI_FOOT_TERMINATION_THRESHOLD,
                    "body_names": list(OMNI_FOOT_BODY_NAMES),
                },
                {
                    "threshold": OMNI_WRIST_TERMINATION_THRESHOLD,
                    "body_names": list(OMNI_WRIST_BODY_NAMES),
                },
            ],
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class OmniStudentObservationsCfg:
    """Depth-student, teacher and critic observations for Omni."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        depth_image = ObservationTermCfg(
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
        projected_gravity = ObservationTermCfg(
            func=student_obs.student_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObservationTermCfg(
            func=student_obs.student_base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_pos = ObservationTermCfg(
            func=student_obs.student_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObservationTermCfg(
            func=student_obs.student_joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObservationTermCfg(func=student_obs.student_last_action)
        velocity_command = ObservationTermCfg(
            func=student_obs.student_velocity_command,
            params={"command": (1.0, 0.0)},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObservationGroupCfg):
        command = ObservationTermCfg(
            func=tracking_mdp.generated_commands, params={"command_name": "motion"}
        )
        motion_anchor_pos_b = ObservationTermCfg(
            func=tracking_mdp.motion_anchor_pos_b, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObservationTermCfg(
            func=tracking_mdp.motion_anchor_ori_b, params={"command_name": "motion"}
        )
        base_lin_vel = ObservationTermCfg(func=tracking_mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=tracking_mdp.base_ang_vel)
        joint_pos = ObservationTermCfg(func=tracking_mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=tracking_mdp.joint_vel_rel)
        actions = ObservationTermCfg(func=tracking_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        command = ObservationTermCfg(
            func=tracking_mdp.generated_commands, params={"command_name": "motion"}
        )
        motion_anchor_pos_b = ObservationTermCfg(
            func=tracking_mdp.motion_anchor_pos_b, params={"command_name": "motion"}
        )
        motion_anchor_ori_b = ObservationTermCfg(
            func=tracking_mdp.motion_anchor_ori_b, params={"command_name": "motion"}
        )
        body_pos = ObservationTermCfg(
            func=tracking_mdp.robot_body_pos_b, params={"command_name": "motion"}
        )
        body_ori = ObservationTermCfg(
            func=tracking_mdp.robot_body_ori_b, params={"command_name": "motion"}
        )
        base_lin_vel = ObservationTermCfg(func=tracking_mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=tracking_mdp.base_ang_vel)
        joint_pos = ObservationTermCfg(func=tracking_mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=tracking_mdp.joint_vel_rel)
        actions = ObservationTermCfg(func=tracking_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()
    critic: CriticCfg = CriticCfg()
    action_label_mask: OmniActionLabelMaskObsCfg = OmniActionLabelMaskObsCfg()


@configclass
class OmniStudentSceneCfg(MySceneCfg):
    depth_camera: TiledCameraCfg = OmniStudentDepthCameraCfg()

    def __post_init__(self):
        super().__post_init__()
        self.box = make_omni_box_cfg()


@configclass
class OmniStudentTerminationsCfg(OmniTerminationsCfg):
    """Student terminations with independently logged feet and wrists.

    ``foot_body_pos OR wrist_body_pos`` is exactly equivalent to the former
    combined ``ee_body_pos`` term while both groups use the same threshold.
    Keeping separate terms only improves failure diagnostics.
    """

    # Disable the inherited combined end-effector term.
    ee_body_pos = None
    foot_body_pos = DoneTermCfg(
        func=tracking_mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.5,
            "body_names": list(OMNI_FOOT_BODY_NAMES),
        },
    )
    wrist_body_pos = DoneTermCfg(
        func=tracking_mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.5,
            "body_names": list(OMNI_WRIST_BODY_NAMES),
        },
    )


@configclass
class OmniStudentEnvCfg(OmniTrackingEnvCfg):
    """Single supported Omni encoder-student distillation task."""

    # RTX tiled depth buffers dominate GPU/Vulkan memory.  Start from a
    # conservative default for long-running training; --num_envs can still
    # raise this after a successful camera smoke test on the target GPU.
    scene: OmniStudentSceneCfg = OmniStudentSceneCfg(num_envs=1024, env_spacing=8)
    observations: OmniStudentObservationsCfg = OmniStudentObservationsCfg()
    # DistillPPO transitions to 90% PPO, so its on-policy objective must retain
    # the successful teacher's climb-specific anchor/wrist reward balance.
    rewards: OmniClimbRewardsCfg = OmniClimbRewardsCfg()
    terminations: OmniStudentTerminationsCfg = OmniStudentTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_omni_tracking(self)
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
        self.events.push_robot = None

        self.terminations.anchor_pos.params["threshold"] = 0.5
        self.terminations.out_of_border = DoneTermCfg(
            func=_omni_student_terrain_out_of_bounds,
            time_out=True,
            params={"max_distance": 10.0},
        )
        self.curriculum.termination_schedule = CurrTerm(
            func=omni_student_termination_schedule,
            params={
                "start_threshold": 0.5,
                "end_threshold": 1.0,
                "schedule_steps": 10000 * 24,
            },
        )


@configclass
class OmniD455StudentDepthCameraCfg(OmniStudentDepthCameraCfg):
    """Low-resolution training model of the real Intel RealSense D455.

    The 64x36 render preserves the D455 16:9 image geometry while avoiding the
    prohibitive cost of rendering the physical 848x480 stream in every
    parallel environment.  The nominal 86x57 degree field of view is used
    until robot-specific ROS CameraInfo intrinsics are supplied.
    """

    spawn = sim_utils.PinholeCameraCfg(
        focal_length=1.0,
        horizontal_aperture=2 * math.tan(math.radians(86.0) / 2),
        vertical_aperture=2 * math.tan(math.radians(57.0) / 2),
        # Render beyond the policy's 5 m limit so preprocessing can distinguish
        # out-of-range returns from a valid sample exactly at 5 m.
        clipping_range=(0.05, 6.0),
    )
    height = 36
    width = 64
    update_period = 1 / 30
    depth_clipping_behavior = "none"


@configclass
class OmniD455StudentSceneCfg(OmniStudentSceneCfg):
    depth_camera: TiledCameraCfg = OmniD455StudentDepthCameraCfg()


@configclass
class OmniD455StudentEnvCfg(OmniStudentEnvCfg):
    """Omni student task matching the real D455 deployment preprocessing."""

    scene: OmniD455StudentSceneCfg = OmniD455StudentSceneCfg(
        num_envs=1024, env_spacing=8
    )

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.depth_image.params.update(
            {
                "min_distance": 0.5,
                "max_distance": 5.0,
                "preprocessing": "range_mask_then_divide_by_max",
            }
        )


def omni_student_termination_schedule(
    env,
    env_ids,
    start_threshold: float = 0.5,
    end_threshold: float = 1.0,
    schedule_steps: int = 240000,
):
    """Relax the Omni student's tracking termination threshold over training."""
    if len(env_ids) == 0:
        return None
    progress = min(float(env.common_step_counter) / max(float(schedule_steps), 1.0), 1.0)
    threshold = start_threshold + (end_threshold - start_threshold) * progress
    for term_name in ("anchor_pos", "foot_body_pos", "wrist_body_pos"):
        term_cfg = getattr(env.cfg.terminations, term_name, None)
        if term_cfg is not None:
            term_cfg.params["threshold"] = threshold
        if hasattr(env, "termination_manager") and term_name in env.termination_manager.active_terms:
            live_term_cfg = env.termination_manager.get_term_cfg(term_name)
            live_term_cfg.params["threshold"] = threshold
            env.termination_manager.set_term_cfg(term_name, live_term_cfg)
    return {"termination_threshold": threshold}


def _omni_student_terrain_out_of_bounds(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float = 5.0,
) -> torch.Tensor:
    """Terminate when the Omni root leaves its own flat-scene workspace."""
    asset = env.scene[asset_cfg.name]
    root_pos_env = asset.data.root_pos_w - env.scene.env_origins
    return torch.norm(root_pos_env[:, :2], dim=-1) > max_distance
