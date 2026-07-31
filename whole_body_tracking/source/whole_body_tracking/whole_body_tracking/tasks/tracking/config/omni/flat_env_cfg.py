"""Omni teacher environment for climbing the motion-matched 1 m box."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.omni import (
    OMNI_ANCHOR_BODY_NAME,
    OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE,
    OMNI_DCMOTOR_IDENTIFIED_CFG,
    OMNI_END_EFFECTOR_BODY_NAMES,
    OMNI_FOOT_BODY_NAMES,
    OMNI_TRACKING_BODY_NAMES,
    OMNI_WRIST_BODY_NAMES,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    MySceneCfg,
    ObservationsCfg,
    VELOCITY_RANGE,
)


OMNI_BOX_SIZE = (1.0, 1.0, 1.0)
# The box is shifted 0.5 m along +X (red) and 0.2 m along +Y (green).  Isaac
# Lab's CuboidCfg position is its geometric centre, hence z=0.5 places the
# bottom at z=0 and the top at z=1 for this 1 m-high box.
OMNI_BOX_POS = (0.38, 0.2, 0.5)
OMNI_BOX_ROT = (1.0, 0.0, 0.0, 0.0)
# Fresh-training teacher settings.  Keep the maximum positive tracking reward
# unchanged while moving 0.5 of weight from the already over-emphasized wrists
# to the global anchor position (old: anchor=0.5, wrists=2.0).
OMNI_ANCHOR_TRACKING_REWARD_WEIGHT = 1.0
OMNI_ANCHOR_TRACKING_REWARD_STD = 0.30
OMNI_WRIST_TRACKING_REWARD_WEIGHT = 1.5
OMNI_WRIST_TRACKING_REWARD_STD = 0.15
# The full-motion rollout first falls behind during the explosive 2.5--3.0 s
# lift.  Give a fresh policy enough room to explore recovery, then tighten
# these thresholds only after deterministic completion is reliable.
OMNI_ANCHOR_TERMINATION_THRESHOLD = 0.35
OMNI_FOOT_TERMINATION_THRESHOLD = 0.35
OMNI_WRIST_TERMINATION_THRESHOLD = 0.50
# A failure in one one-second phase bin must also increase sampling of the
# preceding preparation bins.  With size=1, the old run repeatedly reset into
# the already-lifted 3--4 s states and under-trained the 2.5--2.8 s push-off.
OMNI_ADAPTIVE_KERNEL_SIZE = 3
OMNI_ADAPTIVE_KERNEL_LAMBDA = 0.8
OMNI_UNDESIRED_CONTACT_REGEX = (
    r"^(?!(?:" + "|".join(OMNI_END_EFFECTOR_BODY_NAMES) + r")$).+$"
)


@configclass
class OmniEventCfg:
    """Domain randomization with only Omni entity names."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=OMNI_ANCHOR_BODY_NAME),
            "com_range": {
                "x": (-0.025, 0.025),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class OmniRewardsCfg:
    """Shared tracking rewards with an Omni-only contact selection."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[OMNI_UNDESIRED_CONTACT_REGEX]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class OmniClimbRewardsCfg(OmniRewardsCfg):
    """Teacher-only reward balancing torso lift and accurate hand placement."""

    # Override the shared 0.5 anchor term for the climb teacher only.  Together
    # with the 1.5 wrist term below, this preserves the old total maximum
    # reward while changing the anchor/wrist balance from 0.5/2.0 to 1.0/1.5.
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=OMNI_ANCHOR_TRACKING_REWARD_WEIGHT,
        params={
            "command_name": "motion",
            "std": OMNI_ANCHOR_TRACKING_REWARD_STD,
        },
    )

    # The generic body reward averages over all 14 tracking bodies, so retain
    # a dedicated wrist signal, but no longer let it dominate anchor tracking.
    motion_wrist_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=OMNI_WRIST_TRACKING_REWARD_WEIGHT,
        params={
            "command_name": "motion",
            "std": OMNI_WRIST_TRACKING_REWARD_STD,
            "body_names": list(OMNI_WRIST_BODY_NAMES),
        },
    )


@configclass
class OmniTerminationsCfg:
    """Shared episode terminations expressed with Omni body names."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": list(OMNI_END_EFFECTOR_BODY_NAMES),
        },
    )


@configclass
class OmniClimbTerminationsCfg:
    """Teacher-only relaxed and independently tunable limb terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": OMNI_ANCHOR_TERMINATION_THRESHOLD,
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
        },
    )
    foot_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": OMNI_FOOT_TERMINATION_THRESHOLD,
            "body_names": list(OMNI_FOOT_BODY_NAMES),
        },
    )
    wrist_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": OMNI_WRIST_TERMINATION_THRESHOLD,
            "body_names": list(OMNI_WRIST_BODY_NAMES),
        },
    )


@configclass
class OmniTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-isolated tracking base for all Omni tasks."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=8)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: OmniRewardsCfg = OmniRewardsCfg()
    terminations: OmniTerminationsCfg = OmniTerminationsCfg()
    events: OmniEventCfg = OmniEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Debug markers do not affect physics or rewards and unnecessarily
        # enlarge the rendered scene for thousands of headless environments.
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False


def make_omni_box_cfg():
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleBox",
        spawn=sim_utils.CuboidCfg(
            size=OMNI_BOX_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=OMNI_BOX_POS, rot=OMNI_BOX_ROT),
    )


def configure_omni_tracking(cfg):
    """Install the Omni asset and reference body selection."""
    cfg.scene.robot = OMNI_DCMOTOR_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.scale = OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE
    cfg.commands.motion.anchor_body_name = OMNI_ANCHOR_BODY_NAME
    cfg.commands.motion.body_names = list(OMNI_TRACKING_BODY_NAMES)


@configclass
class OmniClimbEnvCfg(OmniTrackingEnvCfg):
    """Privileged-reference teacher task for the Omni 1 m climb motion."""

    rewards: OmniClimbRewardsCfg = OmniClimbRewardsCfg()
    terminations: OmniClimbTerminationsCfg = OmniClimbTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_omni_tracking(self)
        self.commands.motion.adaptive_kernel_size = OMNI_ADAPTIVE_KERNEL_SIZE
        self.commands.motion.adaptive_lambda = OMNI_ADAPTIVE_KERNEL_LAMBDA
        self.scene.box = make_omni_box_cfg()


@configclass
class OmniClimbPlayEnvCfg(OmniClimbEnvCfg):
    """Deterministic scene used to inspect the Omni reference and box pose."""

    def __post_init__(self):
        super().__post_init__()

        # The reference replay script writes states directly.  Disable every
        # training-only perturbation so the rendered pose is deterministic.
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.push_robot = None
        self.commands.motion.sampling_strategy = "zero"
        self.commands.motion.debug_vis = False
        self.observations.policy.enable_corruption = False
        self.scene.contact_forces.debug_vis = False


@configclass
class OmniClimbWoStateEstimationEnvCfg(OmniClimbEnvCfg):
    """Omni climb teacher without actor-side position/linear-velocity estimation."""

    def __post_init__(self):
        super().__post_init__()
        # Match the G1 Wo-State-Estimation contract exactly: the asymmetric
        # critic keeps its privileged observations, while the deployed actor
        # no longer receives global-position-derived anchor displacement or
        # estimated base linear velocity.
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class OmniClimbWoStateEstimationPlayEnvCfg(OmniClimbPlayEnvCfg):
    """Deterministic zero-phase evaluation for the 154-D Wo-State actor."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
