"""Project-facing Omni 29-DoF robot configuration."""

from whole_body_tracking.assets.omni_29dof.robots import (
    OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE,
    OMNI_DCMOTOR_IDENTIFIED_CFG,
)


OMNI_JOINT_NAMES = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]

OMNI_TRACKING_BODY_NAMES = [
    "base_link",
    "hip_roll_l_link",
    "knee_pitch_l_link",
    "ankle_roll_l_link",
    "hip_roll_r_link",
    "knee_pitch_r_link",
    "ankle_roll_r_link",
    "waist_pitch_link",
    "shoulder_roll_l_link",
    "elbow_pitch_l_link",
    "wrist_roll_l_link",
    "shoulder_roll_r_link",
    "elbow_pitch_r_link",
    "wrist_roll_r_link",
]

OMNI_ANCHOR_BODY_NAME = "waist_pitch_link"
OMNI_FOOT_BODY_NAMES = [
    "ankle_roll_l_link",
    "ankle_roll_r_link",
]
OMNI_WRIST_BODY_NAMES = [
    "wrist_roll_l_link",
    "wrist_roll_r_link",
]
OMNI_END_EFFECTOR_BODY_NAMES = OMNI_FOOT_BODY_NAMES + OMNI_WRIST_BODY_NAMES

__all__ = [
    "OMNI_DCMOTOR_IDENTIFIED_CFG",
    "OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE",
    "OMNI_JOINT_NAMES",
    "OMNI_TRACKING_BODY_NAMES",
    "OMNI_ANCHOR_BODY_NAME",
    "OMNI_FOOT_BODY_NAMES",
    "OMNI_WRIST_BODY_NAMES",
    "OMNI_END_EFFECTOR_BODY_NAMES",
]
