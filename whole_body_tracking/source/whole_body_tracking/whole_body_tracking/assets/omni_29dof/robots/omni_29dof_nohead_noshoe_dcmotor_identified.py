# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import ArticulationCfg

from ..assets import ASSET_DIR
from ..actuators.actuators_pd import DelayedDCMotorCfg
from .tiangong import HRA58P, HRA88P_14_3, HRA88P_22_5, HTM4438_30, HRA55P

OMNI_DCMOTOR_IDENTIFIED_CFG_TEMP = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/omni_29dof_ballhand/urdf/omni_29dof_ballhand_c_camera.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "hip_pitch_.*_joint": -0.26178,
            "knee_.*_joint": 0.52356,
            "ankle_pitch_.*_joint": -0.26178,
            "elbow_pitch_.*_joint": -0.7,
            "shoulder_pitch_.*_joint": 0.3,
            "shoulder_roll_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "HRA88P_22_5": DelayedDCMotorCfg(
            min_delay=2,
            max_delay=5,
            joint_names_expr=["hip_pitch_.*", "hip_roll_.*", "knee_pitch_.*"],
            saturation_effort=HRA88P_22_5.saturation_torque,
            effort_limit=HRA88P_22_5.peak_torque,
            velocity_limit=HRA88P_22_5.peak_velocity,
            stiffness=120,
            damping=5,
            armature={
                "hip_pitch_l_joint": 0.025875279679894447,
                "hip_pitch_r_joint": 0.025875279679894447,
                "hip_roll_l_joint": 0.05765285715460777,
                "hip_roll_r_joint": 0.05765285715460777,
                "knee_pitch_l_joint": 0.06578928977251053,
                "knee_pitch_r_joint": 0.06578928977251053,
            },
            friction={
                "hip_pitch_l_joint": 0.4877849817276001,
                "hip_pitch_r_joint": 0.4877849817276001,
                "hip_roll_l_joint": 2.492764472961426,
                "hip_roll_r_joint": 2.492764472961426,
                "knee_pitch_l_joint": 0.9012159705162048,
                "knee_pitch_r_joint": 0.9012159705162048,
            },
            dynamic_friction={
                "hip_pitch_l_joint": 0.4877849817276001,
                "hip_pitch_r_joint": 0.4877849817276001,
                "hip_roll_l_joint": 2.492764472961426,
                "hip_roll_r_joint": 2.492764472961426,
                "knee_pitch_l_joint": 0.9012159705162048,
                "knee_pitch_r_joint": 0.9012159705162048,
            },
            viscous_friction={
                "hip_pitch_l_joint": 1.5534991025924683,
                "hip_pitch_r_joint": 1.5534991025924683,
                "hip_roll_l_joint": 1.3765937089920044,
                "hip_roll_r_joint": 1.3765937089920044,
                "knee_pitch_l_joint": 0.9860152006149292,
                "knee_pitch_r_joint": 0.9860152006149292,
            },
        ),
        "HRA88P_14_3": DelayedDCMotorCfg(
            min_delay=2,
            max_delay=5,
            joint_names_expr=["hip_yaw_.*", "waist_yaw_joint"],
            saturation_effort=HRA88P_14_3.saturation_torque,
            effort_limit=HRA88P_14_3.peak_torque,
            velocity_limit=HRA88P_14_3.peak_velocity,
            stiffness=100,
            damping=5,
            armature={
                "hip_yaw_l_joint": 0.056901562958955765,
                "hip_yaw_r_joint": 0.056901562958955765,
                "waist_yaw_joint": HRA88P_14_3.armature,
            },
            friction={
                "hip_yaw_l_joint": 0.01984378695487976,
                "hip_yaw_r_joint": 0.01984378695487976,
                "waist_yaw_joint": HRA88P_14_3.peak_torque * 0.05,
            },
            dynamic_friction={
                "hip_yaw_l_joint": 0.01984378695487976,
                "hip_yaw_r_joint": 0.01984378695487976,
                "waist_yaw_joint": HRA88P_14_3.peak_torque * 0.04,
            },
            viscous_friction={
                "hip_yaw_l_joint": 1.678580641746521,
                "hip_yaw_r_joint": 1.678580641746521,
                "waist_yaw_joint": 0.5,
            },
        ),
        "HRA58P_Parallel": DelayedDCMotorCfg(
            min_delay=2,
            max_delay=5,
            joint_names_expr=["ankle_.*"],
            saturation_effort=HRA58P.saturation_torque * 2,
            effort_limit=HRA58P.peak_torque * 2,
            velocity_limit=HRA58P.peak_velocity,
            stiffness=30.0,
            damping=3.0,
            armature={
                "ankle_pitch_l_joint": 0.010765427723526955,
                "ankle_pitch_r_joint": 0.010765427723526955,
                "ankle_roll_l_joint": 0.10608603060245514,
                "ankle_roll_r_joint": 0.10608603060245514,
            },
            friction={
                "ankle_pitch_l_joint": 0.0026833266019821167,
                "ankle_pitch_r_joint": 0.0026833266019821167,
                "ankle_roll_l_joint": 0.7332212924957275,
                "ankle_roll_r_joint": 0.7332212924957275,
            },
            dynamic_friction={
                "ankle_pitch_l_joint": 0.0026833266019821167,
                "ankle_pitch_r_joint": 0.0026833266019821167,
                "ankle_roll_l_joint": 0.7332212924957275,
                "ankle_roll_r_joint": 0.7332212924957275,
            },
            viscous_friction={
                "ankle_pitch_l_joint": 0.817914605140686,
                "ankle_pitch_r_joint": 0.817914605140686,
                "ankle_roll_l_joint": 0.8721524477005005,
                "ankle_roll_r_joint": 0.8721524477005005,
            },
        ),
        "HRA55P_Parallel": DelayedDCMotorCfg(
            min_delay=2,
            max_delay=5,
            joint_names_expr=["waist_(roll|pitch).*"],
            saturation_effort=HRA55P.saturation_torque * 2,
            effort_limit=HRA55P.peak_torque * 2,
            velocity_limit=HRA55P.peak_velocity,
            stiffness=120.0,
            damping=5.0,
            armature=HRA55P.armature * 2,
            friction=HRA55P.peak_torque * 2 * 0.05,
            dynamic_friction=HRA55P.peak_torque * 2 * 0.04,
            viscous_friction=0.5,
        ),
        "HRA58P_Serial": DelayedDCMotorCfg(
            min_delay=2,
            max_delay=5,
            joint_names_expr=[
                "shoulder_pitch_.*",
                "shoulder_roll_.*",
                "shoulder_yaw_.*",
                "elbow_pitch_.*",
                "elbow_yaw_.*",
            ],
            saturation_effort=HRA58P.saturation_torque,
            effort_limit=HRA58P.peak_torque,
            velocity_limit=HRA58P.peak_velocity,
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
            friction=HRA58P.peak_torque * 0.05,
            dynamic_friction=HRA58P.peak_torque * 0.04,
            viscous_friction=0.25,
        ),
        "HTM4438_30": DelayedDCMotorCfg(
            min_delay=2,
            max_delay=5,
            joint_names_expr=["wrist_pitch_.*", "wrist_roll_.*"],
            saturation_effort=HTM4438_30.saturation_torque,
            effort_limit=HTM4438_30.peak_torque,
            velocity_limit=HTM4438_30.peak_velocity,
            stiffness=5.0,
            damping=1.0,
            armature=HTM4438_30.armature,
            friction=HTM4438_30.peak_torque * 0.05,
            dynamic_friction=HTM4438_30.peak_torque * 0.04,
            viscous_friction=0.25,
        ),
    },
)

OMNI_DCMOTOR_IDENTIFIED_CFG = OMNI_DCMOTOR_IDENTIFIED_CFG_TEMP.copy()
for actuator_cfg in OMNI_DCMOTOR_IDENTIFIED_CFG.actuators.values():
    actuator_cfg.effort_limit_sim = actuator_cfg.effort_limit
    actuator_cfg.velocity_limit_sim = actuator_cfg.velocity_limit

OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE = {
    joint: 0.25
    for joint in [
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
}
