from .joint_cfg import JointCfg

HRA55P = JointCfg(
    name="HRA55P",
    rated_torque = 17.5,
    peak_torque  = 55.0,
    peak_velocity = 20.9,
    rated_power  = 183.2,
    armature     = 0.02,
    # armature = 0.013879,
)

HRA58P = JointCfg(
    name="HRA58P",
    rated_torque = 7.6,
    peak_torque  = 25.0,
    peak_velocity = 31.4,
    rated_power  = 167.1,
    armature     = 0.005,
    # armature = 0.0027533,
)

HRA88P_14_3 = JointCfg(
    name="HRA88P_14_3",
    rated_torque = 19.0,
    peak_torque  = 90.0,
    peak_velocity = 32.9,
    rated_power  = 417.8,
    armature     = 0.02,
    # armature = 0.016684
)

HRA88P_22_5 = JointCfg(
    name="HRA88P_22_5",
    rated_torque = 30.0,
    peak_torque  = 140.0,
    peak_velocity = 20.9,
    rated_power  = 417.8,
    armature     = 0.02,
    # armature = 0.041125,
)

HTM4438_30 = JointCfg(
    name="HTM4438_30",
    rated_torque = 2.0,
    peak_torque  = 10.0,
    peak_velocity = 7.85,
    rated_power  = 6.0,
    armature     = 0.004,
)
