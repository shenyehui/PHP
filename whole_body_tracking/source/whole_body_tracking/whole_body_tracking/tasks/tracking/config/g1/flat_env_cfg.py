import isaaclab.sim as sim_utils
# 引入 AssetBaseCfg 用于创建固定不动的静态障碍物
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from whole_body_tracking.robots.g1_spherehand import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

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

# ==================== 【新添加：针对 box_climbing 实验的环境配置】 ====================
@configclass
class G1BoxClimbingEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        # 1. 首先继承前面所有的基础配置（包含机器人、关节映射等）
        super().__post_init__()

        # 2. 将箱子作为静态障碍物动态动态添加进训练场景 (MySceneCfg) 中
        # 因为路径前缀是 "{ENV_REGEX_NS}"，Isaac Lab 会自动在所有并行环境（如 4096 个子环境）的相对位置生成它
        self.scene.box = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ObstacleBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 0.8, 0.608),                         # 对应 JSON 中的 full_size_xyz
                collision_props=sim_utils.CollisionPropertiesCfg(), # 开启碰撞，这样机器人才能踩上去
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)), # 绿色
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.3664, -0.672, 0.304),                    # 对应 JSON 中的 position_xyz
                rot=(0.9983759397857438, 0.0, 0.0, 0.056969139513712616), # 对应 JSON 中的 quat_wxyz
            ),
        )
        #保留这两个观测
        #self.observations.policy.motion_anchor_pos_b = None
        #self.observations.policy.base_lin_vel = None

@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
