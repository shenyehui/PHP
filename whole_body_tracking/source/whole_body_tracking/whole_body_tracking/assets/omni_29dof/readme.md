# Omni 29-DOF 人形机器人模型（omni_29dof_nohead_noshoe）

本仓库提供了 Omni 29 自由度人形机器人在 **NVIDIA Isaac Lab / Isaac Sim** 中的完整模型配置，包含 URDF、网格资源、电机（执行器）参数辨识结果以及可直接导入使用的 `ArticulationCfg`。拿到本模型后，配合下文说明即可快速完成模型导入与强化学习训练。

---

## 1. 环境要求（最低版本）

| 组件 | 最低版本 |
| --- | --- |
| Isaac Lab | **2.2.0** 及以上 |
| Isaac Sim | **5.0.0** 及以上 |
| Python | 3.10+（随 Isaac Lab 提供） |
| PyTorch | 随 Isaac Lab 提供 |

> 说明：本模型直接通过 `sim_utils.UrdfFileCfg` 在运行时由 URDF 转换生成，并使用了 `UrdfConverterCfg.JointDriveCfg.PDGainsCfg`、`replace_cylinders_with_capsules` 等接口，这些 API 需要 Isaac Lab 2.2.0 / Isaac Sim 5.0.0 及以上版本，低于此版本会因接口不兼容而导入失败。

---

## 2. 机器人基础参数

| 参数 | 数值 |
| --- | --- |
| 总自由度（DOF） | **29** |
| 连杆（link）数量 | 30（含 `base_link`） |
| 整机质量 | 约 **38.87 kg** |
| 初始站立高度（base 高度） | 0.8 m |
| 浮动基座 | 是（`fix_base=False`） |
| 自碰撞 | 开启（`enabled_self_collisions=True`） |
| 接触传感器 | 开启（`activate_contact_sensors=True`） |

### 2.1 自由度分布

| 部位 | 关节 | 数量 |
| --- | --- | --- |
| 左/右腿 | hip_pitch、hip_roll、hip_yaw、knee_pitch、ankle_pitch、ankle_roll | 6 × 2 = 12 |
| 腰部 | waist_yaw、waist_roll、waist_pitch | 3 |
| 左/右臂 | shoulder_pitch、shoulder_roll、shoulder_yaw、elbow_pitch、elbow_yaw、wrist_pitch、wrist_roll | 7 × 2 = 14 |
| **合计** | | **29** |

> 头部（head_yaw / head_pitch）已作为固定视觉件合并进 `waist_pitch_link`，不占用自由度；踝部为并联结构，模型中按等效关节处理。

### 2.2 关节限位（来自 URDF，单位：rad / Nm / rad·s⁻¹）

| 关节 | 下限 | 上限 | effort | velocity |
| --- | --- | --- | --- | --- |
| hip_pitch_* | -2.6864 | 2.6864 | 140 | 20.9 |
| hip_roll_l / hip_roll_r | -0.52 / -2.96 | 2.96 / 0.52 | 140 | 20.9 |
| hip_yaw_* | -2.75 | 2.75 | 90 | 32.9 |
| knee_pitch_* | 0 | 2.87 | 140 | 20.9 |
| ankle_pitch_* | -0.87 | 0.52 | 50 | 31.4 |
| ankle_roll_* | -0.26 | 0.26 | 50 | 31.4 |
| waist_yaw | -2.7 | 2.7 | 90 | 32.9 |
| waist_roll / waist_pitch | -0.52 | 0.52 | 50 | 31.4 |
| shoulder_pitch_* | -3.14 | 2.7 | 25 | 31.4 |
| shoulder_roll_l / shoulder_roll_r | -0.52 / -2.355 | 2.355 / 0.52 | 25 | 31.4 |
| shoulder_yaw_* | -2.61 | 2.61 | 25 | 31.4 |
| elbow_pitch_* | -2.61 | 0.52 | 25 | 31.4 |
| elbow_yaw_* | -2.09 | 2.09 | 25 | 31.4 |
| wrist_pitch_* / wrist_roll_* | -1.57 | 1.57 | 10 | 7.8 |

### 2.3 默认初始关节角（`init_state`）

| 关节 | 角度 (rad) |
| --- | --- |
| hip_pitch_* | -0.26178 |
| knee_*  | 0.52356 |
| ankle_pitch_* | -0.26178 |
| elbow_pitch_* | -0.7 |
| shoulder_pitch_* | 0.3 |
| shoulder_roll_* | 0.0 |
| 其余关节 | 0.0 |

### 2.4 执行器（电机）配置

模型采用经参数辨识的直流电机模型 `DelayedDCMotor`（在 `DCMotor` 基础上增加 2~5 个物理步的随机指令延迟，用于 sim-to-real）。电机型号定义于 `robots/tiangong.py`：

| 电机组 | 覆盖关节 | 峰值力矩 (Nm) | 峰值转速 (rad/s) | 刚度 stiffness | 阻尼 damping |
| --- | --- | --- | --- | --- | --- |
| HRA88P_22_5 | hip_pitch / hip_roll / knee_pitch | 140 | 20.9 | 120 | 5 |
| HRA88P_14_3 | hip_yaw / waist_yaw | 90 | 32.9 | 100 | 5 |
| HRA58P_Parallel | ankle_*（并联，力矩×2） | 25×2 | 31.4 | 30 | 3 |
| HRA55P_Parallel | waist_roll / waist_pitch（并联，力矩×2） | 55×2 | 20.9 | 120 | 5 |
| HRA58P_Serial | shoulder_* / elbow_* | 25 | 31.4 | 50 | 2 |
| HTM4438_30 | wrist_pitch / wrist_roll | 10 | 7.85 | 5 | 1 |

> `armature`、`friction`、`viscous_friction`、`dynamic_friction` 等参数为下肢关节的辨识结果，详见 `robots/omni_29dof_nohead_noshoe_dcmotor_identified.py`。饱和力矩 `saturation_torque` 由额定/峰值功率自动推算（见 `robots/joint_cfg.py`）。

---

## 3. 文件结构

```text
omni_29dof_v260705/
├── readme.md                         # 本说明文档
├── assets/                           # 模型资源
│   ├── __init__.py                   # 定义 ASSET_DIR（资源根目录）
│   └── omni_29dof_nohead_noshoe/
│       ├── urdf/
│       │   └── omni_29dof_nohead_noshoe_merged_modify_feet.urdf  # 主 URDF（固定关节已合并，足部经修改）
│       └── meshes/                   # 32 个 STL 网格文件
├── robots/                           # 机器人 ArticulationCfg 配置
│   ├── joint_cfg.py                  # JointCfg 电机基类（含饱和力矩推算）
│   ├── tiangong.py                   # 各电机型号参数（HRA55P/HRA58P/HRA88P/HTM4438）
│   └── omni_29dof_nohead_noshoe_dcmotor_identified.py  # 主配置：OMNI_DCMOTOR_IDENTIFIED_CFG
└── actuators/                        # 自定义执行器模型
    ├── __init__.py
    └── actuators_pd.py               # DelayedDCMotor / DelayedDCMotorCfg（带延迟的直流电机）
```

### 关键对象

- `OMNI_DCMOTOR_IDENTIFIED_CFG`：完整的 `ArticulationCfg`，可直接用于场景中生成机器人。
- `OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE`：各关节动作缩放系数（默认 0.25），用于将策略网络输出映射到关节目标。
- `ASSET_DIR`：资源目录常量，配置中通过它定位 URDF/meshes。

---

## 4. 如何导入模型

### 4.1 前置条件

将本仓库根目录（`omni_29dof_v260705/`）加入 `PYTHONPATH`，以保证 `assets`、`robots`、`actuators` 三个包可被正确 import。例如：

```bash
export PYTHONPATH=<path to your assets>/omni_29dof_v260705:$PYTHONPATH
```

或在你的训练工程中，把这三个文件夹拷贝到工程包路径下并保持相对引用关系不变。

### 4.2 最简导入示例

```python
from robots.omni_29dof_nohead_noshoe_dcmotor_identified import (
    OMNI_DCMOTOR_IDENTIFIED_CFG,
    OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE,
)

# 在场景中生成机器人：将 prim_path 指向你的环境命名空间
robot_cfg = OMNI_DCMOTOR_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

### 4.3 在 InteractiveScene 中使用

```python
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from robots.omni_29dof_nohead_noshoe_dcmotor_identified import OMNI_DCMOTOR_IDENTIFIED_CFG


@configclass
class OmniSceneCfg(InteractiveSceneCfg):
    robot = OMNI_DCMOTOR_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # ... 地形、光照、传感器等
```

首次运行时，Isaac Lab 会根据 URDF 自动转换生成 USD 并缓存（`assets/omni_29dof_nohead_noshoe/.asset_hash` 用于判断资源是否变化）。转换过程会打印各电机注册信息（来自 `JointCfg`），属正常输出。

### 4.4 训练中使用动作缩放

`OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE` 提供了每个关节的动作缩放（默认 0.25），可在动作项配置中使用：

```python
from robots.omni_29dof_nohead_noshoe_dcmotor_identified import OMNI_DCMOTOR_IDENTIFIED_ACTION_SCALE

# 例如在 JointPositionActionCfg 中作为 scale 传入
# action = scale * policy_output + default_joint_pos
```

---

## 5. 常见问题（FAQ）

- **导入报错找不到 `assets` / `robots` / `actuators`**：确认已将仓库根目录加入 `PYTHONPATH`（见 4.1）。
- **`UrdfConverterCfg` / `PDGainsCfg` 相关 API 报错**：多为 Isaac Lab / Isaac Sim 版本过低，请升级到 2.2.0 / 5.0.0 及以上。
- **网格加载失败**：URDF 中 mesh 使用相对路径 `../meshes/*.STL`，请勿改变 `urdf/` 与 `meshes/` 的相对目录关系。
- **踝/腰关节力矩偏大**：并联结构关节的 `saturation_effort` / `effort_limit` 已按 ×2 处理，属预期设计。
