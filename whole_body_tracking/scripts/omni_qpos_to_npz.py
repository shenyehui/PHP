"""Convert Omni retarget ``qpos`` NPZ into tracking MotionLoader format.

Input contract for ``overbox_1m_original.npz``::

    qpos[:, :3]   root xyz
    qpos[:, 3:7]  root quaternion wxyz
    qpos[:, 7:]   29 joints in OMNI_JOINT_NAMES / URDF order

The script resamples the 30 Hz input to the 50 Hz policy rate and uses Isaac
Lab forward kinematics to generate all six fields required by MotionLoader.

Default overbox conversion::

    python scripts/omni_qpos_to_npz.py --headless

Custom motion conversion::

    python scripts/omni_qpos_to_npz.py --input_file raw.npz \
        --output_file motion_isaaclab_fps50.npz --output_fps 50 --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "Datasets/omni_dataset/overbox_1m_original.npz"

parser = argparse.ArgumentParser(description="Convert raw Omni qpos NPZ to Isaac tracking NPZ.")
parser.add_argument(
    "--input_file",
    default=str(DEFAULT_INPUT_FILE),
    help="Raw qpos NPZ (default: Datasets/omni_dataset/overbox_1m_original.npz).",
)
parser.add_argument("--output_file", default=None)
parser.add_argument("--output_fps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.omni import (
    OMNI_DCMOTOR_IDENTIFIED_CFG,
    OMNI_JOINT_NAMES,
)


@configclass
class OmniReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
    )
    robot: ArticulationCfg = OMNI_DCMOTOR_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class OmniQposMotion:
    def __init__(self, input_file: str, output_fps: int, device: str):
        with np.load(input_file) as source:
            if "qpos" not in source:
                raise KeyError("Input NPZ must contain qpos")
            qpos = np.asarray(source["qpos"], dtype=np.float32)
            input_fps = int(np.asarray(source["fps"]).item())

        if qpos.ndim != 2 or qpos.shape[1] != 7 + len(OMNI_JOINT_NAMES):
            raise ValueError("Expected qpos shape (T, 36), got {}".format(qpos.shape))
        if qpos.shape[0] < 3 or input_fps <= 0 or output_fps <= 0:
            raise ValueError("Motion needs >=3 frames and positive input/output FPS")
        if not np.isfinite(qpos).all():
            raise ValueError("qpos contains NaN or Inf")

        qpos_t = torch.from_numpy(qpos).to(device=device)
        root_quat = qpos_t[:, 3:7]
        quaternion_norm = torch.linalg.vector_norm(root_quat, dim=1)
        if torch.max(torch.abs(quaternion_norm - 1.0)).item() > 1.0e-3:
            raise ValueError("qpos[:, 3:7] is not a normalized wxyz quaternion")

        self.input_fps = input_fps
        self.output_fps = output_fps
        self.output_dt = 1.0 / output_fps
        duration = (qpos.shape[0] - 1) / input_fps
        output_frames = int(round(duration * output_fps)) + 1
        times = torch.linspace(0.0, duration, output_frames, device=device)
        source_position = times * input_fps
        index_0 = torch.floor(source_position).long().clamp(max=qpos.shape[0] - 1)
        index_1 = (index_0 + 1).clamp(max=qpos.shape[0] - 1)
        blend = source_position - index_0

        self.root_pos = self._lerp(qpos_t[index_0, :3], qpos_t[index_1, :3], blend[:, None])
        self.root_quat = self._slerp(qpos_t[index_0, 3:7], qpos_t[index_1, 3:7], blend)
        self.joint_pos = self._lerp(qpos_t[index_0, 7:], qpos_t[index_1, 7:], blend[:, None])
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.output_dt, dim=0)[0]
        self.root_ang_vel = self._so3_derivative(self.root_quat, self.output_dt)
        self.joint_vel = torch.gradient(self.joint_pos, spacing=self.output_dt, dim=0)[0]
        self.frame_count = output_frames

        print(
            "[INFO] Omni motion: {} frames @ {} Hz -> {} frames @ {} Hz ({:.3f} s)".format(
                qpos.shape[0], input_fps, output_frames, output_fps, duration
            )
        )

    @staticmethod
    def _lerp(a, b, blend):
        return a * (1.0 - blend) + b * blend

    @staticmethod
    def _slerp(a, b, blend):
        output = torch.empty_like(a)
        for index in range(a.shape[0]):
            output[index] = quat_slerp(a[index], b[index], blend[index])
        return output

    @staticmethod
    def _so3_derivative(quaternions, dt):
        relative = quat_mul(quaternions[2:], quat_conjugate(quaternions[:-2]))
        angular_velocity = axis_angle_from_quat(relative) / (2.0 * dt)
        return torch.cat((angular_velocity[:1], angular_velocity, angular_velocity[-1:]), dim=0)


def convert(sim: SimulationContext, scene: InteractiveScene, motion: OmniQposMotion, output_file: Path):
    robot = scene["robot"]
    joint_indices = robot.find_joints(OMNI_JOINT_NAMES, preserve_order=True)[0]
    if len(joint_indices) != len(OMNI_JOINT_NAMES):
        raise RuntimeError("Isaac robot did not resolve all 29 Omni joints")

    log = {
        # Keep the same one-element shape as this project's existing Isaac
        # tracking NPZ files (for example ``fps == [50]``).
        "fps": np.asarray([motion.output_fps], dtype=np.int64),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for frame in range(motion.frame_count):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = motion.root_pos[frame]
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = motion.root_quat[frame]
        root_state[:, 7:10] = motion.root_lin_vel[frame]
        root_state[:, 10:13] = motion.root_ang_vel[frame]

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indices] = motion.joint_pos[frame]
        joint_vel[:, joint_indices] = motion.joint_vel[frame]

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[key] = np.stack(log[key], axis=0)
        if not np.isfinite(log[key]).all():
            raise RuntimeError("Converted field {} contains NaN/Inf".format(key))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_file), **log)
    print("[INFO] Saved tracking motion: {}".format(output_file))
    print("[INFO] joint_pos={}, body_pos_w={}".format(log["joint_pos"].shape, log["body_pos_w"].shape))


def main():
    input_path = Path(args_cli.input_file).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args_cli.output_file:
        output_path = Path(args_cli.output_file).expanduser().resolve()
    else:
        # ``overbox_1m_original.npz`` becomes the concise, stable training
        # filename ``overbox_1m_isaaclab_fps50.npz``.  Other inputs follow the
        # same rule when their stem ends in ``_original``.
        output_stem = input_path.stem
        if output_stem.endswith("_original"):
            output_stem = output_stem[: -len("_original")]
        output_path = input_path.with_name(
            output_stem + "_isaaclab_fps{}.npz".format(args_cli.output_fps)
        )

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(OmniReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    motion = OmniQposMotion(str(input_path), args_cli.output_fps, sim.device)
    convert(sim, scene, motion, output_path)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
