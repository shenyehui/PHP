# PHP 学生蒸馏与部署指南

本实现以 [Perceptive Humanoid Parkour (arXiv:2602.15827)](https://arxiv.org/abs/2602.15827)
为主设置，并复用 `whole_body_tracking` 的 motion-tracking teacher。

## 论文一致的默认路径

默认任务是 `Tracking-Student-Encoder-Climb-Flat-G1-v0`：

- 深度：`58 x 87`，3 层 CNN + global average pooling，输出 32 维；
- Isaac Lab 2.2.0 使用官方 `TiledCameraCfg` 生成深度；2.2 的 `RayCasterCameraCfg`
  只支持一个静态 mesh，不能同时感知地面和每个环境中的攀爬箱；
- actor：`[2048, 1024, 512, 256, 128]`，ELU；
- policy obs：depth、projected gravity、base angular velocity、joint position、joint velocity、previous action、2D velocity command；
- 不使用 reference motion、base linear velocity 或 proprio history；
- depth 固定预处理：裁剪到 `[0.3, 3.0] m`，映射到 `[-0.5, 0.5]`；
- depth 训练噪声：每帧 offset `+-3 cm`、逐像素 Gaussian std `3 cm`；
- DAgger + PPO 从第 0 轮联合训练，`lambda_D` 在 10k 轮内由 1.0 降到 0.1；
- 总训练 20k 轮、24 steps/env、2 epochs、96 mini-batches、LR `3e-4`；
- student reference phase 均匀采样，不使用 teacher 的 failure-adaptive sampling。

仓库只保留这一条 encoder + DistillPPO 学生路线；旧 VAE 与旧版纯蒸馏实现已移除。

## 训练前检查

1. Teacher checkpoint 必须来自与当前 `teacher` observation 完全相同的任务。
2. Teacher checkpoint 必须包含 `obs_norm_state_dict`；每个 teacher 会使用自己的冻结 normalizer。
3. 深度画面必须能看到箱体，且 policy depth tensor 全部有限并位于 `[-0.5, 0.5]`。
4. 单条 climb reference 最好包含 nominal standing、approach、skill 和 recovery；不要直接用带明显速度的动态帧作为真实系统启动状态。

## 训练

```bash
cd /run/user/1000/gvfs/sftp:host=192.168.130.100/home/nubot/PHP/whole_body_tracking

python scripts/rsl_rl/train_student.py \
  --task Tracking-Student-Encoder-Climb-Flat-G1-v0 \
  --registry_name /path/to/motion.npz \
  --teacher_logdirs /path/to/teacher/run \
  --teacher_mode single \
  --command_vx 1.0 \
  --command_vy 0.0 \
  --num_envs 4096 \
  --headless
```

论文使用 16,384 个并行环境，但 Isaac Lab 2.2 的 tiled rendering 比 Warp ray caster
占用更多显存。先用 `--num_envs 64` 或 `128` 完成深度画面和训练 smoke test，再根据 GPU
显存逐步提高到 `256/512/...`；不要直接从 4,096 开始。减少环境数不改变网络接口，但吞吐量
和优化统计不再与论文完全等价。

恢复训练：

```bash
python scripts/rsl_rl/train_student.py \
  --task Tracking-Student-Encoder-Climb-Flat-G1-v0 \
  --registry_name /path/to/motion.npz \
  --teacher_logdirs /path/to/teacher/run \
  --load_run 20260710_120000 \
  --checkpoint 'model_5000.pt' \
  --headless
```

恢复会加载 student、optimizer、课程进度、learning rate 和（若启用）student normalizer；teacher 及各自 normalizer 会从 teacher checkpoint 重新加载。

## Play 与同步深度可视化

```bash
python scripts/rsl_rl/play_student.py \
  --task Tracking-Student-Encoder-Climb-Flat-G1-v0 \
  --motion_file /path/to/motion.npz \
  --load_run 20260710_120000 \
  --checkpoint 'model_19999.pt' \
  --num_envs 1 \
  --depth_view
```

- `--depth_view`：OpenCV 窗口同步显示 policy 实际收到的 depth tensor；
- Isaac Lab 2.2 tiled camera 没有 3D ray-hit marker；兼容参数 `--depth_ray_debug`
  只会打印提示，不应再使用；
- 默认从 reference phase 0 开始；`--random_start` 才会均匀随机 phase；
- play 会关闭 observation corruption 和 depth noise。

导出目录包含：

- `policy.onnx`：完整 student inference pipeline；若存在 empirical normalizer，已嵌入 ONNX；
- `isaac_obs_raw.npy`：环境拼接后的原始 policy observation；
- `isaac_obs_policy_input.npy`：进入网络前的 observation；
- `isaac_root_state.npy`、`isaac_joint_pos.npy`：sim2sim parity 状态。

不要在 MuJoCo 端对 `normalizer_embedded=true` 的 ONNX 再做一次 normalization。

## 关键监控项

- `Loss/distill`：teacher action MSE；
- `Loss/surrogate`、`Loss/value`：PPO；
- `Loss/lambda_d`、`Loss/lambda_ppo`：课程权重；
- `Loss/label_fraction`：仍满足 teacher 原始 termination、可安全使用 DAgger label 的比例；
- depth 画面：箱体应在中心/前方 ROI 中出现，不得只有地面渐变或 `inf/nan`。

## 默认姿态与 reference 启动

当前 motion command 在训练 reset 时将机器人写到均匀采样的 reference phase，并非固定只从第一帧开始。若 MuJoCo 从另一个默认站姿启动，而训练 reference 从未覆盖该姿态，启动确实属于 OOD，可能立即摔倒，但还必须同时核对 joint order、action offset/scale、PD、depth preprocess 和 observation order。

不建议让 teacher 从任意默认姿态直接追踪一个高速动态 reference。推荐顺序是：

1. reference 加 nominal standing/零速度 hold；
2. 加入平滑的 standing -> approach -> skill 过渡；
3. teacher 仍按对应 reference phase reset；
4. student 覆盖 standing/start phase；
5. 部署先由站立控制器保持，再平滑切入 student。

将 MuJoCo 临时 reset 到 reference frame 0 只适合做 A/B 和 parity 验证，不是最终真实机器人启动方案。
