# Distributed teacher and student training

The teacher and student entry points support optional synchronous multi-GPU
training through PyTorch `torchrun` and NCCL. Existing single-process commands
are unchanged when `--distributed` is omitted.

## Environment-count semantics

In distributed mode, `--distributed_num_envs_mode total` is the default:

- `--num_envs 4096` with 4 GPUs creates 1024 environments per GPU.
- The global rollout still contains 4096 environments.

Use `--distributed_num_envs_mode per_gpu` only when `--num_envs` should apply to
every GPU. For example, 4096 with 4 GPUs creates 16384 environments globally.

The total mode requires `--num_envs` to be evenly divisible by the number of
processes.

## Teacher example (4 GPUs, 4096 environments total)

```bash
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  scripts/rsl_rl/train.py \
  --distributed \
  --distributed_num_envs_mode total \
  --task Tracking-Climb-Flat-Omni-v0 \
  --seed 42 \
  --registry_name Datasets/omni_dataset/overbox_1m_isaaclab_fps50.npz \
  --num_envs 4096 \
  --headless \
  --logger tensorboard
```

## Student example (4 GPUs, 4096 environments total)

```bash
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  scripts/rsl_rl/train_student.py \
  --distributed \
  --distributed_num_envs_mode total \
  --task Tracking-Student-Encoder-Climb-Flat-Omni-v0 \
  --registry_name Datasets/omni_dataset/overbox_1m_isaaclab_fps50.npz \
  --teacher_logdirs logs/rsl_rl/omni_climb_teacher/2026-07-24_02-54-51 \
  --teacher_mode single \
  --command_vx 1.0 \
  --command_vy 0.0 \
  --num_envs 4096 \
  --headless
```

Each rank runs one Isaac Lab environment instance on `cuda:LOCAL_RANK`. Rank
seeds are `base_seed + global_rank`. Policy gradients, adaptive PPO learning
rate, masked student losses and reported loss scalars are synchronized. Only
global rank 0 writes TensorBoard events, configuration files and checkpoints.
The run records the allocation in `params/distributed.yaml`.

Do not pass a fixed `--device cuda:0` to a distributed job. `AppLauncher` and
the training scripts assign the correct local device automatically.

For a single-GPU run, invoke the original script directly and omit
`--distributed`; `--num_envs`, seed, device, logging and checkpoints keep their
original behavior.
