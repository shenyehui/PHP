# Copyright (c) 2025, Whole Body Tracking Contributors
# SPDX-License-Identifier: BSD-3-Clause

"""DistillPPO: A hybrid algorithm that combines policy distillation (behaviour cloning
from one or more teacher policies) with Proximal Policy Optimization (PPO) for RL
fine-tuning.

Supports:
  - Single-teacher or multi-teacher distillation
  - ``teacher_mode``: ``"single"`` | ``"multi_random"`` | ``"multi_best"``
  - DAgger-style teacher action sampling during rollouts
  - Configurable schedule for distill-vs-PPO weight
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCritic, EmpiricalNormalization


class DistillPPO(PPO):
    """Distillation + PPO algorithm for training a student policy from teacher(s).

    Teacher models are standard ``ActorCritic`` policies that have been pretrained
    with full privileged observations (including reference motion). They are loaded
    from disk and kept frozen during student training.
    """

    policy: nn.Module  # StudentEncoderPolicy
    requires_returns = True
    normalizes_teacher_observations = True

    def __init__(
        self,
        policy,
        # --- Distillation ----------------------------------------------------
        teacher_mode: str = "single",
        teacher_logdirs: list[str] | None = None,
        teacher_policy_cfg: dict | None = None,
        teacher_act_prob: float = 0.0,
        distill_loss_coef: float = 1.0,
        distill_loss_type: str = "mse",
        distill_only_iterations: int = 0,
        distill_ppo_weight_schedule: bool = True,
        distill_ppo_schedule_iterations: int = 10000,
        distill_weight_min: float = 0.1,
        distill_loss_schedule_scale: float = 10.0,
        disable_teacher_act_when_ppo: bool = True,
        teacher_act_disable_ppo_weight_threshold: float = 0.1,
        # --- PPO (passed through to super) -----------------------------------
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        weight_decay: float = 0.0,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        # -- PPO init ---------------------------------------------------------
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            weight_decay=weight_decay,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # -- Distillation settings ---------------------------------------------
        self.teacher_mode = teacher_mode
        self.teacher_logdirs = teacher_logdirs or []
        self.teacher_policy_cfg = teacher_policy_cfg or {}
        self.teacher_act_prob = teacher_act_prob
        self.distill_loss_coef = distill_loss_coef
        self.distill_only_iterations = distill_only_iterations
        self.distill_ppo_weight_schedule = distill_ppo_weight_schedule
        self.distill_ppo_schedule_iterations = distill_ppo_schedule_iterations
        self.distill_weight_min = distill_weight_min
        self.distill_loss_schedule_scale = distill_loss_schedule_scale
        self.disable_teacher_act_when_ppo = disable_teacher_act_when_ppo
        self.teacher_act_disable_ppo_weight_threshold = teacher_act_disable_ppo_weight_threshold

        # -- Loss function ----------------------------------------------------
        # Store name for per-sample loss computation in update()
        self._distill_loss_type = distill_loss_type
        if distill_loss_type == "mse":
            self.distill_loss_fn = F.mse_loss
        elif distill_loss_type == "huber":
            self.distill_loss_fn = F.huber_loss
        elif distill_loss_type == "smooth_l1":
            self.distill_loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown distill_loss_type: {distill_loss_type}")

        # -- Teacher models (loaded later via load_teachers) ------------------
        self.teachers = nn.ModuleList()
        self.teacher_normalizers = nn.ModuleList()
        self._teacher_loaded = False
        self._ppo_enabled = False

        # -- Current weight tracking ------------------------------------------
        self._current_lambda_d = 1.0  # distill weight
        self._current_lambda_ppo = 0.0  # PPO weight
        self._last_selected_teacher_idx: torch.Tensor | None = None  # for logging
        self._update_counter = 0  # tracks number of completed updates

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------
    def load_teachers(self, teacher_obs_dim: int, teacher_act_dim: int):
        """Load teacher models from the configured log directories.

        Each directory should contain a ``model_{iter}.pt`` checkpoint.
        The latest checkpoint is loaded automatically.  Input dimensions are
        inferred from the checkpoint weights so that they always match the
        original teacher architecture.
        """
        if self._teacher_loaded:
            return

        for logdir in self.teacher_logdirs:
            model_files = sorted(
                [f for f in os.listdir(logdir) if f.startswith("model_") and f.endswith(".pt")],
                key=lambda x: int(x.replace("model_", "").replace(".pt", "")),
            )
            if not model_files:
                raise FileNotFoundError(f"No model checkpoint found in {logdir}")
            latest = model_files[-1]
            ckpt_path = os.path.join(logdir, latest)

            # Infer teacher dimensions from the checkpoint itself
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model_sd = state_dict.get("model_state_dict", state_dict)

            # actor.0.weight : [hidden, num_actor_obs]
            num_actor_obs = model_sd["actor.0.weight"].shape[1]
            # critic.0.weight : [hidden, num_critic_obs]
            num_critic_obs = model_sd["critic.0.weight"].shape[1]
            # actor.<last>.bias : [num_actions]
            # find the last layer bias in actor
            actor_keys = sorted([k for k in model_sd if k.startswith("actor.") and k.endswith(".bias")])
            num_actions = model_sd[actor_keys[-1]].shape[0]

            print(
                f"[DistillPPO] Inferred teacher dims from checkpoint: "
                f"actor_obs={num_actor_obs}, critic_obs={num_critic_obs}, actions={num_actions}"
            )

            teacher = ActorCritic(
                num_actor_obs=num_actor_obs,
                num_critic_obs=num_critic_obs,
                num_actions=num_actions,
                **self.teacher_policy_cfg,
            ).to(self.device)

            teacher.load_state_dict(model_sd)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

            if num_actor_obs != teacher_obs_dim:
                raise ValueError(
                    f"Teacher observation mismatch for {ckpt_path}: checkpoint expects "
                    f"{num_actor_obs}, environment provides {teacher_obs_dim}."
                )
            if num_actions != teacher_act_dim:
                raise ValueError(
                    f"Teacher action mismatch for {ckpt_path}: checkpoint outputs "
                    f"{num_actions}, environment expects {teacher_act_dim}."
                )

            normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            normalizer_state = state_dict.get("obs_norm_state_dict")
            if normalizer_state is None:
                raise KeyError(
                    f"Teacher checkpoint {ckpt_path} has no obs_norm_state_dict. "
                    "The teacher was configured with empirical normalization, so raw "
                    "observations cannot be used safely."
                )
            normalizer.load_state_dict(normalizer_state)
            normalizer.eval()

            self.teachers.append(teacher)
            self.teacher_normalizers.append(normalizer)
            print(f"[DistillPPO] Loaded teacher from {ckpt_path}")

        self._teacher_loaded = True
        print(f"[DistillPPO] Total teachers loaded: {len(self.teachers)}")

    @property
    def loaded_teacher(self) -> bool:
        return self._teacher_loaded

    # ------------------------------------------------------------------
    # PPO-active logic
    # ------------------------------------------------------------------
    def _ppo_is_active(self) -> bool:
        return self._update_counter >= self.distill_only_iterations

    def _update_weights(self):
        """Update distill / PPO mixing weights based on the schedule."""
        self._ppo_enabled = self._ppo_is_active()

        if self.distill_ppo_weight_schedule and self._ppo_enabled:
            progress = float(self._update_counter - self.distill_only_iterations)
            total = float(max(self.distill_ppo_schedule_iterations, 1))
            lambda_d = max(self.distill_weight_min, 1.0 - progress / total)
            self._current_lambda_d = lambda_d
            self._current_lambda_ppo = 1.0 - lambda_d
        else:
            self._current_lambda_d = 1.0
            self._current_lambda_ppo = float(self._ppo_enabled)

        return self._current_lambda_d, self._current_lambda_ppo

    # ------------------------------------------------------------------
    # Storage: use RL-type + add privileged_actions for teacher targets
    # ------------------------------------------------------------------
    def init_storage(
        self, training_type, num_envs, num_transitions_per_env,
        actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # Always use "rl" type to get PPO fields (values, returns, advantages, etc.)
        super().init_storage("rl", num_envs, num_transitions_per_env,
                             actor_obs_shape, critic_obs_shape, actions_shape)
        # Manually add privileged_actions for distillation
        self.storage.privileged_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.storage.ppo_action_mask = torch.ones(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

    # ------------------------------------------------------------------
    # Action selection (rollout)
    # ------------------------------------------------------------------
    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        teacher_obs: torch.Tensor,
        action_label_mask: torch.Tensor | None = None,
    ):
        """Compute actions during rollout.

        With probability ``teacher_act_prob``, the teacher action is used instead
        of the student's (DAgger-style).
        """
        self._update_weights()

        student_actions = self.policy.act(obs).detach()
        teacher_actions = self._get_teacher_action(obs, teacher_obs)

        teacher_act_prob = self.teacher_act_prob
        if (
            self.disable_teacher_act_when_ppo
            and self._current_lambda_ppo > self.teacher_act_disable_ppo_weight_threshold
        ):
            teacher_act_prob = 0.0

        if teacher_act_prob > 0:
            use_teacher = torch.rand(obs.shape[0], device=self.device) < teacher_act_prob
        else:
            use_teacher = torch.zeros(obs.shape[0], dtype=torch.bool, device=self.device)
        self.transition.actions = torch.where(
            use_teacher.unsqueeze(-1), teacher_actions.detach(), student_actions
        )
        self.transition.ppo_action_mask = (~use_teacher).float().unsqueeze(-1)
        self.transition.action_label_mask = action_label_mask

        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.transition.privileged_actions = teacher_actions.detach()
        return self.transition.actions

    def _get_teacher_action(
        self, student_obs: torch.Tensor, teacher_obs: torch.Tensor
    ) -> torch.Tensor:
        """Get teacher action(s) based on ``teacher_mode``."""
        if not self._teacher_loaded:
            return torch.zeros_like(self.policy.act(student_obs))

        if self.teacher_mode == "single":
            normalized_obs = self.teacher_normalizers[0](teacher_obs)
            return self.teachers[0].act_inference(normalized_obs).detach()

        elif self.teacher_mode == "multi_random":
            idx = torch.randint(0, len(self.teachers), (student_obs.shape[0],), device=self.device)
            self._last_selected_teacher_idx = idx
            num_actions = self.policy.actor[-1].out_features
            actions = torch.zeros(student_obs.shape[0], num_actions, device=self.device)
            for t_idx, teacher in enumerate(self.teachers):
                mask = (idx == t_idx)
                if mask.any():
                    normalized_obs = self.teacher_normalizers[t_idx](teacher_obs[mask])
                    actions[mask] = teacher.act_inference(normalized_obs).detach()
            return actions

        elif self.teacher_mode == "multi_best":
            with torch.no_grad():
                student_act = self.policy.act_inference(student_obs)
            best_actions = torch.zeros_like(student_act)
            best_dist = torch.full((student_obs.shape[0],), float("inf"), device=self.device)

            for t_idx, teacher in enumerate(self.teachers):
                normalized_obs = self.teacher_normalizers[t_idx](teacher_obs)
                t_act = teacher.act_inference(normalized_obs).detach()
                dist = ((t_act - student_act) ** 2).sum(dim=-1)
                mask = dist < best_dist
                best_actions[mask] = t_act[mask]
                best_dist[mask] = dist[mask]

            return best_actions

        else:
            raise ValueError(f"Unknown teacher_mode: {self.teacher_mode}")

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self):
        """Perform one training update (distillation + optional PPO)."""
        self._update_weights()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_distill_loss = 0.0
        mean_total_loss = 0.0
        mean_label_fraction = 0.0
        num_updates = 0

        # Use RL mini_batch_generator (now yields privileged_actions_batch too)
        generator = self.storage.mini_batch_generator(
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        for batch in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                dones_batch,
                rnd_state_batch,
                privileged_actions_batch,
                action_label_mask_batch,
                ppo_action_mask_batch,
            ) = batch

            # -- Distillation loss -------------------------------------------
            distill_loss = torch.tensor(0.0, device=self.device)
            if privileged_actions_batch is not None and self._current_lambda_d > 0:
                student_actions_batch = self.policy.act_inference(obs_batch)
                # Per-sample loss across action dims (matches instinct_rl TPPO)
                if self._distill_loss_type in ("huber", "smooth_l1"):
                    raw_distill_loss = self.distill_loss_fn(
                        student_actions_batch, privileged_actions_batch, reduction="none"
                    ).mean(dim=-1)
                else:  # mse
                    raw_distill_loss = F.mse_loss(
                        student_actions_batch, privileged_actions_batch, reduction="none"
                    ).mean(dim=-1)
                # Apply action_label_mask: only distill when student is close to reference
                if action_label_mask_batch is not None:
                    mask = action_label_mask_batch.to(
                        device=raw_distill_loss.device, dtype=raw_distill_loss.dtype
                    ).squeeze(-1)
                    label_count = mask.sum()
                    if self.is_multi_gpu:
                        # Gradient averaging gives each rank equal weight. Scale
                        # the local numerator by the global valid-label count so
                        # this is equivalent to one global masked mean.
                        global_label_count = label_count.detach().clone()
                        torch.distributed.all_reduce(
                            global_label_count, op=torch.distributed.ReduceOp.SUM
                        )
                        distill_normalizer = (
                            self.gpu_world_size / global_label_count.clamp(min=1.0)
                        )
                    else:
                        distill_normalizer = 1.0 / label_count.clamp(min=1.0)
                    distill_loss = (raw_distill_loss * mask).sum() * distill_normalizer
                else:
                    distill_loss = raw_distill_loss.mean()

            # -- PPO losses --------------------------------------------------
            surrogate_loss = torch.tensor(0.0, device=self.device)
            value_loss = torch.tensor(0.0, device=self.device)
            entropy_bonus = torch.tensor(0.0, device=self.device)

            if self._ppo_enabled and self._current_lambda_ppo > 0:
                self.policy.update_distribution(obs_batch)
                new_actions_log_prob = self.policy.get_actions_log_prob(actions_batch)
                old_actions_log_prob_batch = old_actions_log_prob_batch.squeeze(-1)
                advantages_batch = advantages_batch.squeeze(-1)
                if new_actions_log_prob.shape != old_actions_log_prob_batch.shape:
                    raise RuntimeError(
                        "PPO log-probability shape mismatch: "
                        f"new={tuple(new_actions_log_prob.shape)}, "
                        f"old={tuple(old_actions_log_prob_batch.shape)}"
                    )
                if ppo_action_mask_batch is None:
                    ppo_mask = torch.ones_like(new_actions_log_prob)
                else:
                    ppo_mask = ppo_action_mask_batch.squeeze(-1).to(new_actions_log_prob)
                ppo_count = ppo_mask.sum()
                if self.is_multi_gpu:
                    global_ppo_count = ppo_count.detach().clone()
                    torch.distributed.all_reduce(
                        global_ppo_count, op=torch.distributed.ReduceOp.SUM
                    )
                    ppo_normalizer = (
                        self.gpu_world_size / global_ppo_count.clamp(min=1.0)
                    )
                else:
                    global_ppo_count = ppo_count
                    ppo_normalizer = 1.0 / ppo_count.clamp(min=1.0)
                entropy_bonus = (self.policy.entropy * ppo_mask).sum() * ppo_normalizer

                # Normalize advantages (critical for stability, matches PPO base)
                if self.normalize_advantage_per_mini_batch:
                    valid_advantages = advantages_batch[ppo_mask.bool()]
                    if valid_advantages.numel() > 1:
                        advantages_batch = (
                            advantages_batch - valid_advantages.mean()
                        ) / (valid_advantages.std() + 1e-8)

                ratio = torch.exp(new_actions_log_prob - old_actions_log_prob_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                ) * advantages_batch
                surrogate_loss = -(torch.min(surr1, surr2) * ppo_mask).sum() * ppo_normalizer

                values = self.policy.evaluate(critic_obs_batch)
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + torch.clamp(
                        values - target_values_batch,
                        -self.clip_param,
                        self.clip_param,
                    )
                    value_loss1 = (values - returns_batch).pow(2)
                    value_loss2 = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = (values - returns_batch).pow(2).mean()

                # Match rsl-rl's adaptive Gaussian KL schedule, but only after
                # the paper's PPO weight passes 0.1.
                if self.schedule == "adaptive" and self._current_lambda_ppo > 0.1:
                    with torch.inference_mode():
                        mu_batch = self.policy.action_mean
                        sigma_batch = self.policy.action_std
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (
                                torch.square(old_sigma_batch)
                                + torch.square(old_mu_batch - mu_batch)
                            )
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            dim=-1,
                        )
                        kl_sum = (kl * ppo_mask).sum()
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(
                                kl_sum, op=torch.distributed.ReduceOp.SUM
                            )
                            kl_mean = kl_sum / global_ppo_count.clamp(min=1.0)
                        else:
                            kl_mean = kl_sum / ppo_count.clamp(min=1.0)
                    if self.gpu_global_rank == 0:
                        kl_mean_value = kl_mean.item()
                        if kl_mean_value > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean_value < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # -- Total loss ---------------------------------------------------
            # distill_loss_schedule_scale amplifies distillation (instinctlab uses 10.0)
            total_loss = (
                self._current_lambda_d * self.distill_loss_coef * self.distill_loss_schedule_scale * distill_loss
                + self._current_lambda_ppo * self.value_loss_coef * value_loss
                + self._current_lambda_ppo * surrogate_loss
                - self._current_lambda_ppo * self.entropy_coef * entropy_bonus
            )

            # -- Gradient step ------------------------------------------------
            loss_is_finite = torch.isfinite(total_loss).to(dtype=torch.int32)
            if self.is_multi_gpu:
                # Every rank must either enter or skip the gradient collective;
                # otherwise a single bad rank would deadlock the whole job.
                torch.distributed.all_reduce(
                    loss_is_finite, op=torch.distributed.ReduceOp.MIN
                )
            if not bool(loss_is_finite.item()):
                if self.gpu_global_rank == 0:
                    print(
                        f"[DistillPPO] WARNING: NaN/Inf total_loss at step {self._update_counter}, "
                        f"skipping update on every rank. distill={distill_loss.item():.4g} "
                        f"surrogate={surrogate_loss.item():.4g} value={value_loss.item():.4g}"
                    )
                self.optimizer.zero_grad()
                continue
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_distill_loss += distill_loss.item()
            mean_total_loss += total_loss.item()
            if action_label_mask_batch is not None:
                mean_label_fraction += action_label_mask_batch.float().mean().item()
            else:
                mean_label_fraction += 1.0
            num_updates += 1

        # -- Post-update ------------------------------------------------------
        num_updates = max(num_updates, 1)
        self.storage.clear()
        self._update_counter += 1

        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "distill": mean_distill_loss / num_updates,
            "total": mean_total_loss / num_updates,
            "lambda_d": self._current_lambda_d,
            "lambda_ppo": self._current_lambda_ppo,
            "label_fraction": mean_label_fraction / num_updates,
        }

        if self._last_selected_teacher_idx is not None and len(self.teachers) > 1:
            for t_idx in range(len(self.teachers)):
                loss_dict[f"teacher_{t_idx}_frac"] = (
                    (self._last_selected_teacher_idx == t_idx).float().mean().item()
                )

        return loss_dict

    def get_training_state(self) -> dict:
        return {
            "update_counter": self._update_counter,
            "lambda_d": self._current_lambda_d,
            "lambda_ppo": self._current_lambda_ppo,
            "learning_rate": self.learning_rate,
        }

    def load_training_state(self, state: dict | None):
        if not state:
            return
        self._update_counter = int(state.get("update_counter", self._update_counter))
        self._current_lambda_d = float(state.get("lambda_d", self._current_lambda_d))
        self._current_lambda_ppo = float(state.get("lambda_ppo", self._current_lambda_ppo))
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
