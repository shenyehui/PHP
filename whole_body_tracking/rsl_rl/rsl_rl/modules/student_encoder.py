# Copyright (c) 2025, Whole Body Tracking Contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Depth encoder policy used by the PHP student training route."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.conv2d_encoder import Conv2dEncoder
from rsl_rl.utils import resolve_nn_activation


class StudentEncoderPolicy(nn.Module):
    """Student actor with depth encoding and a privileged-state critic.

    The flat policy observation starts with the depth image, followed by
    projected gravity, base angular velocity, joint position, joint velocity,
    previous action, and a planar velocity command. Reference motion is never
    part of the policy input.
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        encoder_config: dict | None = None,
        obs_components: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "StudentEncoderPolicy.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs))
            )
        actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]
        activation_fn = resolve_nn_activation(activation)

        if encoder_config is None:
            self.encoder = None
            encoder_output_size = 0
            self._depth_channels = 0
            self._depth_height = 0
            self._depth_width = 0
        else:
            self.encoder = Conv2dEncoder(**encoder_config)
            encoder_output_size = self.encoder.output_size
            self._depth_channels = encoder_config.get("input_channels", 1)
            self._depth_height = encoder_config.get("input_height", 18)
            self._depth_width = encoder_config.get("input_width", 32)

        if obs_components is not None:
            self._obs_components = obs_components
        elif self.encoder is not None:
            depth_size = self.depth_image_numel
            self._obs_components = {"depth_image": {"start": 0, "end": depth_size}}
            print(f"[StudentEncoderPolicy] Auto-detected depth_image at obs[0:{depth_size}]")
        else:
            self._obs_components = {}

        actor_input_size = num_obs
        if self.encoder is not None:
            actor_input_size = num_obs - self.depth_image_numel + encoder_output_size
        self.actor = self._make_mlp(
            actor_input_size,
            actor_hidden_dims,
            num_actions,
            activation_fn,
        )
        self.critic = self._make_mlp(
            num_critic_obs,
            critic_hidden_dims,
            1,
            activation_fn,
        )

        self.log_std = nn.Parameter(
            torch.log(torch.tensor(init_noise_std)) * torch.ones(num_actions)
        )
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        print(f"[StudentEncoderPolicy] actor: {self.actor}")
        print(f"[StudentEncoderPolicy] critic: {self.critic}")

    @staticmethod
    def _make_mlp(
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation_fn: nn.Module,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        dimensions = [input_size, *hidden_sizes, output_size]
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(activation_fn)
        return nn.Sequential(*layers)

    @property
    def depth_image_numel(self) -> int:
        return self._depth_channels * self._depth_height * self._depth_width

    def _encode_depth(self, observations: torch.Tensor) -> torch.Tensor:
        if self.encoder is None or "depth_image" not in self._obs_components:
            return observations
        component = self._obs_components["depth_image"]
        start, end = component["start"], component["end"]
        depth = observations[:, start:end].reshape(
            observations.shape[0],
            self._depth_channels,
            self._depth_height,
            self._depth_width,
        )
        latent = self.encoder(depth)
        return torch.cat(
            [observations[:, :start], latent, observations[:, end:]],
            dim=-1,
        )

    def _forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self._encode_depth(observations))

    def update_distribution(self, observations: torch.Tensor):
        mean = self._forward_actor(observations)
        std = torch.exp(self.log_std).clamp(min=1.0e-6, max=1.0)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._forward_actor(observations)

    def evaluate(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a checkpoint and follow the runner's boolean resume contract."""
        super().load_state_dict(state_dict, strict=strict)
        return True
