from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.distributions import LowRankMultivariateNormal
from torch.distributions import Normal


def _activation(name: str) -> torch.nn.Module:
    name = name.lower()
    if name == "elu":
        return torch.nn.ELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "selu":
        return torch.nn.SELU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name in {"identity", "linear"}:
        return torch.nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _range_logit(value: float, low: float, high: float) -> float:
    ratio = min(max((value - low) / (high - low), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(ratio / (1.0 - ratio))


def _bounded(raw: torch.Tensor, low: float, high: float, initial: float) -> torch.Tensor:
    bias = _range_logit(initial, low, high)
    return low + (high - low) * torch.sigmoid(raw + bias)


def _unpad_trajectories(trajectories: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    valid_steps = trajectories.transpose(1, 0)[masks.transpose(1, 0).bool()]
    return valid_steps.view(-1, trajectories.shape[0], *trajectories.shape[2:]).transpose(1, 0)


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Sequence[int], activation: str = "elu"):
        super().__init__()
        layers: list[torch.nn.Module] = []
        last_dim = int(in_dim)
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(last_dim, int(hidden_dim)))
            layers.append(_activation(activation))
            last_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(last_dim, int(out_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaitCpgRNN(torch.nn.Module):
    """Low-dimensional traveling-wave CPG with an RNN-shaped interface.

    Hidden/output state layout:
        [phase, frequency, amplitude, signed_phase_lag, offset, turn_bias, residual_coeffs...]
    """

    def __init__(
        self,
        input_size: int,
        dt: float = 0.02,
        feedback_hidden_dims: Sequence[int] | None = None,
        activation: str = "elu",
        initial_frequency: float = 0.75,
        initial_amplitude: float = 1.0,
        initial_phase_lag: float = 2.0 * math.pi / 7.0,
        initial_offset: float = 0.0,
        initial_turn_bias: float = 0.0,
        residual_scale: float = 0.0,
        residual_dim: int = 5,
        command_start: int = 6,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.command_start = int(command_start)
        self.residual_scale = float(residual_scale)
        self.residual_dim = int(residual_dim) if self.residual_scale > 0.0 else 0
        self.hidden_size = 6 + self.residual_dim
        self.num_layers = 1
        self.dt = float(dt)

        self.initial_frequency = float(initial_frequency)
        self.initial_amplitude = float(initial_amplitude)
        self.initial_phase_lag = float(initial_phase_lag)
        self.initial_offset = float(initial_offset)
        self.initial_turn_bias = float(initial_turn_bias)
        self.frequency_min = 0.15
        self.frequency_max = 2.0
        self.amplitude_min = 0.20
        self.amplitude_max = 1.50
        self.phase_lag_min = 0.35
        self.phase_lag_max = 1.55
        self.offset_limit = 0.35
        self.turn_bias_limit = 0.35
        self.command_speed_ref = 0.20
        self.command_amplitude_floor = 0.85
        self.command_vx_deadband = 0.01
        self.command_vy_scale = 0.10

        if feedback_hidden_dims is None:
            feedback_hidden_dims = (128, 128)
        self.param_net = MLP(self.input_size, 5 + self.residual_dim, feedback_hidden_dims, activation)
        last_layer = self.param_net.net[-1]
        if isinstance(last_layer, torch.nn.Linear):
            torch.nn.init.zeros_(last_layer.weight)
            torch.nn.init.zeros_(last_layer.bias)

    def _command_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.command_start : self.command_start + 3]

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        state[:, 1] = self.initial_frequency
        state[:, 2] = self.initial_amplitude
        state[:, 3] = self.initial_phase_lag
        state[:, 4] = self.initial_offset
        state[:, 5] = self.initial_turn_bias
        h = state.unsqueeze(0)
        c = torch.zeros_like(h)
        return h, c

    def _params_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.param_net(obs)
        command = self._command_from_obs(obs)
        cmd_vx = command[:, 0]
        cmd_vy = command[:, 1]
        cmd_speed = torch.norm(command[:, :2], dim=1)
        command_gate = torch.clamp(cmd_speed / self.command_speed_ref, min=0.0, max=1.0)
        forward_sign = torch.where(
            torch.abs(cmd_vx) > self.command_vx_deadband,
            torch.sign(cmd_vx),
            torch.ones_like(cmd_vx),
        )
        frequency = _bounded(
            raw[:, 0] + 0.45 * command_gate,
            low=self.frequency_min,
            high=self.frequency_max,
            initial=self.initial_frequency,
        )
        learned_amplitude = _bounded(
            raw[:, 1] + 0.35 * command_gate,
            low=self.amplitude_min,
            high=self.amplitude_max,
            initial=self.initial_amplitude,
        )
        amplitude_floor = self.amplitude_min + command_gate * (self.command_amplitude_floor - self.amplitude_min)
        amplitude = torch.maximum(learned_amplitude, amplitude_floor)
        phase_lag_magnitude = _bounded(
            raw[:, 2], low=self.phase_lag_min, high=self.phase_lag_max, initial=self.initial_phase_lag
        )
        phase_lag = forward_sign * phase_lag_magnitude
        lateral_bias = torch.clamp(cmd_vy / self.command_vy_scale, min=-1.0, max=1.0)
        offset = self.offset_limit * torch.tanh(raw[:, 3] + 0.35 * lateral_bias)
        turn_bias = self.turn_bias_limit * torch.tanh(raw[:, 4] + lateral_bias)
        gait_params = torch.stack((frequency, amplitude, phase_lag, offset, turn_bias), dim=-1)
        if self.residual_dim == 0:
            return gait_params
        residual_coeffs = self.residual_scale * torch.tanh(raw[:, 5 : 5 + self.residual_dim])
        return torch.cat((gait_params, residual_coeffs), dim=-1)

    def _step(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        params = self._params_from_obs(obs)
        phase = state[:, 0] + self.dt * 2.0 * math.pi * params[:, 0]
        phase = torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
        next_state = torch.cat((phase.unsqueeze(-1), params), dim=-1)
        return torch.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        squeeze_time = x.dim() == 2
        if squeeze_time:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Expected 2D or 3D input.")

        time_steps, batch_size, _ = x.shape
        if hx is None:
            h, c = self.initial_state(batch_size, x.device, x.dtype)
        else:
            h, c = hx
            h = h.to(device=x.device, dtype=x.dtype)
            c = c.to(device=x.device, dtype=x.dtype)

        state = h[0]
        outputs: list[torch.Tensor] = []
        for step in range(time_steps):
            state = self._step(x[step], state)
            outputs.append(state)

        h_next = state.unsqueeze(0)
        c_next = torch.zeros_like(c)
        output = torch.stack(outputs, dim=0)
        if squeeze_time:
            output = output.squeeze(0)
        return output, (h_next, c_next)


class LSTM(GaitCpgRNN):
    """IsaacLab exporter compatibility shim for the custom gait CPG.

    The default RSL-RL exporter dispatches recurrent policies by class name and supports only ``lstm``/``gru``.
    This adapter keeps the CPG equations and LSTM-like ``(output, (h, c))`` interface unchanged.
    """


class GaitWaveReadout(torch.nn.Module):
    """Expands low-dimensional gait parameters into ordered yaw-joint actions."""

    def __init__(self, num_actions: int, action_limit: float = 1.0):
        super().__init__()
        self.num_actions = int(num_actions)
        self.action_limit = float(action_limit)
        joint_index = torch.arange(self.num_actions, dtype=torch.float32)
        centered = (joint_index - 0.5 * (self.num_actions - 1)) / max(1.0, 0.5 * (self.num_actions - 1))
        self.register_buffer("joint_index", joint_index)
        self.register_buffer("centered_joint_index", centered)

    def forward(self, gait_state: torch.Tensor) -> torch.Tensor:
        phase = gait_state[..., 0:1]
        amplitude = gait_state[..., 2:3]
        phase_lag = gait_state[..., 3:4]
        offset = gait_state[..., 4:5]
        turn_bias = gait_state[..., 5:6]
        spatial_phase = phase + self.joint_index * phase_lag
        wave = amplitude * torch.sin(spatial_phase)
        # Pure CPG fallback:
        # raw_action = wave + offset + turn_bias * self.centered_joint_index
        raw_action = wave + offset + turn_bias * self.centered_joint_index
        if gait_state.shape[-1] >= 11:
            coeffs = gait_state[..., 6:11]
            second_phase = phase + 2.0 * self.joint_index * phase_lag
            residual = coeffs[..., 0:1] * torch.cos(spatial_phase)
            residual = residual + coeffs[..., 1:2] * torch.sin(second_phase)
            residual = residual + coeffs[..., 2:3] * torch.cos(second_phase)
            residual = residual + coeffs[..., 3:4] * self.centered_joint_index * torch.sin(spatial_phase)
            residual = residual + coeffs[..., 4:5] * self.centered_joint_index * torch.cos(spatial_phase)
            raw_action = raw_action + residual
        return self.action_limit * torch.tanh(raw_action)

    def noise_basis(self, gait_state: torch.Tensor) -> torch.Tensor:
        phase = gait_state[..., 0:1]
        phase_lag = gait_state[..., 3:4]
        spatial_phase = phase + self.joint_index * phase_lag
        second_phase = phase + 2.0 * self.joint_index * phase_lag
        basis = torch.stack(
            (
                torch.ones_like(spatial_phase),
                self.centered_joint_index.expand_as(spatial_phase),
                torch.sin(spatial_phase),
                torch.cos(spatial_phase),
                torch.sin(second_phase),
                torch.cos(second_phase),
            ),
            dim=-1,
        )
        basis_norm = torch.sqrt(torch.mean(torch.square(basis), dim=-2, keepdim=True)).clamp_min(1.0e-6)
        return basis / basis_norm


class StructuredCpgActionDistribution:
    """Low-rank Gaussian whose exploration is confined to coordinated CPG wave modes."""

    def __init__(
        self,
        mean: torch.Tensor,
        basis: torch.Tensor,
        latent_std: torch.Tensor,
        residual_std: float,
    ):
        latent_std = latent_std.to(device=mean.device, dtype=mean.dtype)
        cov_factor = basis * latent_std.view(*((1,) * (basis.dim() - 1)), -1)
        cov_diag = torch.full_like(mean, float(residual_std) ** 2)
        self._distribution = LowRankMultivariateNormal(mean, cov_factor, cov_diag)

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean

    @property
    def stddev(self) -> torch.Tensor:
        return self._distribution.stddev

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy()


class CpgMemory(torch.nn.Module):
    def __init__(self, rnn: GaitCpgRNN, batch_size: int):
        super().__init__()
        self.rnn = rnn
        h, c = self.rnn.initial_state(batch_size, device="cpu", dtype=torch.float32)
        self.register_buffer("_hidden_h", h, persistent=False)
        self.register_buffer("_hidden_c", c, persistent=False)

    @property
    def hidden_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._hidden_h, self._hidden_c

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            h, c = self.rnn.initial_state(self._hidden_h.shape[1], self._hidden_h.device, self._hidden_h.dtype)
            self._hidden_h.copy_(h)
            self._hidden_c.copy_(c)
            return
        dones = dones.to(device=self._hidden_h.device, dtype=torch.bool).view(-1)
        if dones.any():
            h, c = self.rnn.initial_state(int(dones.sum().item()), self._hidden_h.device, self._hidden_h.dtype)
            self._hidden_h[:, dones, :] = h
            self._hidden_c[:, dones, :] = c

    def _apply_masks(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        hidden_states: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = hidden_states
        outputs = []
        for step in range(x.shape[0]):
            mask = masks[step].to(device=x.device, dtype=torch.bool).view(-1)
            if not mask.all():
                reset_h, reset_c = self.rnn.initial_state(int((~mask).sum().item()), x.device, x.dtype)
                h = h.to(device=x.device, dtype=x.dtype).clone()
                c = c.to(device=x.device, dtype=x.dtype).clone()
                h[:, ~mask, :] = reset_h
                c[:, ~mask, :] = reset_c
            output, (h, c) = self.rnn(x[step : step + 1], (h, c))
            outputs.append(output.squeeze(0))
        padded_output = torch.stack(outputs, dim=0)
        return _unpad_trajectories(padded_output, masks), (h, c)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        live_mode = hidden_states is None
        if hidden_states is None:
            hidden_states = (self._hidden_h, self._hidden_c)
        elif isinstance(hidden_states, torch.Tensor):
            hidden_states = (hidden_states, torch.zeros_like(hidden_states))

        if masks is not None:
            output, next_hidden = self._apply_masks(x, masks, hidden_states)
        else:
            output, next_hidden = self.rnn(x, hidden_states)

        if live_mode:
            self._hidden_h.copy_(next_hidden[0].detach())
            self._hidden_c.copy_(next_hidden[1].detach())
        return output


class DummyMemory(torch.nn.Module):
    def __init__(self, batch_size: int):
        super().__init__()
        self.register_buffer("_hidden_h", torch.zeros(1, batch_size, 1), persistent=False)
        self.register_buffer("_hidden_c", torch.zeros(1, batch_size, 1), persistent=False)

    @property
    def hidden_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._hidden_h, self._hidden_c

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self._hidden_h.zero_()
            self._hidden_c.zero_()
            return
        dones = dones.to(device=self._hidden_h.device, dtype=torch.bool).view(-1)
        self._hidden_h[:, dones, :] = 0.0
        self._hidden_c[:, dones, :] = 0.0


class CpgActorCritic(torch.nn.Module):
    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        feedback_hidden_dims: Sequence[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        cpg_dt: float = 0.02,
        cpg_action_limit: float = 1.0,
        cpg_residual_scale: float = 0.0,
        cpg_command_start: int = 6,
        cpg_noise_mode: str = "structured",
        cpg_noise_residual_std: float = 0.02,
        cpg_noise_min_std: float = 0.03,
        cpg_noise_max_std: float = 0.35,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"CpgActorCritic.__init__ ignored unexpected arguments: {list(kwargs.keys())}")

        self.obs_groups = obs_groups
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = torch.nn.Identity()
        self.critic_obs_normalizer = torch.nn.Identity()

        num_actor_obs = self._group_dim(obs, obs_groups["policy"])
        num_critic_obs = self._group_dim(obs, obs_groups["critic"])
        batch_size = self._group_batch(obs, obs_groups["policy"])

        if feedback_hidden_dims is None:
            feedback_hidden_dims = actor_hidden_dims

        self.memory_a = CpgMemory(
            LSTM(
                input_size=num_actor_obs,
                dt=cpg_dt,
                feedback_hidden_dims=feedback_hidden_dims,
                activation=activation,
                residual_scale=cpg_residual_scale,
                command_start=cpg_command_start,
            ),
            batch_size=batch_size,
        )
        self.actor = GaitWaveReadout(num_actions, action_limit=cpg_action_limit)
        self.memory_c = DummyMemory(batch_size)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        self.noise_std_type = noise_std_type
        self.cpg_noise_mode = str(cpg_noise_mode)
        self.cpg_noise_residual_std = float(cpg_noise_residual_std)
        self.cpg_noise_min_std = float(cpg_noise_min_std)
        self.cpg_noise_max_std = float(cpg_noise_max_std)
        self.cpg_noise_dim = 6
        if self.cpg_noise_mode == "structured":
            initial_std = min(max(float(init_noise_std), self.cpg_noise_min_std), self.cpg_noise_max_std)
            self.log_std = torch.nn.Parameter(torch.log(initial_std * torch.ones(self.cpg_noise_dim)))
        elif self.noise_std_type == "scalar":
            self.std = torch.nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = torch.nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution: Normal | StructuredCpgActionDistribution | None = None
        Normal.set_default_validate_args(False)

    def _group_dim(self, obs, group_names: Sequence[str]) -> int:
        dim = 0
        for group_name in group_names:
            value = obs[group_name]
            if value.dim() != 2:
                raise ValueError("CpgActorCritic only supports flat observation groups.")
            dim += value.shape[-1]
        return int(dim)

    def _group_batch(self, obs, group_names: Sequence[str]) -> int:
        return int(obs[group_names[0]].shape[0])

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been updated yet.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been updated yet.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been updated yet.")
        entropy = self.distribution.entropy()
        if entropy.dim() == self.action_mean.dim():
            return entropy.sum(dim=-1)
        return entropy

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self, obs):
        return self.act_inference(obs)

    def update_distribution(self, gait_state: torch.Tensor) -> None:
        mean = self.actor(gait_state)
        if self.cpg_noise_mode == "structured":
            basis = self.actor.noise_basis(gait_state)
            latent_std = torch.exp(self.log_std).clamp(min=self.cpg_noise_min_std, max=self.cpg_noise_max_std)
            self.distribution = StructuredCpgActionDistribution(
                mean=mean,
                basis=basis,
                latent_std=latent_std,
                residual_std=self.cpg_noise_residual_std,
            )
        elif self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            self.distribution = Normal(mean, std)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
            self.distribution = Normal(mean, std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

    def act(self, obs, masks: torch.Tensor | None = None, hidden_states=None, **kwargs) -> torch.Tensor:
        if hidden_states is None:
            hidden_states = kwargs.get("hidden_state")
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        gait_state = self.memory_a(actor_obs, masks=masks, hidden_states=hidden_states)
        self.update_distribution(gait_state)
        return self.distribution.sample()

    def act_inference(self, obs) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        gait_state = self.memory_a(actor_obs)
        return self.actor(gait_state)

    def evaluate(self, obs, masks: torch.Tensor | None = None, hidden_states=None, **kwargs) -> torch.Tensor:
        if hidden_states is None:
            hidden_states = kwargs.get("hidden_state")
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        values = self.critic(critic_obs)
        if masks is not None and values.dim() == 3:
            values = _unpad_trajectories(values, masks)
        return values

    def get_actor_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[group_name] for group_name in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[group_name] for group_name in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been updated yet.")
        actions_log_prob = self.distribution.log_prob(actions)
        if actions_log_prob.dim() == actions.dim():
            return actions_log_prob.sum(dim=-1)
        return actions_log_prob

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def update_normalization(self, obs) -> None:
        return None
