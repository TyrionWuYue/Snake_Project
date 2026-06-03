from __future__ import annotations

import math
import re
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _bounded_sigmoid(raw: torch.Tensor, low: float, high: float, initial: float) -> torch.Tensor:
    """Map an unconstrained policy output to a positive bounded CPG parameter."""
    if high <= low:
        raise ValueError(f"Expected high > low, got low={low}, high={high}.")
    ratio = min(max((initial - low) / (high - low), 1.0e-4), 1.0 - 1.0e-4)
    bias = math.log(ratio / (1.0 - ratio))
    return low + (high - low) * torch.sigmoid(raw + bias)


class CpgJointPositionAction(ActionTerm):
    """Expand PPO CPG parameters into coordinated yaw position targets."""

    # [frequency, amplitude, phase_lag, offset, residual(5 low-order shape coefficients)]
    _ACTION_DIM = 9

    def __init__(self, cfg: "CpgJointPositionActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self._asset: Articulation = env.scene[self.cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints < 2:
            raise ValueError("CpgJointPositionAction expects at least two yaw joints.")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)

        self._scale = self._resolve_joint_values(self.cfg.scale, default=1.0)
        if self.cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._offset = self._resolve_joint_values(self.cfg.offset, default=0.0)
        self._clip_low, self._clip_high = self._resolve_clip()

        self._joint_index = torch.arange(self._num_joints, device=self.device, dtype=torch.float32).view(1, -1)
        self._centered_index = torch.linspace(-1.0, 1.0, self._num_joints, device=self.device).view(1, -1)

    @property
    def action_dim(self) -> int:
        return self._ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=self.cfg.raw_action_clip, neginf=-self.cfg.raw_action_clip)
        self._raw_actions[:] = torch.clamp(actions, -self.cfg.raw_action_clip, self.cfg.raw_action_clip)

    def apply_actions(self) -> None:
        target = self._expand_cpg_to_joint_targets()
        self._processed_actions[:] = target
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._offset[env_ids]
        self._phase[env_ids] = 0.0

    def _expand_cpg_to_joint_targets(self) -> torch.Tensor:
        raw = self._raw_actions
        frequency = _bounded_sigmoid(
            raw[:, 0:1], self.cfg.frequency_range[0], self.cfg.frequency_range[1], self.cfg.frequency_init
        )
        amplitude = _bounded_sigmoid(
            raw[:, 1:2], self.cfg.amplitude_range[0], self.cfg.amplitude_range[1], self.cfg.amplitude_init
        )
        phase_lag_mag = _bounded_sigmoid(
            raw[:, 2:3], self.cfg.phase_lag_range[0], self.cfg.phase_lag_range[1], self.cfg.phase_lag_init
        )

        command = self._get_command()
        if command is not None:
            vx = command[:, 0:1]
            lag_sign = torch.where(
                torch.abs(vx) > self.cfg.command_vx_deadband,
                torch.sign(vx),
                torch.ones_like(vx),
            )
            command_speed = torch.linalg.norm(command[:, 0:2], dim=1, keepdim=True)
            gate = torch.clamp(command_speed / max(self.cfg.command_speed_ref, 1.0e-6), 0.0, 1.0)
            floor = self.cfg.amplitude_range[0] + gate * (
                self.cfg.command_amplitude_floor - self.cfg.amplitude_range[0]
            )
            amplitude = torch.maximum(amplitude, floor)
        else:
            lag_sign = torch.ones_like(phase_lag_mag)
        phase_lag = lag_sign * phase_lag_mag

        offset = self.cfg.offset_limit * torch.tanh(raw[:, 3:4])
        residual = self.cfg.residual_scale * torch.tanh(raw[:, 4:9])

        self._phase = torch.remainder(self._phase + (2.0 * math.pi * self.cfg.cpg_dt) * frequency, 2.0 * math.pi)
        spatial_phase = self._phase + self._joint_index * phase_lag

        wave = amplitude * torch.sin(spatial_phase) + offset
        wave = wave + residual[:, 0:1] * torch.cos(spatial_phase)
        wave = wave + residual[:, 1:2] * torch.sin(2.0 * spatial_phase)
        wave = wave + residual[:, 2:3] * torch.cos(2.0 * spatial_phase)
        wave = wave + residual[:, 3:4] * self._centered_index
        wave = wave + residual[:, 4:5] * self._centered_index * torch.cos(spatial_phase)

        joint_action = torch.tanh(wave)
        target = self._offset + self._scale * joint_action
        return torch.clamp(target, min=self._clip_low, max=self._clip_high)

    def _get_command(self) -> torch.Tensor | None:
        if not self.cfg.command_name:
            return None
        try:
            return self._env.command_manager.get_command(self.cfg.command_name)
        except Exception:
            return None

    def _resolve_joint_values(self, value: float | dict[str, float], default: float) -> torch.Tensor:
        values = torch.full((self.num_envs, self._num_joints), float(default), device=self.device)
        if isinstance(value, (float, int)):
            values[:] = float(value)
            return values
        if isinstance(value, dict):
            for pattern, pattern_value in value.items():
                for joint_id, joint_name in enumerate(self._joint_names):
                    if re.fullmatch(pattern, joint_name):
                        values[:, joint_id] = float(pattern_value)
            return values
        raise ValueError(f"Unsupported joint value type: {type(value)}.")

    def _resolve_clip(self) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.full((self.num_envs, self._num_joints), -float("inf"), device=self.device)
        high = torch.full((self.num_envs, self._num_joints), float("inf"), device=self.device)
        if self.cfg.clip is None:
            return low, high
        if not isinstance(self.cfg.clip, dict):
            raise ValueError(f"Unsupported clip type: {type(self.cfg.clip)}.")
        for pattern, bounds in self.cfg.clip.items():
            for joint_id, joint_name in enumerate(self._joint_names):
                if re.fullmatch(pattern, joint_name):
                    low[:, joint_id] = float(bounds[0])
                    high[:, joint_id] = float(bounds[1])
        return low, high


@configclass
class CpgJointPositionActionCfg(ActionTermCfg):
    """CPG-parameter action term that expands a low-dimensional gait action to yaw joint targets."""

    class_type: type[ActionTerm] = CpgJointPositionAction
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 0.25
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False
    use_default_offset: bool = True

    raw_action_clip: float = 5.0
    cpg_dt: float = 0.005
    frequency_range: tuple[float, float] = (0.25, 2.0)
    frequency_init: float = 0.8
    amplitude_range: tuple[float, float] = (0.15, 1.4)
    amplitude_init: float = 0.85
    phase_lag_range: tuple[float, float] = (0.35, 1.35)
    phase_lag_init: float = 0.85
    offset_limit: float = 0.7
    residual_scale: float = 0.35
    command_name: str = "base_velocity"
    command_vx_deadband: float = 0.03
    command_speed_ref: float = 0.2
    command_amplitude_floor: float = 0.55
