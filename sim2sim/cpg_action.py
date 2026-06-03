from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CPG_ACTION_DIM = 9


@dataclass
class CpgActionDecoderCfg:
    raw_action_clip: float = 5.0
    frequency_range: tuple[float, float] = (0.25, 2.0)
    frequency_init: float = 0.8
    amplitude_range: tuple[float, float] = (0.15, 1.4)
    amplitude_init: float = 0.85
    phase_lag_range: tuple[float, float] = (0.35, 1.35)
    phase_lag_init: float = 0.85
    offset_limit: float = 0.7
    residual_scale: float = 0.35
    command_vx_deadband: float = 0.03
    command_speed_ref: float = 0.2
    command_amplitude_floor: float = 0.55
    num_joints: int = 7


def _bounded_sigmoid(raw: float, low: float, high: float, initial: float) -> float:
    ratio = min(max((initial - low) / (high - low), 1.0e-4), 1.0 - 1.0e-4)
    bias = math.log(ratio / (1.0 - ratio))
    return low + (high - low) / (1.0 + math.exp(-(raw + bias)))


class CpgActionDecoder:
    """NumPy mirror of mdp.CpgJointPositionAction for MuJoCo sim2sim."""

    def __init__(self, cfg: CpgActionDecoderCfg | None = None):
        self.cfg = cfg if cfg is not None else CpgActionDecoderCfg()
        self.phase = 0.0
        self.joint_index = np.arange(self.cfg.num_joints, dtype=np.float32)
        self.centered_index = np.linspace(-1.0, 1.0, self.cfg.num_joints, dtype=np.float32)

    def reset(self) -> None:
        self.phase = 0.0

    def decode(self, action: np.ndarray, command: np.ndarray, dt: float) -> np.ndarray:
        raw = np.nan_to_num(action.astype(np.float32), nan=0.0, posinf=self.cfg.raw_action_clip, neginf=-self.cfg.raw_action_clip)
        raw = np.clip(raw, -self.cfg.raw_action_clip, self.cfg.raw_action_clip)
        if raw.shape[0] != CPG_ACTION_DIM:
            raise RuntimeError(f"CPG action dim mismatch: got {raw.shape[0]}, expected {CPG_ACTION_DIM}.")

        frequency = _bounded_sigmoid(raw[0], *self.cfg.frequency_range, self.cfg.frequency_init)
        amplitude = _bounded_sigmoid(raw[1], *self.cfg.amplitude_range, self.cfg.amplitude_init)
        phase_lag_mag = _bounded_sigmoid(raw[2], *self.cfg.phase_lag_range, self.cfg.phase_lag_init)

        vx = float(command[0]) if command is not None else 0.0
        if abs(vx) > self.cfg.command_vx_deadband:
            lag_sign = 1.0 if vx > 0.0 else -1.0
        else:
            lag_sign = 1.0
        phase_lag = lag_sign * phase_lag_mag

        if command is not None:
            command_speed = float(np.linalg.norm(command[:2]))
            gate = np.clip(command_speed / max(self.cfg.command_speed_ref, 1.0e-6), 0.0, 1.0)
            floor = self.cfg.amplitude_range[0] + gate * (
                self.cfg.command_amplitude_floor - self.cfg.amplitude_range[0]
            )
            amplitude = max(amplitude, floor)

        offset = self.cfg.offset_limit * math.tanh(float(raw[3]))
        residual = self.cfg.residual_scale * np.tanh(raw[4:9])

        self.phase = (self.phase + (2.0 * math.pi * dt) * frequency) % (2.0 * math.pi)
        spatial_phase = self.phase + self.joint_index * phase_lag

        wave = amplitude * np.sin(spatial_phase) + offset
        wave = wave + residual[0] * np.cos(spatial_phase)
        wave = wave + residual[1] * np.sin(2.0 * spatial_phase)
        wave = wave + residual[2] * np.cos(2.0 * spatial_phase)
        wave = wave + residual[3] * self.centered_index
        wave = wave + residual[4] * self.centered_index * np.cos(spatial_phase)
        return np.tanh(wave).astype(np.float32)
