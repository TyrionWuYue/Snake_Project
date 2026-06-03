from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .virtual_chassis import compute_virtual_chassis_command_terms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(num_envs: int, device: torch.device | str, env_ids) -> torch.Tensor | None:
    if env_ids is None or isinstance(env_ids, slice):
        return None
    if isinstance(env_ids, torch.Tensor):
        return env_ids
    return torch.tensor(env_ids, device=device, dtype=torch.long)


def joint_amplitude(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward sustained motion by averaging absolute active-joint velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def motion_coordination(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize all active joints bending or moving in the same direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    pos_sign_mean = torch.abs(torch.mean(torch.sign(joint_pos), dim=1))
    vel_sign_mean = torch.abs(torch.mean(torch.sign(joint_vel), dim=1))
    return (pos_sign_mean + vel_sign_mean) / 2.0


def phase_propagation(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward alternating velocity directions between adjacent controlled joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    if joint_vel.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)

    vel_product = joint_vel[:, :-1] * joint_vel[:, 1:]
    vel_mag = torch.abs(joint_vel[:, :-1]) * torch.abs(joint_vel[:, 1:]) + 1.0e-6
    normalized_product = vel_product / vel_mag
    return -torch.mean(normalized_product, dim=1)


class RawActionRatePenalty(ManagerTermBase):
    """L2 penalty on the first-order raw action-rate."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.prev_raw_action = None
        params = getattr(cfg, "params", {}) or {}
        self.rate_clip = float(params.get("rate_clip", 10.0))

    def reset(self, env_ids=None) -> dict[str, float]:
        if self.prev_raw_action is None:
            return {}
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_raw_action.zero_()
        else:
            self.prev_raw_action[env_ids] = 0.0
        return {}

    def __call__(self, env: "ManagerBasedRLEnv", action_term_name: str = "joint_pos") -> torch.Tensor:
        action_term = env.action_manager.get_term(action_term_name)
        raw_action = action_term.raw_actions
        if self.prev_raw_action is None:
            self.prev_raw_action = torch.zeros_like(raw_action)
        delta = raw_action - self.prev_raw_action
        if self.rate_clip > 0.0:
            delta = torch.clamp(delta, min=-self.rate_clip, max=self.rate_clip)
        self.prev_raw_action.copy_(raw_action)
        return torch.sum(torch.square(delta), dim=1)


class ActionWaveFitPenalty(ManagerTermBase):
    """Penalty for action components outside a low-order spatial wave."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.projection = None
        self.action_dim = 0

    def _build_projection(self, action_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        phase = torch.arange(action_dim, device=device, dtype=dtype) * (2.0 * torch.pi / action_dim)
        basis = torch.stack(
            (torch.ones_like(phase), torch.sin(phase), torch.cos(phase), torch.sin(2.0 * phase), torch.cos(2.0 * phase)),
            dim=1,
        )
        return basis @ torch.linalg.pinv(basis)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        action_term_name: str = "joint_pos",
        penalty_clip: float = 1.0,
    ) -> torch.Tensor:
        raw_action = env.action_manager.get_term(action_term_name).raw_actions
        raw_action = torch.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        action_dim = raw_action.shape[1]

        if action_dim < 3:
            return torch.zeros(raw_action.shape[0], device=raw_action.device)

        if (
            self.projection is None
            or self.action_dim != action_dim
            or self.projection.device != raw_action.device
            or self.projection.dtype != raw_action.dtype
        ):
            self.projection = self._build_projection(action_dim, raw_action.device, raw_action.dtype)
            self.action_dim = action_dim

        wave_action = raw_action @ self.projection.T
        residual = raw_action - wave_action
        penalty = torch.mean(torch.square(residual), dim=1)
        if penalty_clip > 0.0:
            penalty = torch.clamp(penalty, max=penalty_clip)
        return penalty


class VirtualChassisTrackLinVelXYExp(ManagerTermBase):
    """Reward planar command tracking in the virtual chassis frame."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        linear_coef: float = 0.0,
        decompose_command: bool = False,
        parallel_std: float | None = None,
        perp_std: float | None = None,
        command_deadband: float = 0.03,
        reverse_coef: float = 0.0,
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not torch.isfinite(body_pos_w).all():
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, actual_lin_vel_vc, _ = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_lin_vel_vc).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        command_xy = env.command_manager.get_command(command_name)[:, :2]
        actual_xy = actual_lin_vel_vc[:, :2]

        # legacy_l2_error fallback:
        # lin_vel_error = torch.sum(torch.square(command_xy - actual_xy), dim=1)
        # exp_reward = torch.exp(-lin_vel_error / std**2)
        # lin_penalty = linear_coef * torch.sqrt(lin_vel_error)
        # return exp_reward - lin_penalty
        if not decompose_command:
            lin_vel_error = torch.sum(torch.square(command_xy - actual_xy), dim=1)
            exp_reward = torch.exp(-lin_vel_error / std**2)
            lin_penalty = linear_coef * torch.sqrt(lin_vel_error)
            return exp_reward - lin_penalty

        command_norm = torch.norm(command_xy, dim=1)
        active_command = command_norm > command_deadband
        command_dir = command_xy / torch.clamp(command_norm.unsqueeze(1), min=1.0e-6)
        actual_parallel = torch.sum(actual_xy * command_dir, dim=1)
        actual_perp = actual_xy - actual_parallel.unsqueeze(1) * command_dir
        actual_perp_speed = torch.norm(actual_perp, dim=1)

        target_parallel = torch.where(active_command, command_norm, torch.zeros_like(command_norm))
        parallel_error = actual_parallel - target_parallel
        zero_command_speed = torch.norm(actual_xy, dim=1)

        parallel_sigma = std if parallel_std is None else parallel_std
        perp_sigma = std if perp_std is None else perp_std
        decomposed_error = torch.square(parallel_error) / parallel_sigma**2
        decomposed_error = decomposed_error + torch.square(actual_perp_speed) / perp_sigma**2
        exp_reward = torch.exp(-decomposed_error)

        tracking_distance = torch.abs(parallel_error) + actual_perp_speed
        tracking_distance = torch.where(active_command, tracking_distance, zero_command_speed)
        reverse_penalty = reverse_coef * torch.relu(-actual_parallel)
        reverse_penalty = torch.where(active_command, reverse_penalty, torch.zeros_like(reverse_penalty))
        lin_penalty = linear_coef * tracking_distance
        return exp_reward - lin_penalty - reverse_penalty


class VirtualChassisCommandDirectionReward(ManagerTermBase):
    """Reward virtual-chassis velocity projected onto the commanded planar direction."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_deadband: float = 0.03,
        progress_clip: float = 1.0,
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not torch.isfinite(body_pos_w).all():
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, actual_lin_vel_vc, _ = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_lin_vel_vc).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        command_xy = env.command_manager.get_command(command_name)[:, :2]
        command_norm = torch.norm(command_xy, dim=1)
        command_dir = command_xy / torch.clamp(command_norm.unsqueeze(1), min=1.0e-6)
        projected_speed = torch.sum(actual_lin_vel_vc[:, :2] * command_dir, dim=1)
        normalized_progress = projected_speed / torch.clamp(command_norm, min=1.0e-6)
        if progress_clip > 0.0:
            normalized_progress = torch.clamp(normalized_progress, min=-progress_clip, max=progress_clip)
        active_command = command_norm > command_deadband
        return torch.where(active_command, normalized_progress, torch.zeros_like(normalized_progress))


class VirtualChassisCommandProgressFloorPenalty(VirtualChassisCommandDirectionReward):
    """Penalty when commanded motion does not produce enough projected progress."""

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_deadband: float = 0.08,
        progress_clip: float = 1.0,
        progress_floor: float = 0.25,
    ) -> torch.Tensor:
        progress = super().__call__(
            env=env,
            command_name=command_name,
            asset_cfg=asset_cfg,
            command_deadband=command_deadband,
            progress_clip=progress_clip,
        )
        command_xy = env.command_manager.get_command(command_name)[:, :2]
        active_command = torch.norm(command_xy, dim=1) > command_deadband
        penalty = torch.relu(progress_floor - progress)
        return torch.where(active_command, penalty, torch.zeros_like(penalty))


class VirtualChassisTrackAngVelZExp(ManagerTermBase):
    """Reward yaw-rate command tracking around the virtual chassis z-axis."""

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_axes_w = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.has_prev_axes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> dict[str, float]:
        env_ids = _resolve_env_ids(self.num_envs, self.device, env_ids)
        if env_ids is None:
            self.prev_axes_w.zero_()
            self.has_prev_axes.zero_()
        else:
            self.prev_axes_w[env_ids] = 0.0
            self.has_prev_axes[env_ids] = False
        return {}

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        body_pos_w = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, :]
        body_lin_vel_w = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids, :]
        body_ang_vel_w = self.asset.data.body_ang_vel_w[:, self.asset_cfg.body_ids, :]

        if not torch.isfinite(body_pos_w).all():
            return torch.zeros(self.num_envs, device=self.device)

        _, axes_w, _, actual_ang_vel_z_vc = compute_virtual_chassis_command_terms(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            prev_axes_w=self.prev_axes_w,
            has_prev=self.has_prev_axes,
        )

        if not torch.isfinite(axes_w).all() or not torch.isfinite(actual_ang_vel_z_vc).all():
            return torch.zeros(self.num_envs, device=self.device)

        self.prev_axes_w.copy_(axes_w)
        self.has_prev_axes[:] = True

        ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - actual_ang_vel_z_vc)
        return torch.exp(-ang_vel_error / std**2)


def contact_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.0,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    forces_on_bodies = net_forces[:, sensor_cfg.body_ids, :]
    in_contact = torch.any(torch.norm(forces_on_bodies, dim=-1) > threshold, dim=1)
    return in_contact.float()
