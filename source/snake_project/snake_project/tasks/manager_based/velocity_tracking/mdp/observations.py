from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import observations as builtin_obs
from isaaclab.managers import SceneEntityCfg

from .virtual_chassis import compute_virtual_chassis_command_terms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def last_raw_actions(env: "ManagerBasedEnv", action_name: str = "joint_pos") -> torch.Tensor:
    return torch.nan_to_num(env.action_manager.get_term(action_name).raw_actions, nan=0.0, posinf=0.0, neginf=0.0)


def base_ang_vel(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.base_ang_vel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def projected_gravity(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.projected_gravity(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def joint_pos_rel(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.joint_pos_rel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def joint_vel_rel(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    return torch.nan_to_num(builtin_obs.joint_vel_rel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def generated_commands(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    return torch.nan_to_num(builtin_obs.generated_commands(env, command_name), nan=0.0, posinf=0.0, neginf=0.0)


def virtual_chassis_velocity(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]

    if (
        not torch.isfinite(body_pos_w).all()
        or not torch.isfinite(body_lin_vel_w).all()
        or not torch.isfinite(body_ang_vel_w).all()
    ):
        return torch.zeros((body_pos_w.shape[0], 3), device=body_pos_w.device)

    _, _, lin_vel_vc, ang_vel_z_vc = compute_virtual_chassis_command_terms(
        body_pos_w=body_pos_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    obs = torch.cat((lin_vel_vc[:, :2], ang_vel_z_vc.unsqueeze(1)), dim=1)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
