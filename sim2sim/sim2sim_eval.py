#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim2sim_eval.py
---------------
Batch velocity-tracking evaluation for the 14DOF-DW snake robot.

Sweeps a 5x5 grid of (vx, vy) commands over [-0.4, 0.4] with step 0.2,
runs 15 s per condition (first 3 s warm-up excluded from MAE), and produces:
  - summary_heatmap.png       (planar / vx / vy MAE)
  - summary_trajectory.png     (base_link + VC XY per condition)
  - summary_velocity_vx.png    (cmd vs VC vx per condition)
  - summary_velocity_vy.png    (cmd vs VC vy per condition)
  - data/eval_data.npz         (raw logs)
  - data/eval_mae.csv          (MAE table)

Plot style follows D:\\RL2\\Snake Plus\\03-Mujoco (blue base, red VC).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

try:
    import mujoco
except ImportError:
    raise ImportError("MuJoCo is required. Install with: pip install mujoco")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalCfg:
    sim_dt_train: float = 0.005
    decimation_train: int = 4
    policy_dt_train: float = 0.005 * 4

    action_scale: float = 0.25
    clip_actions: float = 100.0
    clip_observations: float = 100.0

    base_body_name: str = "base_link"
    controlled_joints: Tuple[str, ...] = (
        "yaw1", "yaw2", "yaw3", "yaw4", "yaw5", "yaw6", "yaw7",
    )
    virtual_chassis_bodies: Tuple[str, ...] = (
        "base_link",
        "link1", "link2", "link3", "link4", "link5", "link6", "link7",
        "link8", "link9", "link10", "link11", "link12", "link13", "link14",
    )
    default_joint_angles: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    device: str = "cpu"
    seed: int = 1

    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_wz: float = 0.0


# ---------------------------------------------------------------------------
# Math utils
# ---------------------------------------------------------------------------

def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    mat9 = np.zeros((9,), dtype=np.float64)
    mujoco.mju_quat2Mat(mat9, q_wxyz.astype(np.float64))
    return mat9.reshape(3, 3).copy()


def yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Virtual chassis (SVD-based, matching sim2sim_mujoco.py and Isaac Lab)
# ---------------------------------------------------------------------------

def compute_virtual_chassis_state(
    data, vc_body_ids: np.ndarray, prev_axes_w: np.ndarray, has_prev: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    body_pos_w = np.array(data.xpos[vc_body_ids], dtype=np.float64)
    body_com_w = np.array(data.xipos[vc_body_ids], dtype=np.float64)
    body_cvel = np.array(data.cvel[vc_body_ids], dtype=np.float64)
    body_ang_vel_w = body_cvel[:, 0:3]
    body_lin_vel_w = body_cvel[:, 3:6] + np.cross(body_ang_vel_w, body_pos_w - body_com_w)

    origin_w = body_pos_w.mean(axis=0)
    centered = body_pos_w - origin_w
    data_matrix = centered.T
    axes_w, _, _ = np.linalg.svd(data_matrix, full_matrices=False)

    if has_prev:
        dots = np.sum(axes_w * prev_axes_w, axis=0)
        signs = np.where(dots >= 0.0, 1.0, -1.0)
        axes_w = axes_w * signs
    else:
        head_to_tail_w = body_pos_w[-1] - body_pos_w[0]
        x_dot = float(np.dot(axes_w[:, 0], head_to_tail_w))
        if x_dot < 0.0:
            axes_w[:, 0] = -axes_w[:, 0]
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_dot = float(np.dot(axes_w[:, 2], world_up))
        if z_dot < 0.0:
            axes_w[:, 2] = -axes_w[:, 2]

    if np.linalg.det(axes_w) < 0.0:
        axes_w[:, 1] = -axes_w[:, 1]

    vc_lin_vel_w = body_lin_vel_w.mean(axis=0)
    vc_ang_vel_w = body_ang_vel_w.mean(axis=0)
    lin_vel_vc = axes_w.T @ vc_lin_vel_w
    ang_vel_vc = axes_w.T @ vc_ang_vel_w

    return origin_w, axes_w, lin_vel_vc, float(ang_vel_vc[2])


# ---------------------------------------------------------------------------
# Single-condition runner
# ---------------------------------------------------------------------------

class EvalRunner:
    def __init__(self, mjcf_path: str, policy_path: str, cfg: EvalCfg):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.mj_dt = float(self.model.opt.timestep)
        self.policy_dt = float(cfg.policy_dt_train)
        self.decimation = max(1, int(round(self.policy_dt / self.mj_dt)))

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body_name)
        if self.base_body_id < 0:
            raise ValueError(f"Base body '{cfg.base_body_name}' not found.")

        self.vc_body_ids = []
        for bn in cfg.virtual_chassis_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bn)
            if bid < 0:
                raise ValueError(f"VC body '{bn}' not found.")
            self.vc_body_ids.append(bid)
        self.vc_body_ids = np.array(self.vc_body_ids, dtype=np.int32)

        self.joint_ids = []
        self.qpos_adr = []
        self.qvel_adr = []
        for jn in cfg.controlled_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise ValueError(f"Joint '{jn}' not found.")
            self.joint_ids.append(jid)
            self.qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            self.qvel_adr.append(int(self.model.jnt_dofadr[jid]))
        self.joint_ids = np.array(self.joint_ids, dtype=np.int32)
        self.qpos_adr = np.array(self.qpos_adr, dtype=np.int32)
        self.qvel_adr = np.array(self.qvel_adr, dtype=np.int32)

        self.act_ids = self._map_actuators_to_joints(self.joint_ids)
        self.ctrl_low, self.ctrl_high = self._get_ctrl_ranges(self.act_ids)
        self.q_default = np.array(cfg.default_joint_angles, dtype=np.float32)

        self.policy = torch.jit.load(policy_path, map_location=cfg.device)
        self.policy.eval()

        self.last_actions = np.zeros((len(cfg.controlled_joints),), dtype=np.float32)

        self.command = np.array([cfg.cmd_vx, cfg.cmd_vy, cfg.cmd_wz, 0.0], dtype=np.float32)

        self._log_t: List[float] = []
        self._log_vc_vx: List[float] = []
        self._log_vc_vy: List[float] = []
        self._log_vc_wz: List[float] = []
        self._log_vc_x: List[float] = []
        self._log_vc_y: List[float] = []
        self._log_base_vx_w: List[float] = []
        self._log_base_vy_w: List[float] = []
        self._log_base_vz_w: List[float] = []
        self._log_base_x: List[float] = []
        self._log_base_y: List[float] = []

    def _map_actuators_to_joints(self, joint_ids: np.ndarray) -> np.ndarray:
        nu = self.model.nu
        if nu <= 0:
            raise RuntimeError("Model has no actuators.")
        trnid = np.array(self.model.actuator_trnid, dtype=np.int32)
        joint_to_act = {}
        for a in range(nu):
            j0 = int(trnid[a, 0])
            if j0 >= 0:
                joint_to_act[j0] = a
        act_ids = []
        for jid in joint_ids:
            jid_int = int(jid)
            if jid_int not in joint_to_act:
                raise RuntimeError(f"No actuator for joint id {jid_int}")
            act_ids.append(int(joint_to_act[jid_int]))
        return np.array(act_ids, dtype=np.int32)

    def _get_ctrl_ranges(self, act_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cr = np.array(self.model.actuator_ctrlrange, dtype=np.float32)
        low = cr[act_ids, 0].copy()
        high = cr[act_ids, 1].copy()
        for i in range(len(low)):
            if np.isclose(low[i], 0.0) and np.isclose(high[i], 0.0):
                low[i], high[i] = -1e9, 1e9
        return low, high

    def _get_base_kinematics(self):
        q_wxyz = np.array(self.data.xquat[self.base_body_id], dtype=np.float64)
        R_wb = quat_wxyz_to_rotmat(q_wxyz)
        v_world = np.array(self.data.qvel[0:3], dtype=np.float64)
        w_world = np.array(self.data.qvel[3:6], dtype=np.float64)
        w_body = (R_wb.T @ w_world).astype(np.float32)
        g_world_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        g_body = (R_wb.T @ g_world_dir).astype(np.float32)
        return w_body, g_body

    def _build_obs(self) -> np.ndarray:
        w_body, g_body = self._get_base_kinematics()
        q = np.array([self.data.qpos[a] for a in self.qpos_adr], dtype=np.float32)
        qd = np.array([self.data.qvel[a] for a in self.qvel_adr], dtype=np.float32)
        cmd3 = self.command[:3].astype(np.float32)
        obs = np.concatenate([
            w_body,
            g_body,
            cmd3,
            (q - self.q_default),
            qd,
            self.last_actions.astype(np.float32),
        ], axis=0)
        obs = np.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)
        return obs

    def _policy(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(obs).float().to(self.cfg.device).unsqueeze(0)
            a = self.policy(x)
            if isinstance(a, (tuple, list)):
                a = a[0]
            a = a.squeeze(0).detach().cpu().numpy().astype(np.float32)
        a = np.clip(a, -self.cfg.clip_actions, self.cfg.clip_actions)
        return a

    def _apply_position_targets(self, action: np.ndarray):
        q_target = self.q_default + self.cfg.action_scale * action
        q_target = np.clip(q_target, self.ctrl_low, self.ctrl_high).astype(np.float32)
        self.data.ctrl[self.act_ids] = q_target

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, adr in enumerate(self.qpos_adr):
            self.data.qpos[adr] = float(self.q_default[i])
        for adr in self.qvel_adr:
            self.data.qvel[adr] = 0.0
        self.last_actions[:] = 0.0
        self._log_t.clear()
        self._log_vc_vx.clear()
        self._log_vc_vy.clear()
        self._log_vc_wz.clear()
        self._log_vc_x.clear()
        self._log_vc_y.clear()
        self._log_base_vx_w.clear()
        self._log_base_vy_w.clear()
        self._log_base_vz_w.clear()
        self._log_base_x.clear()
        self._log_base_y.clear()
        mujoco.mj_forward(self.model, self.data)

    def step(self, sim_t: float):
        """Run one control step: observe -> policy -> actuate -> log -> physics."""
        self._log_step(sim_t)
        obs = self._build_obs()
        action = self._policy(obs)
        self.last_actions[:] = action
        self._apply_position_targets(action)

    def _log_step(self, t: float):
        origin_w, _, lin_vel_vc, ang_vel_z_vc = compute_virtual_chassis_state(
            self.data, self.vc_body_ids,
            np.zeros((3, 3), dtype=np.float64), False,
        )
        pos_w = np.array(self.data.xpos[self.base_body_id], dtype=np.float64)
        vel_w = np.array(self.data.qvel[0:3], dtype=np.float64)

        self._log_t.append(float(t))
        self._log_vc_vx.append(float(lin_vel_vc[0]))
        self._log_vc_vy.append(float(lin_vel_vc[1]))
        self._log_vc_wz.append(float(ang_vel_z_vc))
        self._log_vc_x.append(float(origin_w[0]))
        self._log_vc_y.append(float(origin_w[1]))
        self._log_base_vx_w.append(float(vel_w[0]))
        self._log_base_vy_w.append(float(vel_w[1]))
        self._log_base_vz_w.append(float(vel_w[2]))
        self._log_base_x.append(float(pos_w[0]))
        self._log_base_y.append(float(pos_w[1]))

    def run(self, seconds: float) -> Dict[str, np.ndarray]:
        self.reset()
        steps = int(seconds / self.mj_dt)
        sim_t = 0.0
        for k in range(steps):
            if k % self.decimation == 0:
                self._log_step(sim_t)
                if sim_t >= 0.5:
                    obs = self._build_obs()
                    action = self._policy(obs)
                    self.last_actions[:] = action
                    self._apply_position_targets(action)
            mujoco.mj_step(self.model, self.data)
            sim_t += self.mj_dt
        return self._logs_to_dict()

    def _logs_to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "t": np.array(self._log_t, dtype=np.float32),
            "vc_vx": np.array(self._log_vc_vx, dtype=np.float32),
            "vc_vy": np.array(self._log_vc_vy, dtype=np.float32),
            "vc_wz": np.array(self._log_vc_wz, dtype=np.float32),
            "vc_x": np.array(self._log_vc_x, dtype=np.float32),
            "vc_y": np.array(self._log_vc_y, dtype=np.float32),
            "base_vx_w": np.array(self._log_base_vx_w, dtype=np.float32),
            "base_vy_w": np.array(self._log_base_vy_w, dtype=np.float32),
            "base_vz_w": np.array(self._log_base_vz_w, dtype=np.float32),
            "base_x": np.array(self._log_base_x, dtype=np.float32),
            "base_y": np.array(self._log_base_y, dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# MAE computation
# ---------------------------------------------------------------------------

def compute_mae(
    cmd_vx: float,
    cmd_vy: float,
    data: Dict[str, np.ndarray],
    warmup: float,
) -> Tuple[float, float, float]:
    t = data["t"]
    mask = t >= warmup
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")
    vc_vx = data["vc_vx"][mask]
    vc_vy = data["vc_vy"][mask]
    mae_vx = float(np.mean(np.abs(cmd_vx - vc_vx)))
    mae_vy = float(np.mean(np.abs(cmd_vy - vc_vy)))
    vc_wz = data["vc_wz"][mask]
    mae_planar = float(np.mean(np.sqrt((cmd_vx - vc_vx) ** 2 + (cmd_vy - vc_vy) ** 2 + vc_wz ** 2)))
    return mae_planar, mae_vx, mae_vy


def compute_wz_mae(data: Dict[str, np.ndarray], warmup: float) -> float:
    t = data["t"]
    mask = t >= warmup
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(data["vc_wz"][mask])))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

BASE_COLOR = "#1f77b4"
VC_COLOR = "#d62728"
CMD_COLOR = "black"
GRID_ALPHA = 0.3
LINE_WIDTH = 2.0
CURVE_LINE_WIDTH = 1.5
VERTICAL_LINE_STYLE = (0, (8, 4))


def plot_summary_heatmap(
    vx_vals, vy_vals,
    mae_planar_grid, mae_vx_grid, mae_vy_grid, mae_wz_grid,
    output_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    titles = ["Planar MAE", "vx MAE (m/s)", "vy MAE (m/s)", "wz MAE (rad/s)"]
    grids = [mae_planar_grid, mae_vx_grid, mae_vy_grid, mae_wz_grid]
    axes = axes.flatten()
    nx, ny = len(vx_vals), len(vy_vals)

    for ax, title, grid in zip(axes, titles, grids):
        im = ax.imshow(grid, origin="lower", cmap="YlOrRd", aspect="auto",
                       extent=[vy_vals[0] - 0.1, vy_vals[-1] + 0.1,
                               vx_vals[0] - 0.1, vx_vals[-1] + 0.1])
        for i in range(nx):
            for j in range(ny):
                ax.text(vy_vals[j], vx_vals[i], f"{grid[i, j]:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if grid[i, j] < np.nanmean(grid) * 1.5 else "white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("vy command (m/s)")
        ax.set_ylabel("vx command (m/s)")
        ax.set_xticks(vy_vals)
        ax.set_yticks(vx_vals)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("Velocity Tracking MAE Summary", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved: {output_path}")


def plot_summary_trajectory(
    vx_vals, vy_vals,
    all_data: dict,
    mae_planar_grid,
    output_path: str,
):
    nx, ny = len(vx_vals), len(vy_vals)
    fig, axes = plt.subplots(nx, ny, figsize=(ny * 4, nx * 4))
    axes = np.atleast_2d(axes)

    for i, vx in enumerate(vx_vals):
        for j, vy in enumerate(vy_vals):
            ax = axes[i, j]
            key = f"vx{vx}_vy{vy}"
            if key not in all_data:
                ax.set_visible(False)
                continue
            d = all_data[key]
            mae = mae_planar_grid[i, j]
            ax.plot(d["base_x"], d["base_y"], color=BASE_COLOR, linewidth=LINE_WIDTH, label="base_link")
            ax.plot(d["vc_x"], d["vc_y"], color=VC_COLOR, linewidth=LINE_WIDTH, label="virtual_chassis")
            ax.scatter(d["base_x"][0], d["base_y"][0], color=BASE_COLOR, marker="o", s=30)
            ax.scatter(d["vc_x"][0], d["vc_y"][0], color=VC_COLOR, marker="o", s=30)
            ax.scatter(d["base_x"][-1], d["base_y"][-1], color=BASE_COLOR, marker="x", s=40)
            ax.scatter(d["vc_x"][-1], d["vc_y"][-1], color=VC_COLOR, marker="x", s=40)
            ax.set_title(f"vx={vx} vy={vy} | MAE={mae:.3f}", fontsize=8)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=GRID_ALPHA)
            if i == nx - 1:
                ax.set_xlabel("X (m)", fontsize=7)
            if j == 0:
                ax.set_ylabel("Y (m)", fontsize=7)
            ax.tick_params(labelsize=6)

    handles = [
        plt.Line2D([0], [0], color=BASE_COLOR, lw=2, label="base_link"),
        plt.Line2D([0], [0], color=VC_COLOR, lw=2, label="virtual_chassis"),
    ]
    fig.legend(handles, ["base_link", "virtual_chassis"], loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=9)
    fig.suptitle("XY Trajectory Summary (base_link + Virtual Chassis)", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved: {output_path}")


def plot_summary_velocity(
    vx_vals, vy_vals,
    all_data: dict,
    direction: str,
    output_path: str,
):
    nx, ny = len(vx_vals), len(vy_vals)
    fig, axes = plt.subplots(nx, ny, figsize=(ny * 4, nx * 3.5))
    axes = np.atleast_2d(axes)

    for i, vx in enumerate(vx_vals):
        for j, vy in enumerate(vy_vals):
            ax = axes[i, j]
            key = f"vx{vx}_vy{vy}"
            if key not in all_data:
                ax.set_visible(False)
                continue
            d = all_data[key]
            t = d["t"]
            if direction == "vx":
                ax.plot(t, np.full_like(t, vx), color=CMD_COLOR, linestyle="--",
                        linewidth=CURVE_LINE_WIDTH, label="cmd")
                ax.plot(t, d["vc_vx"], color=VC_COLOR, linestyle="-",
                        linewidth=CURVE_LINE_WIDTH, label="vc")
                ax.set_ylabel("vx (m/s)", fontsize=7)
            elif direction == "vy":
                ax.plot(t, np.full_like(t, vy), color=CMD_COLOR, linestyle="--",
                        linewidth=CURVE_LINE_WIDTH, label="cmd")
                ax.plot(t, d["vc_vy"], color=VC_COLOR, linestyle="-",
                        linewidth=CURVE_LINE_WIDTH, label="vc")
                ax.set_ylabel("vy (m/s)", fontsize=7)
            else:  # wz
                ax.plot(t, np.zeros_like(t), color=CMD_COLOR, linestyle="--",
                        linewidth=CURVE_LINE_WIDTH, label="cmd")
                ax.plot(t, d["vc_wz"], color=VC_COLOR, linestyle="-",
                        linewidth=CURVE_LINE_WIDTH, label="vc")
                ax.set_ylabel("wz (rad/s)", fontsize=7)
            ax.set_title(f"vx={vx} vy={vy}", fontsize=8)
            ax.grid(True, alpha=GRID_ALPHA)
            ax.tick_params(labelsize=6)
            ax.axvline(x=3.0, color="gray", linestyle=VERTICAL_LINE_STYLE, alpha=0.5, linewidth=0.8)
            if i == nx - 1:
                ax.set_xlabel("Time (s)", fontsize=7)

    fig.suptitle(f"Velocity Tracking Summary — {direction.upper()}", fontweight="bold", fontsize=13)
    handles = [
        plt.Line2D([0], [0], color=CMD_COLOR, linestyle="--", lw=1.5, label="command"),
        plt.Line2D([0], [0], color=VC_COLOR, linestyle="-", lw=1.5, label="virtual_chassis"),
    ]
    fig.legend(handles, ["command", "virtual_chassis"], loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main batch evaluation
# ---------------------------------------------------------------------------

def run_eval(args):
    vx_vals = np.arange(-0.4, 0.41, 0.2)
    vy_vals = np.arange(-0.2, 0.21, 0.1)
    nx, ny = len(vx_vals), len(vy_vals)

    mae_planar_grid = np.full((nx, ny), np.nan)
    mae_vx_grid = np.full((nx, ny), np.nan)
    mae_vy_grid = np.full((nx, ny), np.nan)
    mae_wz_grid = np.full((nx, ny), np.nan)
    all_data: dict = {}
    mae_table: list = []

    total = nx * ny
    count = 0
    for i, vx in enumerate(vx_vals):
        for j, vy in enumerate(vy_vals):
            count += 1
            cfg = EvalCfg(
                device=args.device,
                seed=args.seed,
                cmd_vx=float(vx),
                cmd_vy=float(vy),
                cmd_wz=0.0,
            )
            runner = EvalRunner(args.mjcf, args.policy, cfg)
            print(f"[{count}/{total}] Running vx={vx:+.1f} vy={vy:+.1f} ... ", end="", flush=True)
            data = runner.run(seconds=args.seconds)
            mae_planar, mae_vx, mae_vy = compute_mae(float(vx), float(vy), data, warmup=args.warmup)
            mae_wz = compute_wz_mae(data, warmup=args.warmup)
            print(f"planar_MAE={mae_planar:.4f}")
            key = f"vx{vx}_vy{vy}"
            all_data[key] = data
            mae_planar_grid[i, j] = mae_planar
            mae_vx_grid[i, j] = mae_vx
            mae_vy_grid[i, j] = mae_vy
            mae_wz_grid[i, j] = mae_wz
            mae_table.append({
                "vx": float(vx), "vy": float(vy),
                "planar_mae": mae_planar, "vx_mae": mae_vx, "vy_mae": mae_vy, "wz_mae": mae_wz,
            })

    # --- output plots ---
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    plot_summary_heatmap(vx_vals, vy_vals, mae_planar_grid, mae_vx_grid, mae_vy_grid, mae_wz_grid,
                         os.path.join(out_dir, "summary_heatmap.png"))
    plot_summary_trajectory(vx_vals, vy_vals, all_data, mae_planar_grid,
                            os.path.join(out_dir, "summary_trajectory.png"))
    plot_summary_velocity(vx_vals, vy_vals, all_data, "vx",
                          os.path.join(out_dir, "summary_velocity_vx.png"))
    plot_summary_velocity(vx_vals, vy_vals, all_data, "vy",
                          os.path.join(out_dir, "summary_velocity_vy.png"))
    plot_summary_velocity(vx_vals, vy_vals, all_data, "wz",
                          os.path.join(out_dir, "summary_velocity_wz.png"))

    # --- data export ---
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    np.savez_compressed(os.path.join(data_dir, "eval_data.npz"), **all_data)

    csv_path = os.path.join(data_dir, "eval_mae.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vx", "vy", "planar_mae", "vx_mae", "vy_mae", "wz_mae"])
        writer.writeheader()
        writer.writerows(mae_table)
    print(f"[Data] Saved: {csv_path}")

    # --- console summary ---
    print(f"\n{'='*55}")
    print(f"Evaluation complete — {count}/{total} conditions")
    print(f"Warm-up: {args.warmup}s excluded")
    best_idx = np.nanargmin(mae_planar_grid)
    best_i, best_j = np.unravel_index(best_idx, (nx, ny))
    print(f"Best planar MAE: {mae_planar_grid[best_i, best_j]:.4f} @ vx={vx_vals[best_i]:+.1f} vy={vy_vals[best_j]:+.1f}")
    worst_idx = np.nanargmax(mae_planar_grid)
    worst_i, worst_j = np.unravel_index(worst_idx, (nx, ny))
    print(f"Worst planar MAE: {mae_planar_grid[worst_i, worst_j]:.4f} @ vx={vx_vals[worst_i]:+.1f} vy={vy_vals[worst_j]:+.1f}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Batch sim2sim velocity-tracking evaluation")
    ap.add_argument("--policy", type=str, required=True,
                    help="Path to TorchScript policy (jit .pt)")
    ap.add_argument("--mjcf", type=str,
                    default="source/snake_project/snake_project/data/Snake/14DOF-DW.xml",
                    help="Path to MJCF model XML")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="Simulation duration per condition (s)")
    ap.add_argument("--warmup", type=float, default=3.0,
                    help="Warm-up duration excluded from MAE (s)")
    ap.add_argument("--output-dir", type=str,
                    default="sim2sim/eval_output",
                    help="Output directory for plots and data")
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[Eval] Policy : {args.policy}")
    print(f"[Eval] MJCF   : {args.mjcf}")
    print(f"[Eval] Device : {args.device}")
    print(f"[Eval] Grid   : 5x5  (vx ∈ [-0.4, -0.2, 0.0, 0.2, 0.4], vy ∈ [-0.2, -0.1, 0.0, 0.1, 0.2])")
    print(f"[Eval] Each   : {args.seconds}s  (warm-up {args.warmup}s excluded)")
    run_eval(args)


if __name__ == "__main__":
    main()
