#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mujoco_runner.py
----------------
Sim2Sim runner: Legged-Gym (snake) policy -> MuJoCo (MJCF: snake-advanced.xml)

Aligned with velocity_tracking (Play):
- Obs dim = 46
- Obs layout (snake_project velocity_tracking observations):
    [ base_ang_vel_body (3)
        projected_gravity_body (3)
        commands[:3] (3)
        14 servo (q - default_q) (14)
        14 servo qd (14)
        last_actions (9 CPG params) ]
- Action -> target position:
  PPO emits CPG parameters, then the CPG action layer expands them to 7 yaw targets:
  q_target = default_q + action_scale * cpg_yaw_action
- Control decimation:
  policy_dt = cfg.sim_dt_train * cfg.decimation_train = 0.005 * 4 = 0.02s (50 Hz)
  MuJoCo timestep from XML (snake-advanced.xml) is typically 0.002s => decimation ≈ 10

IMPORTANT (matches your play.py):
- Disable observation noise by default.
"""

import argparse
import time
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import torch
import mujoco

from cpg_action import CPG_ACTION_DIM, CpgActionDecoder

# fix seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    import mujoco.viewer
    HAS_VIEWER = True
except Exception:
    HAS_VIEWER = False


# ----------------------------- Config (velocity_tracking) -----------------------------

@dataclass
class LeggedGymLikeCfg:
    # training-side sim/control (velocity_tracking)
    sim_dt_train: float = 0.005
    decimation_train: int = 4
    policy_dt_train: float = 0.005 * 4  # 0.02s

    # obs scales (velocity_tracking uses raw values by default)
    obs_scale_ang_vel: float = 1.0
    obs_scale_dof_pos: float = 1.0
    obs_scale_dof_vel: float = 1.0
    clip_observations: float = 100.0

    # action scaling (snake_robot_config.py control)
    action_scale: float = 0.25
    cpg_action_dim: int = CPG_ACTION_DIM
    clip_actions: float = 100.0

    # commands (velocity_tracking)
    resampling_time: float = 10.0
    heading_command: bool = False
    cmd_range_lin_vel_x: Tuple[float, float] = (0.0, 0.6)
    cmd_range_lin_vel_y: Tuple[float, float] = (0.0, 0.0)
    cmd_range_ang_vel_yaw: Tuple[float, float] = (0.0, 0.0)
    cmd_range_heading: Tuple[float, float] = (0.0, 0.0)
    planar_zero_threshold: float = 0.0

    # commands_scale in velocity_tracking is identity
    cmd_scale_lin_vel: float = 1.0
    cmd_scale_ang_vel: float = 1.0

    # joints / base name
    base_body_name: str = "base_link"
    controlled_joints: Tuple[str, ...] = ("yaw1", "yaw2", "yaw3", "yaw4", "yaw5", "yaw6", "yaw7")
    servo_joints: Tuple[str, ...] = (
        "yaw1", "pitch1",
        "yaw2", "pitch2",
        "yaw3", "pitch3",
        "yaw4", "pitch4",
        "yaw5", "pitch5",
        "yaw6", "pitch6",
        "yaw7", "pitch7",
    )
    virtual_chassis_bodies: Tuple[str, ...] = (
        "base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "link8",
        "link9",
        "link10",
        "link11",
        "link12",
        "link13",
        "link14",
    )

    # init default_joint_angles: all zeros in your cfg
    default_joint_angles: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    default_servo_joint_angles: Tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )

    # runtime
    device: str = "cpu"
    seed: int = 1

    # whether to add obs noise (play.py sets False)
    add_noise: bool = False  # keep False for inference

    # fixed command (optional): if set, use this instead of random sampling
    # format: (vx, vy, wz_or_heading) or None for random sampling
    fixed_command: Tuple[float, float, float] = None


# ----------------------------- Math utils -----------------------------

def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion is (w, x, y, z). Returns R_world_from_body."""
    mat9 = np.zeros((9,), dtype=np.float64)
    mujoco.mju_quat2Mat(mat9, q_wxyz.astype(np.float64))
    return mat9.reshape(3, 3).copy()


def yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ----------------------------- Command sampler (matches snake_robot.py logic) -----------------------------

class CommandSampler:
    """
    commands = [vx, vy, wz, heading]
    if heading_command=True:
      wz = clip(0.5 * wrap_to_pi(heading_target - yaw), -1, 1)
    
    If cfg.fixed_command is set, use fixed values instead of random sampling.
    """
    def __init__(self, cfg: LeggedGymLikeCfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.commands = np.zeros((4,), dtype=np.float32)
        self.t_left = 0.0
        self.use_fixed = cfg.fixed_command is not None
        
        if self.use_fixed:
            # fixed_command format: (vx, vy, wz_or_heading)
            self.commands[0] = float(cfg.fixed_command[0])  # vx
            self.commands[1] = float(cfg.fixed_command[1])  # vy
            if cfg.heading_command:
                self.commands[3] = float(cfg.fixed_command[2])  # heading target
            else:
                self.commands[2] = float(cfg.fixed_command[2])  # wz
            print(f"[CommandSampler] Using fixed command: vx={self.commands[0]:.3f}, vy={self.commands[1]:.3f}, " +
                  (f"heading={self.commands[3]:.3f}" if cfg.heading_command else f"wz={self.commands[2]:.3f}"))

    def reset(self):
        if not self.use_fixed:
            self.commands[:] = 0.0
            self.t_left = 0.0

    def _resample(self):
        if self.use_fixed:
            # Don't resample when using fixed command
            return
        self.commands[0] = self.rng.uniform(*self.cfg.cmd_range_lin_vel_x)
        self.commands[1] = self.rng.uniform(*self.cfg.cmd_range_lin_vel_y)
        if self.cfg.heading_command:
            self.commands[3] = self.rng.uniform(*self.cfg.cmd_range_heading)
        else:
            self.commands[2] = self.rng.uniform(*self.cfg.cmd_range_ang_vel_yaw)
        self.t_left = self.cfg.resampling_time

    def step(self, dt: float, yaw: float) -> np.ndarray:
        if not self.use_fixed:
            if self.t_left <= 0.0:
                self._resample()
            else:
                self.t_left -= dt

        if self.cfg.heading_command:
            target_heading = float(self.commands[3])
            wz = 0.5 * wrap_to_pi(target_heading - yaw)
            self.commands[2] = float(np.clip(wz, -1.0, 1.0))

        planar_norm = float(np.linalg.norm(self.commands[:2]))
        if planar_norm <= self.cfg.planar_zero_threshold:
            self.commands[0] = 0.0
            self.commands[1] = 0.0

        return self.commands.copy()


# ----------------------------- Runner -----------------------------

class MujocoSim2SimRunner:
    def __init__(self, mjcf_path: str, policy_path: str, cfg: LeggedGymLikeCfg, headless: bool):
        self.cfg = cfg
        self.headless = headless

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Load MuJoCo model/data
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # MuJoCo sim dt from XML
        self.mj_dt = float(self.model.opt.timestep)

        # Policy dt should match training policy_dt_train (0.02s)
        self.policy_dt = float(cfg.policy_dt_train)

        # Compute MuJoCo decimation to match policy dt
        self.decimation = max(1, int(round(self.policy_dt / self.mj_dt)))

        # Base body
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body_name)
        if self.base_body_id < 0:
            raise ValueError(f"Base body '{cfg.base_body_name}' not found in MJCF.")

        self.vc_body_ids = []
        for body_name in cfg.virtual_chassis_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Virtual chassis body '{body_name}' not found in MJCF.")
            self.vc_body_ids.append(body_id)
        self.vc_body_ids = np.array(self.vc_body_ids, dtype=np.int32)
        self.prev_vc_axes_w = np.zeros((3, 3), dtype=np.float64)
        self.has_prev_vc_axes = False

        # Joint indices: qpos/qvel addresses
        self.joint_ids = []
        self.qpos_adr = []
        self.qvel_adr = []
        for jn in cfg.controlled_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise ValueError(f"Joint '{jn}' not found in MJCF.")
            self.joint_ids.append(jid)
            self.qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            self.qvel_adr.append(int(self.model.jnt_dofadr[jid]))
        self.joint_ids = np.array(self.joint_ids, dtype=np.int32)
        self.qpos_adr = np.array(self.qpos_adr, dtype=np.int32)
        self.qvel_adr = np.array(self.qvel_adr, dtype=np.int32)

        self.servo_joint_ids = []
        self.servo_qpos_adr = []
        self.servo_qvel_adr = []
        for jn in cfg.servo_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise ValueError(f"Servo joint '{jn}' not found in MJCF.")
            self.servo_joint_ids.append(jid)
            self.servo_qpos_adr.append(int(self.model.jnt_qposadr[jid]))
            self.servo_qvel_adr.append(int(self.model.jnt_dofadr[jid]))
        self.servo_joint_ids = np.array(self.servo_joint_ids, dtype=np.int32)
        self.servo_qpos_adr = np.array(self.servo_qpos_adr, dtype=np.int32)
        self.servo_qvel_adr = np.array(self.servo_qvel_adr, dtype=np.int32)
        # Map actuators to joints (your MJCF uses <position joint="jointX">)
        self.act_ids = self._map_actuators_to_joints(self.joint_ids)
        self.ctrl_low, self.ctrl_high = self._get_ctrl_ranges(self.act_ids)

        # Default joint angles (training config uses all zeros)
        self.q_default = np.array(cfg.default_joint_angles, dtype=np.float32)
        self.servo_q_default = np.array(cfg.default_servo_joint_angles, dtype=np.float32)

        # Load TorchScript policy
        self.policy = torch.jit.load(policy_path, map_location=cfg.device)
        self.policy.eval()

        # Runtime buffers
        self.last_actions = np.zeros((cfg.cpg_action_dim,), dtype=np.float32)
        self.cpg_decoder = CpgActionDecoder()
        self.cmd_sampler = CommandSampler(cfg)

        # Viewer
        self.viewer = None
        
        # Logs for tracking evaluation (velocity tracking)
        self._log_t: List[float] = []
        self._log_cmd_vx: List[float] = []
        self._log_cmd_vy: List[float] = []
        self._log_cmd_wz: List[float] = []
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

        print("[Runner]")
        print(f"  MuJoCo dt        = {self.mj_dt:.6f}s")
        print(f"  Train policy dt  = {self.policy_dt:.6f}s (from 0.005*4)")
        print(f"  Using decimation = {self.decimation} (policy rate ~ {1.0/(self.decimation*self.mj_dt):.2f} Hz)")
        print(f"  Actuator mapping = {list(zip(cfg.controlled_joints, self.act_ids.tolist()))}")

    def _map_actuators_to_joints(self, joint_ids: np.ndarray) -> np.ndarray:
        nu = self.model.nu
        if nu <= 0:
            raise RuntimeError("Model has no actuators (nu=0). Your MJCF should define 7 <position> actuators.")

        # actuator_trnid[a,0] is the joint id for joint transmissions
        trnid = np.array(self.model.actuator_trnid, dtype=np.int32)  # (nu,2)
        joint_to_act = {}
        for a in range(nu):
            j0 = int(trnid[a, 0])
            if j0 >= 0:
                joint_to_act[j0] = a

        act_ids = []
        missing = []
        for jid in joint_ids:
            jid_int = int(jid)
            if jid_int not in joint_to_act:
                missing.append(jid_int)
            else:
                act_ids.append(int(joint_to_act[jid_int]))

        if missing:
            names = []
            for mid in missing:
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, mid)
                names.append(nm if nm is not None else str(mid))
            raise RuntimeError(
                f"Cannot find actuators for joints: {names}. "
                f"Please ensure each joint1..joint7 has an actuator with joint transmission."
            )
        return np.array(act_ids, dtype=np.int32)

    def _get_ctrl_ranges(self, act_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cr = np.array(self.model.actuator_ctrlrange, dtype=np.float32)  # (nu,2)
        low = cr[act_ids, 0].copy()
        high = cr[act_ids, 1].copy()
        # if ctrlrange is unset (0,0), treat as unbounded
        for i in range(len(low)):
            if np.isclose(low[i], 0.0) and np.isclose(high[i], 0.0):
                low[i], high[i] = -1e9, 1e9
        return low, high

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # Set controlled joints to training default (all zeros)
        for i, adr in enumerate(self.qpos_adr):
            self.data.qpos[adr] = float(self.q_default[i])
        for adr in self.qvel_adr:
            self.data.qvel[adr] = 0.0

        self.last_actions[:] = 0.0
        self.cpg_decoder.reset()
        self.cmd_sampler.reset()
        
        # Clear logs
        self._log_t.clear()
        self._log_cmd_vx.clear()
        self._log_cmd_vy.clear()
        self._log_cmd_wz.clear()
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
        self.has_prev_vc_axes = False
        self.prev_vc_axes_w[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def _get_base_kinematics(self):
        q_wxyz = np.array(self.data.xquat[self.base_body_id], dtype=np.float64)
        R_wb = quat_wxyz_to_rotmat(q_wxyz)
        yaw = yaw_from_quat_wxyz(q_wxyz)

        # Your MJCF has a freejoint at root => qvel[0:3] linear vel world, qvel[3:6] ang vel world
        v_world = np.array(self.data.qvel[0:3], dtype=np.float64)
        w_world = np.array(self.data.qvel[3:6], dtype=np.float64)

        v_body = (R_wb.T @ v_world).astype(np.float32)
        w_body = (R_wb.T @ w_world).astype(np.float32)

        # projected gravity (body frame): R^T * [0,0,-1]
        g_world_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        g_body = (R_wb.T @ g_world_dir).astype(np.float32)

        return v_body, w_body, g_body, yaw

    def _get_base_world_state(self) -> tuple[np.ndarray, np.ndarray]:
        pos_w = np.array(self.data.xpos[self.base_body_id], dtype=np.float64)
        vel_w = np.array(self.data.qvel[0:3], dtype=np.float64)
        return pos_w, vel_w

    def _compute_virtual_chassis_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        body_pos_w = np.array(self.data.xpos[self.vc_body_ids], dtype=np.float64)
        body_com_w = np.array(self.data.xipos[self.vc_body_ids], dtype=np.float64)
        body_cvel = np.array(self.data.cvel[self.vc_body_ids], dtype=np.float64)
        body_ang_vel_w = body_cvel[:, 0:3]
        body_lin_vel_w = body_cvel[:, 3:6] + np.cross(body_ang_vel_w, body_pos_w - body_com_w)

        origin_w = body_pos_w.mean(axis=0)
        centered = body_pos_w - origin_w
        data_matrix = centered.T
        axes_w, _, _ = np.linalg.svd(data_matrix, full_matrices=False)

        if self.has_prev_vc_axes:
            dots = np.sum(axes_w * self.prev_vc_axes_w, axis=0)
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

        self.prev_vc_axes_w = axes_w
        self.has_prev_vc_axes = True

        return origin_w, lin_vel_vc, ang_vel_vc, float(ang_vel_vc[2])

    def _build_obs(self, commands4: np.ndarray) -> np.ndarray:
        _, w_body, g_body, _ = self._get_base_kinematics()

        q = np.array([self.data.qpos[a] for a in self.servo_qpos_adr], dtype=np.float32)
        qd = np.array([self.data.qvel[a] for a in self.servo_qvel_adr], dtype=np.float32)

        cmd3 = commands4[:3].astype(np.float32)
        cmd3_scaled = cmd3 * np.array(
            [self.cfg.cmd_scale_lin_vel, self.cfg.cmd_scale_lin_vel, self.cfg.cmd_scale_ang_vel],
            dtype=np.float32
        )

        obs = np.concatenate([
              w_body * self.cfg.obs_scale_ang_vel,                           # 3
              g_body,                                                        # 3
              cmd3_scaled,                                                   # 3
              (q - self.servo_q_default) * self.cfg.obs_scale_dof_pos,       # 14
              qd * self.cfg.obs_scale_dof_vel,                               # 14
              self.last_actions.astype(np.float32),                          # 9
        ], axis=0)

        # inference: match normalization clip
        obs = np.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)

        if obs.shape[0] != 46:
            raise RuntimeError(f"Obs dim mismatch: got {obs.shape[0]}, expected 46.")
        return obs

    def _policy(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(obs).float().to(self.cfg.device).unsqueeze(0)
            a = self.policy(x)
            if isinstance(a, (tuple, list)):
                a = a[0]
            a = a.squeeze(0).detach().cpu().numpy().astype(np.float32)

        a = np.clip(a, -self.cfg.clip_actions, self.cfg.clip_actions)
        if a.shape[0] != self.cfg.cpg_action_dim:
            raise RuntimeError(f"Action dim mismatch: got {a.shape[0]}, expected {self.cfg.cpg_action_dim}.")
        return a

    def _apply_default_position_targets(self):
        q_target = np.clip(self.q_default, self.ctrl_low, self.ctrl_high).astype(np.float32)
        self.data.ctrl[self.act_ids] = q_target

    def _apply_position_targets(self, action: np.ndarray, command: np.ndarray):
        """
        Your MJCF actuators are <position ...> so:
          data.ctrl[act_id] = q_target
        where q_target = q_default + action_scale * cpg_yaw_action
        then clamp to actuator ctrlrange.
        """
        yaw_action = self.cpg_decoder.decode(action, command, dt=self.mj_dt)
        q_target = self.q_default + self.cfg.action_scale * yaw_action
        q_target = np.clip(q_target, self.ctrl_low, self.ctrl_high).astype(np.float32)
        self.data.ctrl[self.act_ids] = q_target

    def _log_step(self, t: float, cmd4: np.ndarray):
        """Log command and measured base velocities for tracking plots."""
        origin_w, lin_vel_vc, _, ang_vel_z_vc = self._compute_virtual_chassis_state()
        pos_w, vel_w = self._get_base_world_state()
        self._log_t.append(float(t))
        self._log_cmd_vx.append(float(cmd4[0]))
        self._log_cmd_vy.append(float(cmd4[1]))
        self._log_cmd_wz.append(float(cmd4[2]))
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

    def plot_playback_tracking(self, plot_path: str, show: bool = True):
        if len(self._log_t) == 0:
            print("[Plot] No logs collected. Did you run the simulation?")
            return

        import matplotlib.pyplot as plt

        out_dir = os.path.dirname(plot_path) if plot_path else ""
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        t = np.array(self._log_t, dtype=np.float32)
        cmd_vx = np.array(self._log_cmd_vx, dtype=np.float32)
        cmd_vy = np.array(self._log_cmd_vy, dtype=np.float32)
        cmd_wz = np.array(self._log_cmd_wz, dtype=np.float32)
        vc_vx = np.array(self._log_vc_vx, dtype=np.float32)
        vc_vy = np.array(self._log_vc_vy, dtype=np.float32)
        vc_wz = np.array(self._log_vc_wz, dtype=np.float32)
        vc_x = np.array(self._log_vc_x, dtype=np.float32)
        vc_y = np.array(self._log_vc_y, dtype=np.float32)
        base_vx = np.array(self._log_base_vx_w, dtype=np.float32)
        base_vy = np.array(self._log_base_vy_w, dtype=np.float32)
        base_vz = np.array(self._log_base_vz_w, dtype=np.float32)
        base_x = np.array(self._log_base_x, dtype=np.float32)
        base_y = np.array(self._log_base_y, dtype=np.float32)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(t, cmd_vx, label="cmd_vx")
        axes[0].plot(t, vc_vx, label="vc_vx")
        axes[0].set_ylabel("vx")
        axes[1].plot(t, cmd_vy, label="cmd_vy")
        axes[1].plot(t, vc_vy, label="vc_vy")
        axes[1].set_ylabel("vy")
        axes[2].plot(t, cmd_wz, label="cmd_wz")
        axes[2].plot(t, vc_wz, label="vc_wz")
        axes[2].set_ylabel("wz")
        axes[2].set_xlabel("time (s)")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right")
        fig.tight_layout()

        vc_plot_path = plot_path if plot_path else "vc_velocity_tracking.png"
        fig.savefig(vc_plot_path, dpi=200)
        plt.close(fig)

        fig, axis = plt.subplots(1, 1, figsize=(6, 6))
        axis.plot(vc_x, vc_y, label="virtual_chassis")
        axis.scatter([vc_x[0]], [vc_y[0]], label="vc_start", marker="o", s=30)
        axis.scatter([vc_x[-1]], [vc_y[-1]], label="vc_end", marker="x", s=40)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        fig.tight_layout()
        vc_traj_path = os.path.join(out_dir, "virtual_chassis_trajectory_xy.png") if out_dir else "virtual_chassis_trajectory_xy.png"
        fig.savefig(vc_traj_path, dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(t, base_vx, label="base_vx_w")
        axes[0].set_ylabel("vx")
        axes[1].plot(t, base_vy, label="base_vy_w")
        axes[1].set_ylabel("vy")
        axes[2].plot(t, base_vz, label="base_vz_w")
        axes[2].set_ylabel("vz")
        axes[2].set_xlabel("time (s)")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right")
        fig.tight_layout()
        base_vel_path = os.path.join(out_dir, "base_link_velocity_world.png") if out_dir else "base_link_velocity_world.png"
        fig.savefig(base_vel_path, dpi=200)
        plt.close(fig)

        fig, axis = plt.subplots(1, 1, figsize=(6, 6))
        axis.plot(base_x, base_y, label="base_link")
        axis.scatter([base_x[0]], [base_y[0]], label="base_start", marker="o", s=30)
        axis.scatter([base_x[-1]], [base_y[-1]], label="base_end", marker="x", s=40)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        fig.tight_layout()
        base_traj_path = os.path.join(out_dir, "base_link_trajectory_xy.png") if out_dir else "base_link_trajectory_xy.png"
        fig.savefig(base_traj_path, dpi=200)
        plt.close(fig)

        print(f"[Plot] Saved: {vc_plot_path}")
        print(f"[Plot] Saved: {vc_traj_path}")
        print(f"[Plot] Saved: {base_vel_path}")
        print(f"[Plot] Saved: {base_traj_path}")
        if show:
            plt.show()

    """
    def run(self):
        self.reset()

        if (not self.headless) and HAS_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        steps = int(self.cfg.episode_seconds / self.dt)
        t0 = time.time()

        for k in range(steps):
            # current yaw for heading command
            _, _, _, yaw = self._get_base_kinematics()

            # update commands at sim dt
            cmd4 = self.cmd_sampler.step(self.dt, yaw)

            # policy step at decimation
            if k % self.cfg.decimation == 0:
                obs = self._build_obs(cmd4)
                action = self._policy(obs)
                self.last_actions[:] = action

            self._apply_position_targets(self.last_actions)

            mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

        if self.viewer is not None:
            self.viewer.close()

        wall = time.time() - t0
        print(f"[Done] Sim {self.cfg.episode_seconds:.2f}s, wall {wall:.2f}s, RTF={self.cfg.episode_seconds / max(wall,1e-6):.2f}x")
    """

    def run(self, seconds: float, realtime: bool = True, realtime_factor: float = 1.0, lead: float = 0.001):
        self.reset()

        if (not self.headless) and HAS_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        steps = int(seconds / self.mj_dt)
        t0 = time.time()

        # Real-time pacing: keep (simulated time) synced with wall-clock time.
        # - realtime=True: run at ~1.0x by sleeping when simulation is ahead.
        # - realtime_factor: 1.0 => real-time, 2.0 => 2x faster, 0.5 => half speed.
        # - lead: small margin to reduce sleep jitter.
        t_wall0 = time.perf_counter()
        sim_elapsed = 0.0

        sim_t = 0.0
        for k in range(steps):
            # commands updated every sim step, resampled every resampling_time
            _, _, _, yaw = self._get_base_kinematics()
            cmd4 = self.cmd_sampler.step(self.mj_dt, yaw)

            # policy update every decimation steps (deferred 3s for physics settling)
            if k % self.decimation == 0:
                if sim_t >= 0.5:
                    obs = self._build_obs(cmd4)
                    action = self._policy(obs)
                    self.last_actions[:] = action

            # apply target positions
            if sim_t >= 0.5:
                self._apply_position_targets(self.last_actions, cmd4)
            else:
                self._apply_default_position_targets()

            # log before stepping so velocities correspond to this control step
            self._log_step(sim_t, cmd4)

            mujoco.mj_step(self.model, self.data)

            sim_t += self.mj_dt

            if realtime:
                sim_elapsed += self.mj_dt
                target_wall = sim_elapsed / max(realtime_factor, 1e-6)
                wall_elapsed = time.perf_counter() - t_wall0
                sleep_s = target_wall - wall_elapsed - lead
                if sleep_s > 0:
                    time.sleep(sleep_s)

            if self.viewer is not None:
                self.viewer.sync()

        if self.viewer is not None:
            self.viewer.close()

        wall = time.time() - t0
        print(f"[Done] Sim {seconds:.2f}s, wall {wall:.2f}s, RTF={seconds / max(wall,1e-6):.2f}x")


# ----------------------------- CLI -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", type=str, required=False, default="source/snake_project/snake_project/data/Snake/14DOF-DW.xml", help="Path to MJCF (e.g. snake-advanced.xml)")
    ap.add_argument("--policy", type=str, required=False, default="logs/rsl_rl/snake_14dofdw_velocity_flat_tracking/2026-05-17_13-25-51/exported/policy.pt", help="Path to TorchScript policy (jit .pt)")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--headless", type=int, default=0, help="1=headless, 0=viewer")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--realtime", type=int, default=1, help="1=pace to real-time (default), 0=run as fast as possible")
    ap.add_argument("--rtf", type=float, default=1.0, help="Real-time factor when --realtime=1. 1.0=real-time, 2.0=2x, 0.5=0.5x")
    ap.add_argument("--lead", type=float, default=0.001, help="Sleep lead margin in seconds to reduce jitter (default 1ms)")
    ap.add_argument("--plot", type=int, default=1, help="1=save tracking plot at end, 0=disable")
    ap.add_argument("--plot_path", type=str, default="sim2sim/figures/velocity_tracking.png", help="Output path for the tracking plot")
    ap.add_argument("--plot_show", type=int, default=1, help="1=show matplotlib window, 0=only save")
    
    # Fixed command options (if not specified, use random sampling)
    ap.add_argument("--cmd_vx", type=float, default=0.0, help="Fixed vx command (m/s). If set, disables random sampling.")
    ap.add_argument("--cmd_vy", type=float, default=0.0, help="Fixed vy command (m/s). If set, disables random sampling.")
    ap.add_argument("--cmd_wz", type=float, default=0.0, help="Fixed wz command (rad/s) or heading target (rad) depending on heading_command. If set, disables random sampling.")
    
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Prepare fixed command if specified
    fixed_cmd = None
    if args.cmd_vx is not None or args.cmd_vy is not None or args.cmd_wz is not None:
        # If any command component is specified, build fixed_command tuple
        vx = args.cmd_vx if args.cmd_vx is not None else 0.0
        vy = args.cmd_vy if args.cmd_vy is not None else 0.0
        wz = args.cmd_wz if args.cmd_wz is not None else 0.0
        fixed_cmd = (vx, vy, wz)
        print(f"[Main] Using fixed command: vx={vx:.3f} m/s, vy={vy:.3f} m/s, wz/heading={wz:.3f}")
    else:
        print("[Main] Using random command sampling")
    
    cfg = LeggedGymLikeCfg(device=args.device, seed=args.seed, fixed_command=fixed_cmd)
    runner = MujocoSim2SimRunner(args.mjcf, args.policy, cfg, headless=bool(args.headless))
    runner.run(seconds=float(args.seconds), realtime=bool(args.realtime), realtime_factor=float(args.rtf), lead=float(args.lead))
    if int(args.plot) == 1:
        runner.plot_playback_tracking(plot_path=str(args.plot_path), show=bool(args.plot_show))

if __name__ == "__main__":
    main()
