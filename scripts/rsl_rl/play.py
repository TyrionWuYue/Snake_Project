# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--manual_command", action="store_true", default=False, help="Override velocity commands each step.")
parser.add_argument("--cmd_vx", type=float, default=0.0, help="Manual command: linear velocity x.")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="Manual command: linear velocity y.")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="Manual command: angular velocity z.")
parser.add_argument("--plot", action="store_true", default=False, help="Save PNG plots after playback.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time

import numpy as np
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SOURCE = REPO_ROOT / "source" / "snake_project"
if str(LOCAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE))

import snake_project.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # fix all random seeds for reproducibility
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(agent_cfg.seed)
        torch.cuda.manual_seed_all(agent_cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    base_env = env.unwrapped
    manual_command_enabled = False
    command_term = None
    is_velocity_tracking = "VelocityTracking" in args_cli.task and "Residual" not in args_cli.task
    if is_velocity_tracking and (args_cli.manual_command or args_cli.plot):
        try:
            command_term = base_env.command_manager.get_term("base_velocity")
        except Exception as exc:
            if args_cli.manual_command:
                print(f"[WARN] Failed to enable manual command: {exc}")
    if args_cli.manual_command:
        if is_velocity_tracking and command_term is not None:
            manual_command_enabled = True
        else:
            print("[WARN] Manual command is only supported for VelocityTracking tasks.")

    record_plot = args_cli.plot
    vc_cmd_times = []
    vc_cmd_vx = []
    vc_cmd_vy = []
    vc_cmd_wz = []
    vc_act_vx = []
    vc_act_vy = []
    vc_act_wz = []
    vc_xy = []
    base_xy = []
    base_vel = []
    base_body_id = None
    if record_plot:
        try:
            robot = base_env.scene["robot"]
            base_body_ids, _ = robot.find_bodies(["base_link"], preserve_order=True)
            base_body_id = base_body_ids[0]
        except Exception as exc:
            print(f"[WARN] Plotting disabled (base_link access failed): {exc}")
            record_plot = False
    if record_plot and command_term is None:
        print("[WARN] Plotting disabled (base_velocity command term unavailable).")
        record_plot = False

    def apply_manual_command() -> None:
        if not manual_command_enabled:
            return
        env_ids = torch.arange(command_term.num_envs, device=command_term.device)
        command_term.vel_command_b[env_ids, 0] = args_cli.cmd_vx
        command_term.vel_command_b[env_ids, 1] = args_cli.cmd_vy
        command_term.vel_command_b[env_ids, 2] = args_cli.cmd_wz

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    max_steps = None
    if hasattr(base_env, "max_episode_length_s"):
        max_steps = int(base_env.max_episode_length_s / dt)
    elif hasattr(env_cfg, "episode_length_s"):
        max_steps = int(env_cfg.episode_length_s / dt)

    # reset environment
    apply_manual_command()
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        apply_manual_command()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if record_plot and command_term is not None:
                origin_w, _, lin_vel_vc, ang_vel_z_vc = command_term._compute_virtual_state()
                vc_cmd_times.append(timestep * dt)
                vc_cmd_vx.append(float(args_cli.cmd_vx))
                vc_cmd_vy.append(float(args_cli.cmd_vy))
                vc_cmd_wz.append(float(args_cli.cmd_wz))
                vc_act_vx.append(float(lin_vel_vc[0, 0].item()))
                vc_act_vy.append(float(lin_vel_vc[0, 1].item()))
                vc_act_wz.append(float(ang_vel_z_vc[0].item()))
                vc_xy.append((float(origin_w[0, 0].item()), float(origin_w[0, 1].item())))
                if base_body_id is not None:
                    robot = base_env.scene["robot"]
                    base_pos_w = robot.data.body_pos_w[0, base_body_id, :]
                    base_vel_w = robot.data.body_lin_vel_w[0, base_body_id, :]
                    base_xy.append((float(base_pos_w[0].item()), float(base_pos_w[1].item())))
                    base_vel.append(
                        (float(base_vel_w[0].item()), float(base_vel_w[1].item()), float(base_vel_w[2].item()))
                    )
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1
        if max_steps is not None and timestep >= max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    print(f"[INFO] Finished after {timestep} steps ({timestep * dt:.2f} seconds)")

    if record_plot and vc_cmd_times:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_dir = os.path.join(log_dir, "plots", "play")
            os.makedirs(plot_dir, exist_ok=True)

            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            axes[0].plot(vc_cmd_times, vc_cmd_vx, label="cmd_vx")
            axes[0].plot(vc_cmd_times, vc_act_vx, label="act_vx")
            axes[0].set_ylabel("vx")
            axes[1].plot(vc_cmd_times, vc_cmd_vy, label="cmd_vy")
            axes[1].plot(vc_cmd_times, vc_act_vy, label="act_vy")
            axes[1].set_ylabel("vy")
            axes[2].plot(vc_cmd_times, vc_cmd_wz, label="cmd_wz")
            axes[2].plot(vc_cmd_times, vc_act_wz, label="act_wz")
            axes[2].set_ylabel("wz")
            axes[2].set_xlabel("time (s)")
            for axis in axes:
                axis.grid(True, alpha=0.3)
                axis.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, "vc_velocity_tracking.png"), dpi=200)
            plt.close(fig)

            if vc_xy:
                vc_x, vc_y = zip(*vc_xy)
                fig, axis = plt.subplots(1, 1, figsize=(6, 6))
                axis.plot(vc_x, vc_y, label="virtual_chassis")
                axis.scatter([vc_x[0]], [vc_y[0]], label="vc_start", marker="o", s=30)
                axis.scatter([vc_x[-1]], [vc_y[-1]], label="vc_end", marker="x", s=40)
                if base_xy:
                    base_x, base_y = zip(*base_xy)
                    axis.plot(base_x, base_y, label="base_link")
                    axis.scatter([base_x[0]], [base_y[0]], label="base_start", marker="o", s=30)
                    axis.scatter([base_x[-1]], [base_y[-1]], label="base_end", marker="x", s=40)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel("x")
                axis.set_ylabel("y")
                axis.grid(True, alpha=0.3)
                axis.legend(loc="best")
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, "trajectories_xy.png"), dpi=200)
                plt.close(fig)

            if base_vel:
                base_vx, base_vy, base_vz = zip(*base_vel)
                fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
                axes[0].plot(vc_cmd_times, base_vx, label="base_vx")
                axes[0].set_ylabel("vx")
                axes[1].plot(vc_cmd_times, base_vy, label="base_vy")
                axes[1].set_ylabel("vy")
                axes[2].plot(vc_cmd_times, base_vz, label="base_vz")
                axes[2].set_ylabel("vz")
                axes[2].set_xlabel("time (s)")
                for axis in axes:
                    axis.grid(True, alpha=0.3)
                    axis.legend(loc="upper right")
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, "base_link_velocity.png"), dpi=200)
                plt.close(fig)
        except Exception as exc:
            print(f"[WARN] Plotting failed: {exc}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
