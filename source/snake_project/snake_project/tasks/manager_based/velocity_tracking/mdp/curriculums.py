from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _set_command_range(command_term, x_abs: float, y_abs: float) -> tuple[float, float, float, float]:
    x_min = round(-abs(float(x_abs)), 4)
    x_max = round(abs(float(x_abs)), 4)
    y_min = round(-abs(float(y_abs)), 4)
    y_max = round(abs(float(y_abs)), 4)
    command_term.current_lin_vel_x_range = [x_min, x_max]
    command_term.current_lin_vel_y_range = [y_min, y_max]
    command_term.cfg.ranges.lin_vel_x = (x_min, x_max)
    command_term.cfg.ranges.lin_vel_y = (y_min, y_max)
    return x_min, x_max, y_min, y_max


def scheduled_command_velocity_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    command_name: str = "base_velocity",
    min_lin_vel_x: float = 0.2,
    min_lin_vel_y: float = 0.1,
    max_lin_vel_x: float = 0.4,
    max_lin_vel_y: float = 0.2,
    schedule_start_steps: int = 48_000,
    schedule_end_steps: int = 96_000,
) -> dict[str, float]:
    """Deterministically widen x/y command ranges on a fixed training schedule."""
    command_term = env.command_manager.get_term(command_name)

    if env.common_step_counter <= schedule_start_steps:
        progress = 0.0
    elif env.common_step_counter >= schedule_end_steps:
        progress = 1.0
    else:
        progress = (env.common_step_counter - schedule_start_steps) / max(1, schedule_end_steps - schedule_start_steps)

    x_abs = min_lin_vel_x + progress * (max_lin_vel_x - min_lin_vel_x)
    y_abs = min_lin_vel_y + progress * (max_lin_vel_y - min_lin_vel_y)
    x_min, x_max, y_min, y_max = _set_command_range(command_term, x_abs=x_abs, y_abs=y_abs)

    return {
        "lin_vel_x_min": x_min,
        "lin_vel_x_max": x_max,
        "lin_vel_y_min": y_min,
        "lin_vel_y_max": y_max,
        "schedule_progress": progress,
    }


def command_velocity_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy_exp",
    max_curriculum: float = 0.4,
    min_curriculum: float = 0.1,
    step_size: float = 0.05,
    max_lin_vel_x: float | None = None,
    max_lin_vel_y: float | None = None,
    min_lin_vel_x: float | None = None,
    min_lin_vel_y: float | None = None,
    step_size_x: float | None = None,
    step_size_y: float | None = None,
    threshold_ratio: float = 0.8,
    ema_decay: float = 0.05,
    min_env_count: int = 10,
    warmup_steps: int = 0,
    update_interval_steps: int = 1,
) -> dict[str, float]:
    """Expand/shrink the x/y command ranges using EMA-smoothed reward.

    Uses exponential moving average over per-episode tracking rewards.  Only updates
    the range when at least ``min_env_count`` environments have finished, and the
    EMA crosses the expand / shrink thresholds.  The separate x/y bounds keep the
    curriculum aligned with the asymmetric baseline command range.
    """
    command_term = env.command_manager.get_term(command_name)
    x_min, x_max = command_term.current_lin_vel_x_range
    y_min, y_max = command_term.current_lin_vel_y_range
    max_x = max_curriculum if max_lin_vel_x is None else max_lin_vel_x
    max_y = max_curriculum if max_lin_vel_y is None else max_lin_vel_y
    min_x = min_curriculum if min_lin_vel_x is None else min_lin_vel_x
    min_y = min_curriculum if min_lin_vel_y is None else min_lin_vel_y
    step_x = step_size if step_size_x is None else step_size_x
    step_y = step_size if step_size_y is None else step_size_y

    if env_ids is None or len(env_ids) == 0:
        return {
            "lin_vel_x_min": x_min, "lin_vel_x_max": x_max,
            "lin_vel_y_min": y_min, "lin_vel_y_max": y_max,
        }

    episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    mean_tracking_reward = float(torch.mean(episode_sums) / env.max_episode_length_s)
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    threshold = threshold_ratio * reward_cfg.weight

    if not hasattr(command_term, "_tracking_reward_ema"):
        command_term._tracking_reward_ema = mean_tracking_reward
    else:
        command_term._tracking_reward_ema = (
            (1.0 - ema_decay) * command_term._tracking_reward_ema + ema_decay * mean_tracking_reward
        )
    ema = command_term._tracking_reward_ema

    if env.common_step_counter < warmup_steps:
        return {
            "lin_vel_x_min": x_min, "lin_vel_x_max": x_max,
            "lin_vel_y_min": y_min, "lin_vel_y_max": y_max,
            "mean_tracking_reward": ema,
        }

    if update_interval_steps > 1:
        last_update_step = getattr(command_term, "_curriculum_last_update_step", -update_interval_steps)
        if env.common_step_counter - last_update_step < update_interval_steps:
            return {
                "lin_vel_x_min": x_min, "lin_vel_x_max": x_max,
                "lin_vel_y_min": y_min, "lin_vel_y_max": y_max,
                "mean_tracking_reward": ema,
            }

    if len(env_ids) >= min_env_count:
        if ema > threshold:
            x_min, x_max = command_term.expand_lin_vel_x(step_size=step_x, max_curriculum=max_x)
            y_min, y_max = command_term.expand_lin_vel_y(step_size=step_y, max_curriculum=max_y)
            command_term._curriculum_last_update_step = env.common_step_counter
        elif ema < 0.6 * threshold:
            x_min, x_max = command_term.shrink_lin_vel_x(step_size=step_x, min_curriculum=min_x)
            y_min, y_max = command_term.shrink_lin_vel_y(step_size=step_y, min_curriculum=min_y)
            command_term._curriculum_last_update_step = env.common_step_counter

    return {
        "lin_vel_x_min": x_min, "lin_vel_x_max": x_max,
        "lin_vel_y_min": y_min, "lin_vel_y_max": y_max,
        "mean_tracking_reward": ema,
    }
