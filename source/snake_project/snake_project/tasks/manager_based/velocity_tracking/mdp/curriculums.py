from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

    if len(env_ids) >= min_env_count:
        if ema > threshold:
            x_min, x_max = command_term.expand_lin_vel_x(step_size=step_x, max_curriculum=max_x)
            y_min, y_max = command_term.expand_lin_vel_y(step_size=step_y, max_curriculum=max_y)
        elif ema < 0.6 * threshold:
            x_min, x_max = command_term.shrink_lin_vel_x(step_size=step_x, min_curriculum=min_x)
            y_min, y_max = command_term.shrink_lin_vel_y(step_size=step_y, min_curriculum=min_y)

    return {
        "lin_vel_x_min": x_min, "lin_vel_x_max": x_max,
        "lin_vel_y_min": y_min, "lin_vel_y_max": y_max,
        "mean_tracking_reward": ema,
    }
