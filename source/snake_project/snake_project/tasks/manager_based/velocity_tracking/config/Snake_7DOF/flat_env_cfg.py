# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from snake_project.assets.Snake import SNAKE_CFG
from snake_project.tasks.manager_based.velocity_tracking.velocity_env_cfg import SnakeVelocityEnvCfg


@configclass
class SnakeVelocityFlatEnvCfg(SnakeVelocityEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = SNAKE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


class SnakeVelocityFlatEnvCfg_PLAY(SnakeVelocityFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = True
        self.observations.critic.enable_corruption = True
