# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import snake_project.tasks.manager_based.velocity_tracking.mdp as mdp
from snake_project.assets.Snake import SNAKE_CFG

YAW_JOINT_NAMES = [f"yaw{index}" for index in range(1, 8)]
VIRTUAL_CHASSIS_BODY_NAMES = ("base_link",) + tuple(f"link{index}" for index in range(1, 15))
ROBOT_CFG = SNAKE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def yaw_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=YAW_JOINT_NAMES, preserve_order=True)


def virtual_chassis_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=list(VIRTUAL_CHASSIS_BODY_NAMES), preserve_order=True)


@configclass
class SnakeVelocitySceneCfg(InteractiveSceneCfg):
    """Scene for the snake velocity-tracking task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=True,
    )

    robot: ArticulationCfg = ROBOT_CFG

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=1,
        debug_vis=False,
    )


@configclass
class SnakeVelocityCommandsCfg:
    """Command specifications for the velocity-tracking MDP."""

    base_velocity = mdp.VirtualChassisVelocityCommandCfg(
        class_type=mdp.VirtualChassisVelocityCommand,
        asset_name="robot",
        body_names=VIRTUAL_CHASSIS_BODY_NAMES,
        resampling_time_range=(10.0, 10.0),
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        planar_zero_threshold=0.0,
        debug_vis_env_idx=0,
        velocity_marker_max_speed=0.75,
        velocity_marker_z_offset=0.10,
        ranges=mdp.VirtualChassisVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.0, 0.0),
            heading=(-0.0, 0.0),
        ),
        debug_vis=True,
    )


@configclass
class SnakeVelocityActionsCfg:
    """Action specifications for the velocity-tracking MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=YAW_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        clip={"yaw.*": (-1.57, 1.57)},
    )


@configclass
class SnakeVelocityObservationsCfg:
    """Observation specifications for the velocity-tracking MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.0125, n_max=0.0125))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.001, n_max=0.001))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": yaw_joint_cfg()}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": yaw_joint_cfg()}, noise=Unoise(n_min=-0.01, n_max=0.01))
        last_actions = ObsTerm(func=mdp.last_raw_actions, params={"action_name": "joint_pos"})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.0125, n_max=0.0125))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.001, n_max=0.001))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": yaw_joint_cfg()}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": yaw_joint_cfg()}, noise=Unoise(n_min=-0.01, n_max=0.01))
        last_actions = ObsTerm(func=mdp.last_raw_actions, params={"action_name": "joint_pos"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class SnakeVelocityEventCfg:
    """Configuration for reset and randomization events."""

    reset_robot = EventTerm(
        func=mdp.reset_snake_state,
        mode="reset",
        params={
            "asset_cfg": yaw_joint_cfg(),
            "joint_position_range": (0.00, 0.00),
            "pose_range": {"x": (-0.2, 0.2), "y": (0.2, 0.2), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )
    randomize_robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    """
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_link_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.005, 0.005),
            },
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    """

@configclass
class SnakeVelocityRewardsCfg:
    """Reward terms for the velocity-tracking task."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.VirtualChassisTrackLinVelXYExp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.4, "linear_coef": 0.5, "asset_cfg": virtual_chassis_body_cfg()},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.VirtualChassisTrackAngVelZExp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": virtual_chassis_body_cfg()},
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4, params={"asset_cfg": yaw_joint_cfg()})
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": yaw_joint_cfg()})
    raw_action_rate = RewTerm(func=mdp.RawActionRatePenalty, weight=-0.01, params={"action_term_name": "joint_pos"})
    joint_amplitude = RewTerm(func=mdp.joint_amplitude, weight=0.2, params={"asset_cfg": yaw_joint_cfg()})
    phase_propagation = RewTerm(func=mdp.phase_propagation, weight=0.4, params={"asset_cfg": yaw_joint_cfg()})
    motion_coordination = RewTerm(func=mdp.motion_coordination, weight=-0.5, params={"asset_cfg": yaw_joint_cfg()})
    # is_terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)
    # contact_penalty = RewTerm(func=mdp.contact_penalty, weight=-5.0, params={
    #     "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=list(VIRTUAL_CHASSIS_BODY_NAMES)),
    #     "threshold": 0.0,
    # })

@configclass
class SnakeVelocityTerminationsCfg:
    """Termination conditions for the velocity-tracking task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_state = DoneTerm(
        func=mdp.invalid_state,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_root_lin_vel": 2.0,
            "max_root_ang_vel": 5.0,
            "min_root_height": -0.2,
            "max_root_height": 0.5,
        },
    )


@configclass
class SnakeVelocityCurriculumCfg:
    """Curriculum hooks for the velocity-tracking task."""

    # command = CurrTerm(
    #     func=mdp.command_velocity_curriculum,
    #     params={
    #         "command_name": "base_velocity",
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "max_curriculum": 0.4,
    #         "min_curriculum": 0.1,
    #         "step_size": 0.05,
    #         "threshold_ratio": 0.8,
    #     },
    # )


@configclass
class SnakeVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for the snake velocity-tracking task."""

    scene: SnakeVelocitySceneCfg = SnakeVelocitySceneCfg(num_envs=4096, env_spacing=3.0)
    commands: SnakeVelocityCommandsCfg = SnakeVelocityCommandsCfg()
    observations: SnakeVelocityObservationsCfg = SnakeVelocityObservationsCfg()
    actions: SnakeVelocityActionsCfg = SnakeVelocityActionsCfg()
    events: SnakeVelocityEventCfg = SnakeVelocityEventCfg()
    rewards: SnakeVelocityRewardsCfg = SnakeVelocityRewardsCfg()
    terminations: SnakeVelocityTerminationsCfg = SnakeVelocityTerminationsCfg()
    curriculum: SnakeVelocityCurriculumCfg = SnakeVelocityCurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.25)
