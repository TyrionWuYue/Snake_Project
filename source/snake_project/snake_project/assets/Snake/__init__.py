import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from snake_project import SNAKE_ROOT_DIR


@configclass
class SnakeArticulationCfg(ArticulationCfg):
    """Configuration for Snake articulations."""

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


SNAKE_CFG = SnakeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{SNAKE_ROOT_DIR}/data/Snake/urdf/14DOF-DW.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=1.0,
            max_angular_velocity=1.57,
            max_depenetration_velocity=10.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={
            "yaw.*": 0.0,
            "pitch.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "body_joints": ImplicitActuatorCfg(
            joint_names_expr=["yaw.*", "pitch.*"],
            effort_limit_sim=30.0,
            stiffness=30.0,
            damping=0.6,
            armature=0.028,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["lw_joint.*", "rw_joint.*"],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.1,
            armature=0.0,
            friction=0.0,
            velocity_limit=5.0
        ),
    },
)


__all__ = ["SNAKE_CFG", "SnakeArticulationCfg"]
