# 蛇形机器人速度跟踪训练作业说明

## 概述

蛇形机器人具备高冗余的自由度，依赖与地面的摩擦完成各种步态。本项目基于[IsaacLab](https://github.com/isaac-sim/IsaacLab.git)与[rsl_rl](https://github.com/leggedrobotics/rsl_rl)构建，旨在使用深度强化学习完成蛇形机器人的速度跟踪任务。本项目使用的蛇形机器人模型代号名称为14DOF-DW，其中14DOF表示机器人具有14个自由度，DW即double_wheel，表示机器人的底盘具有两个被动轮。蛇形机器人的运动方式如下图所示。

<img src="https://github.com/Zomnk/Snake_Project/blob/main/fig/fig1.jpg" style="zoom:50%;" />

![fig2](https://github.com/Zomnk/Snake_Project/blob/main/fig/fig2.gif)

观察上方机器人的运动，我们发现机器人在运动时每个link都在摆动。在这里我们引入**Virtual Chassis**: <https://ieeexplore.ieee.org/document/6094645>

通过数学方法，从蛇形机器人连续扭动的身体中抽象出一个宏观的、相对平稳的“虚拟参考系”，从而将机器人内部的形变运动与外部的宏观位移彻底解耦。记录上述运动步态中base_link（头部）和Virtual Chassis的轨迹信息，可见后者的轨迹更平滑，符合预期。

![fig3](https://github.com/Zomnk/Snake_Project/blob/main/fig/fig3.png)

接着绘制Virtual Chassis与Base_link的速度情况，对比可见其速度的波动情况较小，符合我们的预期（vx速度方向相反是因为Virtual Chassis的X轴始终对齐蛇尾link）。

![fig4](https://github.com/Zomnk/Snake_Project/blob/main/fig/fig4.png)

本作业为蛇形机器人速度跟踪训练，command作用在Virtual Chassis上，通过在mujoco进行sim2sim迁移，进行固定速度指令组的追踪，评估训练算法在Virtual Chassis速度跟踪任务上面的**累积误差**。



## 安装

### 1. 本地安装

- 使用anaconda创建Python环境

  ```bash
  # create a virtual environment named env_isaaclab with python3.11 and pip
  conda create -n env_isaaclab python=3.11
  # activate the virtual environment
  conda activate env_isaaclab
  ```

- 安装pytorch 2.7.0

  ```bash
  pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  ```

- 安装IsaacSim

    ```bash
    pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

- 克隆IsaacLab的项目，由于作者使用的是老版本的IsaacLab，直接克隆最新的可能会报错，因此作者将本版本上传到Github上，请克隆作者的版本。

    ```bash
    git clone https://github.com/Zomnk/Isaac_Old_Version.git
    ```

- 接下来安装IsaacLab的环境

    ```bash
    # Linux
    ./isaaclab.sh --install # or "./isaaclab.sh -i"
    # Windows
    isaaclab.bat --install :: or "isaaclab.bat -i"
    ```

- 可以通过以下方式验证安装是否正确:

    - 列出可以使用的环境:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```
        
    - 运行任务:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/rsl_rl/train.py --task <TASK_NAME> 
        ```

- 回到上级目录，克隆本训练项目

  ```bash
  cd ..
  git clone https://github.com/Zomnk/Snake_Project.git
  ```

- 安装环境

  ```bash
  python -m pip install -e source/snake_project
  ```

- 拉取Git LFS Assets（USD模型需要通过这个工具获取）

  ```bash
  git lfs install
  git lfs pull
  ```

- 列出可以使用的环境

  ```bash
  python scripts/list_envs.py
  ```

  > 本项目中使用的环境为 Snake-VelocityTracking-Flat-v0 和 Snake-VelocityTracking-Flat-Play-v0

- 运行训练代码

  ```bash
  python scripts/rsl_rl/train.py --task Snake-VelocityTracking-Flat-v0 --num_envs 4096
  ```

- 执行训练好的策略

  ```bash
  python scripts/rsl_rl/play.py --task Snake-VelocityTracking-Flat-Play-v0 --checkpoint <your policy> --video --cmd_vx <your speed> --cmd_vy <your speed>
  ```

  

### 2. 使用服务器

* 使用启智平台的镜像创建本项目

![fig5](https://github.com/Zomnk/Snake_Project/blob/main/fig/fig5.jpg)

- 进入配置好的anaconda环境

  ```bash
  conda env list
  conda activate lab23
  ```

- 克隆本训练项目

  ```
  git clone https://github.com/Zomnk/Snake_Project.git
  ```

- 安装环境

  ```bash
  python -m pip install -e source/snake_project
  ```

- 拉取Git LFS Assets（USD模型需要通过这个工具获取）

  ```bash
  git lfs install
  git lfs pull
  ```
  
- 列出可以使用的环境

  ```bash
  python scripts/list_envs.py
  ```

  > 本项目中使用的环境为 Snake-VelocityTracking-Flat-v0 和 Snake-VelocityTracking-Flat-Play-v0

- 运行训练代码

  ```bash
  python scripts/rsl_rl/train.py --task Snake-VelocityTracking-Flat-v0 --num_envs 4096 --headless
  ```

- 执行训练好的策略

  ```bash
  python scripts/rsl_rl/play.py --task Snake-VelocityTracking-Flat-Play-v0 --checkpoint <your policy> --video --cmd_vx <your speed> --cmd_vy <your speed> --headless
  ```

  


## 代码可修改的部分

* **velocity_env_cfg.py**

  * **观测量**，PolicyCfg不可以修改，CriticCfg特权观测可以修改，噪声幅度与缩放比例可以修改

    ```python
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
                self.enable_corruption = False
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
    ```

  * **域随机化**：当前没有开启域随机化内容，如果sim2sim状况不理想，可考虑添加域随机化内容增强策略的鲁棒性

    ```python
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
    ```

  * **奖励函数**：当前奖励函数较基础，可自行添加新的奖励函数，修改奖励函数的权重来促进训练

    ```Python
    class SnakeVelocityRewardsCfg:
        """Reward terms for the velocity-tracking task."""
    
        track_lin_vel_xy_exp = RewTerm(
            func=mdp.VirtualChassisTrackLinVelXYExp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": virtual_chassis_body_cfg()},
        )
        track_ang_vel_z_exp = RewTerm(
            func=mdp.VirtualChassisTrackAngVelZExp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": virtual_chassis_body_cfg()},
        )
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4, params={"asset_cfg": yaw_joint_cfg()})
        joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": yaw_joint_cfg()})
        raw_action_rate = RewTerm(func=mdp.RawActionRatePenalty, weight=-0.01, params={"action_term_name": "joint_pos"})
        joint_amplitude = RewTerm(func=mdp.joint_amplitude, weight=0.2, params={"asset_cfg": yaw_joint_cfg()})
        phase_propagation = RewTerm(func=mdp.phase_propagation, weight=0.4, params={"asset_cfg": yaw_joint_cfg()})
        motion_coordination = RewTerm(func=mdp.motion_coordination, weight=-0.3, params={"asset_cfg": yaw_joint_cfg()})
    ```

  * **课程学习**：可以修改当前课程学习的实现形式与相关参数，来促进训练的平稳

    ```python
    class SnakeVelocityCurriculumCfg:
        """Curriculum hooks for the velocity-tracking task."""
    
        command = CurrTerm(
            func=mdp.command_velocity_curriculum,
            params={
                "command_name": "base_velocity",
                "reward_term_name": "track_lin_vel_xy_exp",
                "max_curriculum": 0.4,
                "step_size": 0.1,
                "threshold_ratio": 0.8,
            },
        )
    ```

  - **action输出**：若端到端性能不佳，可考虑其他方法

    ```python
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
    ```

- **rsl_rl_ppo_cfg.py**

  - PPO算法参数与训练配置

    ```python
    class SnakeVelocityFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 24
        max_iterations = 20000
        save_interval = 500
        experiment_name = "snake_velocity_flat_tracking"
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=0.01,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    ```



## 提交作业成果

* 请运行sim2sim_python.py来可视化查看指定command的机器人运动情况

  ```
  python sim2sim/sim2sim_mujoco.py --cmd_vx <your speed> --cmd_vy <your speed> --policy <your jit policy>
  ```

* 请运行sim2sim_eval.py来评估机器人在25组指令下的速度追踪情况

  ```
  python sim2sim/sim2sim_eval.py --policy <your jit policy>
  ```

  > 请注意Mujoco脚本使用的是jit策略，一般需要先用IsaacLab运行play.py，在对应log文件夹下找到export\xxx.pt，这个才是jit的格式

sim2sim_python.py运行后会将Virtual Chassis和base_link的速度与轨迹保存到figures文件夹下

sim2sim_eval.py运行后会生成25组指令对应的Virtual Chassis和base_link的速度、轨迹与累计**MAE**误差结果，保存到eval_output文件夹下

![fig6](https://github.com/Zomnk/Snake_Project/blob/main/fig/fig6.jpg)

请提交以下内容给课程助教：

1. **Snake_Project/tree/main/source/snake_project/snake_project/tasks/manager_based/velocity_tracking 压缩为zip文件** 
2. **评估效果最佳的策略模型pt文件**
3. **Snake_Project/tree/main/source/snake_project/snake_project/sim2sim/eval_output 压缩为zip文件**