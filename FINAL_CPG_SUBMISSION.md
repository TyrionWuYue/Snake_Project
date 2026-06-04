# 最终版本：CPG + Scheduled Curriculum

本文说明从原始 baseline（参考 [Zomnk/Snake_Project README](https://github.com/Zomnk/Snake_Project/blob/main/README.md)）到最终提交版本的主要改动。作业要求中允许修改奖励、课程学习，并在端到端 action 输出效果不佳时采用其他 action 形式；最终方案遵守这一边界，没有向 policy 引入额外特权观测。

## 1. 从逐关节 MLP 到 CPG 参数化动作空间

baseline 的 actor 本身是 MLP，其输出直接对应 7 个 yaw 关节的 position target。这个形式虽然端到端、表达能力强，但对蛇形机器人并不一定是最合适的归纳偏置。蛇形机器人的有效推进不是任意 7 个 yaw 角的组合，而是沿身体传播的周期性弯曲波。若让 MLP 独立决定每个 yaw target，策略需要同时学习周期性、相邻关节相位关系、整体波形幅值、前后向切换和侧向修正；这会使探索空间中存在大量“物理上可执行但运动学上无效”的动作组合，容易导致关节之间互相抵消，产生高频抖动或局部摆动但缺少净位移。

因此最终版本没有取消 MLP，而是改变了 MLP 的输出语义：MLP 不再输出低层关节动作，而是输出 9 维 CPG gait parameters：

```text
frequency, amplitude, phase_lag, offset, residual(5)
```

随后由 action layer 将这些参数展开为 7 个 yaw target。这个设计受 CPG（Central Pattern Generator）以及 CPG-Actor 类方法启发：将神经策略与具有周期结构的运动生成器结合，使策略主要学习如何调制步态，而不是从零发明每个时刻的关节波形。换句话说，策略搜索空间从“高冗余关节空间”转为“低维步态参数空间”。

## 2. 为什么 CPG 更适合这个任务

该任务的目标是 Virtual Chassis 速度跟踪。对蛇形机器人而言，速度跟踪质量主要由整条身体产生的 traveling wave 决定，而不是某个单独关节的瞬时误差。CPG 参数化提供了几个有用的学术性归纳偏置：

- **动作空间降维**：从逐关节 action 转为少量 gait parameters，减少无效动作组合，提高 PPO 在有限样本下找到可推进步态的概率。
- **相位耦合**：`phase_lag` 显式编码相邻关节之间的相位差，使身体波形天然具有传播方向和空间连续性。
- **周期稳定性**：内部 phase state 随时间累积，保证动作在时间上连续，而不是完全依赖 MLP 每一步重新生成独立动作。
- **形态先验**：蛇形推进的主要模式被建模为正弦波，符合机器人身体结构和接触推进机理。
- **有限自由度修正**：5 维 residual basis 允许策略修正基础正弦波形，适应摩擦、被动轮和侧向指令，但不会退化回完全无结构的逐关节控制。

因此，最终方案不是简单地“加一个手工步态”，而是采用 **structured policy output**：CPG 提供可解释、稳定的底层运动流形，PPO 在这个流形上学习适合不同速度指令的调制参数。

## 3. 奖励与课程学习的改动

| 模块 | Baseline | 最终版本 | 目的 |
|---|---|---|---|
| Action 输出 | MLP 直接输出 7 个 yaw target | MLP 输出 9 维 CPG 参数，再展开为 7 个 yaw target | 将低层逐关节控制转为高层步态调制，提升协调性和稳定性。 |
| 步态生成 | 无显式周期结构 | 主正弦波 + 相位传播 + residual basis | 保留 CPG 周期性，同时允许 RL 进行有限形状修正。 |
| Reward | 基础速度跟踪与平滑项 | 强化 Virtual Chassis 线速度跟踪，加入 command direction progress / progress floor，并降低 raw action rate 对 CPG 参数的压制 | 训练目标更贴近 sim2sim 的 Virtual Chassis MAE 指标，鼓励沿 command 方向产生净位移。 |
| Curriculum | 基于 reward 阈值扩展 command range | 固定日程扩展：0-2000 iter 固定 `vx ±0.2, vy ±0.1`；2000-4000 iter 线性扩到 `vx ±0.4, vy ±0.2`；4000 后保持 full range | reward-gated curriculum 在 CPG 策略上过于保守；固定日程确保策略见到 2 倍速度范围，改善边界速度泛化。 |

## 4. 为什么没有采用更复杂的 CPG 变体

实验中也尝试过显式 curvature、free-carrier、zero-vx drift reward 等更复杂版本。这些方法能修复少数 outlier，但会破坏更大范围内原本稳定的主步态。例如，显式曲率先验会改善某些侧向命令，却可能干扰前向 `vx=0.1` 附近的稳定 traveling wave。最终选择保守的 Param-CPG 主干，是因为它在稳定性、可解释性和整体 sim2sim 误差之间取得了更好的平衡。

## 5. 最终模型选择与结果

最终选择 `paramcpg_r050_body14_v1_schedwide_v2/model_4000.pt`。`model_5000.pt` 的平均误差接近，但出现单个明显 outlier；`model_4000.pt` 没有坏点，更适合作为最终提交版本。

| 版本 | planar MAE mean | median | max | `planar MAE <= 0.20` |
|---|---:|---:|---:|---:|
| baseline | 0.4073 | 0.4032 | 0.5500 | 0/25 |
| 第一版 Param-CPG | 0.1819 | 0.1335 | 0.5040 | 17/25 |
| 最终 `schedwide_v2/model_4000` | **0.1197** | **0.1233** | **0.1617** | **25/25** |

相对 baseline，最终版本的 planar MAE 平均下降约 **70.6%**，且 25 组 sim2sim 指令全部低于 0.20。该结果说明：相比直接逐关节 MLP action，带有 CPG 结构先验的动作参数化更适合蛇形机器人的速度跟踪任务。
