# Demystifying Reinforcement Learning

这是 UCLA 2025 年初春的 RL 课程笔记，感谢周博磊老师的课件。我把这系列的名字其名为 Demystifying Reinforcement Learning，因为自己一直想要做一些 RLHF framework 的工作，所以对 RL 算法的直观非常重要。出于我个人的学习习惯，我对理论不感兴趣，好在对于 ML SYS researcher 而言，这样的直观已然可以带来巨大的帮助。

为了在直观和扎实的理论之间做权衡，一部分内容读上去似乎会显然到觉得愚蠢。**然而曾有一位我的 co-author 告诉我，许多平凡到看似愚蠢的定义反而是 ML Theory 的核心**。我虽然不会 ML Theory，但是心中只有谦卑，故而不会略去这些平凡的定义。

最后，这系列笔记在我的 Awesome-ML-Sys 仓库中，自己也会从 SYS 角度思考一些问题，督促自己学习。

## Introduction

- **RL 和 supervised learning 的区别**

老实说这是我本科就没有想明白的内容，老师的课件此处写的极好。

1. 输入的数据是序列的，先后顺序有重大影响
2. learner 并不会被告知做出如何的 action，需要自己去发现最终收益最大的 action
3. trial and error exploration，需要在 exploration（探索新策略）和 exploitation（利用当前策略）之间权衡
4. 不存在监督者，只有 reward 信号，而且 reward 信号是延后的

- **强化学习的性质**

1. trial and error exploration
2. delayed reward
3. time matters, sequential decision making, not i.i.d.
4. Actions affect the evironment

- **RL 的可能问题**

1. interpretability
2. diversity of the environment
3. overfitting on training environment
4. reward engineering
5. no safty guarantee
6. low sample efficiency

## RL Basic

### RL 的常见要素

1. Agent 和 Environment
2. State
3. Observation：注意 state 和 observation 会在[下一个段落](#Seqential-Decision-Making)做出区分
4. Reward：reward 是环境给予的标量反馈，表征当前时间步 t 下，agent 的表现如何；所有强化学习的目标都可以被概述为最大化累计期望 reward

### Seqential Decision Making

Agent 的目标是选择一系列的 action 以最大化累计期望 reward，其所选择的 Actions 需要具有长时间的影响。Reward 往往是延迟的，而 Agent 需要在即刻 reward 和长期 reward 之间做出权衡。

- History（历史，$H_t$）：历史是一组由观察（Observations）、行动（Actions）和奖励（Rewards）组成的序列，记作 
$H_t = O_1, A_1, R_1, O_2, A_2, R_2, \ldots, O_t, A_t, R_t$。它记录了智能体与环境从时间 1 到 t 的全部交互过程。虽然在完全可观察的 MDP 中，未来的状态和奖励只依赖当前状态和行动，不需要完整的 $H_t$，但历史可以作为智能体的学习记录或调试工具。

- Environment State（环境状态，$S_{t}^{e}$）：环境中某个时刻的真实、完整状态，描述了环境的所有相关信息（比如迷宫中的所有位置、墙壁、出口等）。在 MDP 中，智能体可以直接观察到 $S_{t}^{e}$，且状态转移只依赖当前 $S_{t}^{e}$ 和行动 $A_t$，不依赖历史。

- Observation（观察，$O_t$）：智能体从环境中接收到的信息。在完全可观察的 MDP 中，观察等于环境状态，即 $O_t = S_{t}^{e}$，因为环境是完全可观察的，智能体可以看到环境的全部内容。

- Agent State（智能体状态，$S_{t}^{a}$）：智能体对环境的表示。在完全可观察的 MDP 中，智能体状态直接等于环境状态，即 $S_{t}^{a} = O_t = S_{t}^{e}$，因为智能体可以直接看到环境的真实状态。

### RL Agent 的主要组成

1. Model（模型）

模型预测环境下一步将会发生什么，可以预测下一步的环境的 state（状态），或者环境的 reward（奖励）。

2. Policy（策略）

模型预测好下一步将会发生什么后，策略负责采取具体的行动，分为随机策略（Stochastic Policy）和确定性策略（Deterministic Policy）。

3. Value Function（价值函数）

价值函数用于评估 state（状态）或 state-action pair（状态-行动对）在特定策略下的长期回报，间接判断怎样的策略更优。具体有两种形式：

- State Value Function（状态价值函数）用于评估某个状态 $s$ 有多好，表示在状态 $s$ 下遵循策略 $\pi$ 的期望折现奖励。公式为：

$$
v_{\pi}(s) = E_{\pi}[G_t | S_t = s] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s]
$$

其中，$G_t$ 是从时间 $t$ 开始的未来奖励总和，$R_{t+k+1}$ 是每个时间步的奖励，$\gamma$ 是折扣因子，用于衡量长期奖励的重要性。

- Q-function（Q 函数，状态-行动价值函数）用于评估在状态 $s$ 下采取行动 $a$ 有多好，表示在状态 $s$ 下采取行动 $a$ 遵循策略 $\pi$ 的期望折现奖励，用于在多个行动之间做出选择。公式为：

$$
q_{\pi}(s, a) = E_{\pi}[G_t | S_t = s, A_t = a] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s, A_t = a]
$$

价值函数（无论是 $V(s)$ 还是 $Q(s,a)$）的核心作用是评估状态或状态-行动对在特定策略下的长期回报，而不是直接评估策略本身。它为优化策略提供了基础，通过比较不同策略下的价值函数，可以选择表现更好的策略。
