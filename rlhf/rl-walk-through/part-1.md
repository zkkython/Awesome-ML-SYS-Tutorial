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

## RL 的分类

### Value-Based（基于价值的）

- **定义**：Value-Based Agent 主要通过学习“价值函数”来做出决策。价值函数衡量某个状态（或状态-行动对）的好坏，表示如果从那个状态开始，按照某种策略行动，可以获得的预期累积奖励。例如，$V(s)$ 是状态 $s$ 的价值，$Q(s, a)$ 是执行动作 $a$ 在状态 $s$ 下的价值。
- **特点**：代理不直接学习“怎么做”（策略），而是学习“做什么最好”（价值的评估）。然后根据价值函数推导出隐式的策略（比如选择价值最高的动作）。
- **学习目标**：学习准确的价值函数（比如 $Q(s, a)$），以便知道每个状态或动作对的长期回报。最终通过价值函数间接生成隐式策略。
- **例子**：Q-Learning 和 Deep Q-Network（DQN）都是典型的 Value-Based 方法。它们学习 $Q$ 值表或函数，然后根据 $Q$ 值选择动作。

### Policy-Based（基于策略的）

- **定义**：Policy-Based Agent 直接学习一个策略（policy），即从状态到动作的映射（可以是概率分布）。策略可以是显式的，比如“在状态 $s$ 下，80% 概率选择动作 $a_1$，20% 概率选择 $a_2$”。
- **特点**：Agent 不依赖价值函数，而是直接优化策略，使其最大化累积奖励。策略可以是参数化的（如神经网络），通过梯度下降等方式调整。
- **学习目标**：直接学习最优策略（比如 $\pi(a|s)$，表示在状态 $s$ 下选择动作 $a$ 的概率），使得长期奖励最大化。不需要价值函数（虽然有时候会结合使用）。
- **例子**：REINFORCE 算法是典型的 Policy-Based 方法，直接优化策略参数。

### Actor-Critic（演员-评论家）
- **特点**：Actor-Critic 结合了 Value-Based 和 Policy-Based 的优点。它有两个部分：
  - **Actor**：负责学习策略（类似 Policy-Based），直接输出动作或动作概率。
  - **Critic**：负责学习价值函数（类似 Value-Based），评估当前策略的好坏（比如 $Q(s, a)$ 或 $V(s)$），指导 Actor 改进。
- **学习目标**：同时学习策略（Actor）和价值函数（Critic），目标是优化策略以最大化长期奖励，同时用价值函数提供更准确的反馈。
- **例子**：A3C（Asynchronous Advantage Actor-Critic）或 PPO（Proximal Policy Optimization）都是常见的 Actor-Critic 方法。


### Model-Based（基于模型的）

- **定义**：Model-Based Agent 通过学习环境的模型来做出决策。模型包括状态转移规则（比如“如果执行动作 $a$ 在状态 $s$，会转移到状态 $s'$”）和奖励函数（“执行这个动作能得到多少奖励”）。通俗来说，这就像是旅行之前先做了旅行攻略，把游玩路线基本都决定好了。
- **特点**：代理不仅学习策略或价值，还学习环境的内部结构（模型）。有了模型后，可以通过规划（planning）或模拟来选择最佳动作，而不完全依赖试错。
- **学习目标**：学习准确的环境模型（状态转移和奖励），然后利用模型优化策略或价值函数，以最大化长期奖励。
- **例子**：使用动态规划（Dynamic Programming）或 AlphaGo 的树搜索（Monte Carlo Tree Search, MCTS）时，模型是关键，用来预测未来的状态并指导行动选择。

### Model-Free（无模型的）

- **定义**：Model-Free Agent 不学习环境模型，而是直接从经验（试错）中学习策略或价值函数。它们只关心当前观察到的状态、动作和奖励，不需要预测未来的状态转移或奖励。类似于在一个城市随机旅行，没有提前的攻略。
- **特点**：简单直接，计算成本低，但可能需要更多的样本（经验）来学习，因为缺乏环境结构的指导。
- **学习目标**：直接学习策略（Policy-Based）或价值函数（Value-Based），以最大化累积奖励，而不依赖环境模型。
- **例子**：Q-Learning（Value-Based）、REINFORCE（Policy-Based）、PPO 或 DQN 都是 Model-Free 方法。

| 类型           | 学什么                     | 模型       | 价值函数       | 策略       | 优点                          | 缺点                          |
|----------------|-----------------------------|-------------------|-----------------------|-------------------|-----------------------------|-----------------------------|
| Value-Based    | 价值函数（间接推导策略）     | 不需要             | 是                   |隐式         | 简单，样本效率高             | 策略可能不灵活               |
| Policy-Based   | 策略                         | 不需要             | 否                   | 是                 | 策略灵活，可处理连续动作     | 样本效率低，训练不稳定       |
| Actor-Critic   | 策略 + 价值函数             | 不需要             | 是                   | 是                 | 结合两者的优点，稳定高效     | 复杂，调参难度高             |
| Model-Based    | 环境模型 + 策略/价值         | 需要               | 可选                 | 可选               | 样本效率高，能规划           | 模型学习困难，计算成本高     |
| Model-Free     | 策略或价值函数               | 不需要             | 可选                 | 可选               | 简单，易实现                 | 样本效率低，依赖大量试错     |

