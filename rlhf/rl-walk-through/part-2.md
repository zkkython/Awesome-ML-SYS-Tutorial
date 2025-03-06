# Demystifying Reinforcement Learning (Part 2)

书接上文，想不到两次写作这一笔记，中间隔了整整一周。究竟是什么事情让我的学习进度一直被打断呢？

按照朋友所言，作为一个项目的 leader，我太在乎话语权了。话语权是最大的权利，同时也让我完全没法将手头的事情分享给他人。基于此，我决定每天至少有 1/3 的时间，关掉微信和 slack，全心自己学习，来琢磨一些事情。

## 马尔科夫决策过程

### Greedy Action and $\epsilon$-Greedy Action

Q(a) 是行动（action a）的值函数，表示如果选择某个行动 a，智能体期望获得的平均奖励。具体来说，$Q_t(a)$ 是到第 $t$ 步为止，每次选择行动 $a$ 获得的奖励总和，除以选择 $a$ 的次数。这是一个基于经验的平均值，用来估计行动 $a$ 的“好坏”。

$$
  Q_t(a) = \frac{\text{sum of rewards when action a was taken prior to t}}{\text{number of times a was taken prior to t}}
$$

举例来说，行动 A 有时给你 10 分，有时给你 2 分。你尝试了行动 A 5 次，总共得到 40 分。那么 \( Q(A) = 40 / 5 = 8 \) 分。这是行动 A 的当前估计价值。

有了 $Q_t(a)$ 的定义，我们可以定义贪婪策略（Greedy Strategy），也即每次选择 $Q_t(a)$ 最大的行动 $a$。这种策略其实就是 local minima，掉入了 Exploration-Exploitation Dilemma。为此，$\epsilon$-贪婪策略引入了探索。它在大多数时候使用贪婪策略选择最佳行动，但以很小的概率 $\epsilon$ 随机选择一个行动（从所有可能的行动中均匀选择）。

这里可以给出一段经典的伪代码。考虑一个有 k 个拉杆（arms 或 actions）的老虎机，智能体需要决定每次拉哪个拉杆以最大化奖励。ε-贪婪策略在这里用来平衡探索和利用。

```plaintext
1: for a = 1 to k do
2:    Q(a) = 0, N(a) = 0
3: end for
4: loop
5:    A = {
         arg max_a Q(a)    with probability 1 - ε
         uniform(A)        with probability ε
       }
6:    r = bandit(A)
7:    N(A) = N(A) + 1
8:    Q(A) = Q(A) + 1/N(A) [r - Q(A)]
9: end loop
```

1 到 3 进行初始化，从零开始学习每个行动的价值。4 到 9 是主循环：

- 行动选择：按照前文所述的方式，进行 $\epsilon$-greedy 选择。
- 获取奖励：选择了行动 A 后，拉下拉杆，获得奖励 r。
- 更新选择次数：每次选择行动 A，将其被选择次数 $N(A)$ 增加 1。来跟踪每个行动被尝试的频率。
- 更新行动价值：使用增量式更新公式来调整 $Q(A)$，基于新获得的奖励 $r$ 和当前估计 $Q(A)$。式中，$1/N(A)$ 是“学习率”或“步长”（step size），随着 $N(A)$ 增加（即行动被选择更多次），步长变小，更新变得更平滑。$r - Q(A)$ 是“误差”或“偏差”：新奖励 $r$ 减去当前估计 $Q(A)$。如果 $r > Q(A)$，说明行动比我们估计的更好；如果 $r < Q(A)$，说明估计过高。
- 循环继续，重复选择行动、获取奖励、更新估计。

更具体的来说：

$$
  Q_t(a_t) = Q_{t-1} + \frac{1}{N_t(a_t)} (r_t - Q_{t-1}(a_t))\\
  \text{NewEstimate} = \text{OldEstimate} + \text{StepSize} \times (\text{Target} - \text{OldEstimate})
$$

这是第 8 行公式的更形式化的表达，明确了时间步（time step）的依赖。$Q_t(a_t)$ 是在时间步 $t$ 对行动 $a_t$ 的估计值。$Q_{t-1}(a_t)$ 是时间步 $t-1$ 对同一行动的估计值（旧估计）。$r_t$ 是时间步 $t$ 选择 $a_t$ 后获得的奖励。$N_t(a_t)$ 是到时间步 $t$ 为止，行动 $a_t$ 被选择的总次数。公式和第 8 行一样，表明新估计是通过旧估计加上一个基于误差 $ [r_t - Q_{t-1}(a_t)] $ 和学习率 $ \frac{1}{N_t(a_t)} $ 的调整得到的。

当然，这个多臂老虎机模型是最为简化的 RL 任务，因为奖励是不延迟的，state 不会变化，而且 action 之间不会有后续影响。

### 马尔科夫奖励过程（Markov Reward Process, MRP）与马尔科夫决策过程（Markov Decision Process, MDP）

马尔科夫过程意味着”此时此刻，过去不会影响未来“，这一抽象能够建模相当多的实际问题。虽然 MDP 是更广为人知的概念，但是可以从 MRP 开始，作为 MDP 的基础。

在 MRP 中，