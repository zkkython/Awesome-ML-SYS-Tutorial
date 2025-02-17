# veRL 原文浅析

众所周知，我一直在 SGLang team 负责端茶倒水 + RLHF。之前主要的工作在 OpenRLHF 上，而 veRL 也是非常优秀的 RLHF 框架。其独到的 hybrid engine 想法和我在尝试优化 RLHF 框架时的一些想法不谋而合。其实 SGLang 团队一直有成员在负责这块的工作推动，但是由于 DeepSeek 支持以及团队成员的精力有限，没能将这系列的改动 up stream 到 main branch 上。终于，到了今天，随着 grok 的发布，我们将在这段时间内快速将 SGLang 对于 hybrid engine 的支持（主要是 SPMD）up stream 到 main branch 上。这里浅浅预告一番，也欢迎社区期待我们更多的合作。

这篇文章将会是我在 veRL 的系列工作的开端笔记，也督促自己更加全面地来理解 training and inference co-design 的思路。

之前也有基于 nemo-aligner 和 openRLHF 做一些解析，欢迎大家参考，也感谢这些框架作者的精彩贡献。大家都是写开源框架的人，其中的酸甜苦辣，自不必多说。

- [浅析主流 Alignment 算法与 NeMo-Aligner 框架](https://zhuanlan.zhihu.com/p/5220718268)
- [浅析以 OpenRLHF 为代表的 post-training 系统的计算流程](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/OpenRLHF/readme.md)

## Single-Controller vs Multi-Controller

在介绍 introduction 之前，我们先来引入 veRL 最重要的概念——single controller 和 multi controller。可以简单直观地来思考下，在一个复杂的工作流程中，single controller 只有一个程序负责管理，而其他的子模块只负责执行。所有的控制逻辑都可以写在这唯独的 single controller 上，因此控制逻辑实现简单，便于调试。然而，所有子模块都只由一个程序来管理，single controller 所承担的管理压力其实非常大，一来可能因为通讯强度过大而效率堪忧，二来，倘若 single controller 崩溃，整个系统将彻底失效。反过来，multi controller 则有多个控制程序来管理不同的子模块，每个子模块仍旧只负责执行自己的功能。如此以来，单个控制程序的管理压力降低，系统更加鲁棒可扩展。然而，所有的控制逻辑都分散在多个程序中，因此控制逻辑实现复杂，难以调试。

基于此，我们可以回顾下 [RLHF 的工作流程](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/OpenRLHF/readme.md#%E6%9B%B4%E6%96%B0%E6%B5%81%E7%A8%8B)：

- 准备一个 batch 的 prompts；
- 将这个 batch 的 prompts 输入给 Actor，解码得到 responses；
- 将 prompt + responses 输入给 Critic/Reward/Reference，分别计算得得到所有 token 的 values、最后一个 token 的 reward 和所有 token 的 log probs，按照强化学习的术语，称这些数据为经验（experiences）；
- 根据 experiences 多轮计算 actor loss 和 critic loss 并更新 Actor 和 Critic 模型。

如此以来，很自然我们能够想到一个 naive 想法——用 single controller 来管理整个 RLHF 的工作流程，然后每个子模块（Actor、Critic、Reward、Reference）再各自由一个 single controller 来管理。然而，这种最 naive 的实现里，两层控制结构都是 single controller，因此系统内的通讯 overhead 非常大。一个可能不太直观的点是，最高层负责总调度的 single controller 承担的通讯压力反而比起每个子模块的单个 controller 承担的通讯压力要小。可以这么理解，inference engine 只需要把 prompt + response 反馈给上层 controller 就好了，这个通讯量重启不过几 KB，而 engine 内部的通讯就非常大了，不然怎么会把 NV Link 接近 T 为单位的通讯速度都打满呢？

因此，实际上第二层的 controller 承担的通讯压力更大，如果这一层还都是 single contrller 的，可以想见效率堪忧。因此，实际上主流的训练引擎都是 multi controller 的，譬如 FSDP Megatron 和 DeepSpeed。既然训练引擎是 multi controller 的（具体来说是 SPMD 的），那么 RLHF 依赖的推理引擎呢，是否也该是 multi controller 的？从理论上来讲，是的，直觉告诉我们，SPMD 的 training engine 和 SPMD 的 inference engine 相互通讯会比和 single controller 的 inference engine 通讯要高效得多。

这个直觉是否正确呢？简单想想，其实很好理解。从架构角度来看，SPMD 的 training engine 与 SPMD 的 inference engine 都采用了分布式、多控制器的设计，各节点之间可以直接、高效地进行点对点通信，这样可以充分利用高速互联（例如 NVLink 或 InfiniBand），实现数据并行传输和计算。而如果 inference engine 采用的是 single controller 模式，所有来自多个训练节点的数据都必须集中到单一的控制节点上进行汇总和调度，这就不可避免地引入了通信瓶颈和单点故障风险。

因此，SPMD 的 training engine 和 SPMD 的 inference engine 简直是绝配，然而由于历史原因，目前主流的推理引擎还是以 single controller 为主的。所以这是 SGLang team 的一大工作目标，将 SGLang 由 single controller 改为 SPMD 的 inference engine。实际上已经有了非常成熟的 PR，这里可以[参考这个 branch](https://github.com/fzyzcjy/sglang/tree/feat/overall_verl)。

总之，这一部分花费了巨大的篇幅来简述 single controller 和 multi controller 的优劣，以及为什么 SGLang 需要从 single controller 改为 multi controller。理解了这些概念后，我们可以正式进入 veRL 的 introduction 了。

## Introduction

- [TODO]

## 