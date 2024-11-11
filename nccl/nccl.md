# NCCL Tutorial

## What is NCCL

NCCL (NVIDIA Collective Communications Library) 是 NVIDIA 推出的一个用于 GPU 之间高性能通信的库。随着深度学习模型规模的增长（如 GPT-3 的 1750 亿参数），单个 GPU 已无法满足训练需求。这就需要将模型或数据分割到多个 GPU 上进行并行训练，而 GPU 之间必然需要进行数据交换。NCCL 就是为了解决这个场景而生的。它主要解决以下问题：

- 在多 GPU 训练场景下实现高效的数据交换
- 自动识别并优化 GPU 间的通信拓扑
- 提供标准化的集合通信接口
- 支持机内（单机多卡）和机间（多机多卡）通信

## NCCL 重要概念

- Collective Operations：NCCL 支持各种集体通信操作，如广播（Broadcast）、规约（Reduction）、聚合（Aggregation）、AllReduce、AllGather 等。这些操作可以在多个 GPU 或节点之间进行数据同步和合并。
- Processes and Groups：NCCL 中的进程（Processes）表示参与通信的计算节点。进程可以组织成组（Groups），以便在组内进行集体通信。
- Communicators：Communicators 是 NCCL 中用于定义进程之间通信关系的对象。它指定了参与通信的进程组和通信模式（如点对点、广播等）。
- Devices and Streams：NCCL 与 GPU 设备密切相关，它可以在多个 GPU 设备之间进行通信。同时，NCCL 还支持与 CUDA 流（Streams）的集成，以实现异步通信和并行计算。
- Synchronization：在分布式计算中，同步是至关重要的。NCCL 提供了各种同步原语，如 barrier 同步，以确保进程在执行集体通信操作时达到一致的状态。
- Performance Optimization：NCCL 注重性能优化，它提供了一些技术来提高通信效率，如集体通信的合并、数据传输的批量处理、通信与计算的重叠等。
- Fault Tolerance：NCCL 还考虑了容错性，支持在部分节点故障或网络不稳定的情况下进行可靠的通信。

## NCCL 支持的操作类型

NCCL 支持以下几种主要的集合通信操作：

1. **AllReduce**：所有 GPU 的数据先进行规约（如求和），然后广播到所有 GPU
2. **Broadcast**：从一个源 GPU 向其他所有 GPU 广播数据
3. **Reduce**：将所有 GPU 的数据规约到一个目标 GPU
4. **AllGather**: 收集所有 GPU 的数据并分发给所有 GPU
5. **ReduceScatter**: 规约后将结果分散到所有 GPU
6. **Send/Recv**: 点对点通信
7. **AllToAll**: 将数据分发到所有 GPU

## NCCL 的设备要求

- NVIDIA GPU (支持 CUDA)
- 推荐使用支持 NVLink 的 GPU 以获得最佳性能
- 对于多机通信，建议使用 InfiniBand 或 RoCE 网络
- 需要安装对应版本的 CUDA 和 GPU 驱动

## NCCL in PyTorch

PyTorch 内置 NCCL 后端支持，使用非常简单：

```python
import torch.distributed as dist

# 初始化进程组
dist.init_process_group(backend='nccl')

# 创建分布式模型
model = torch.nn.parallel.DistributedDataParallel(model)
```

## NCCL 的工作方式

**Ring Algorithm**

NCCL 在机内通信时主要使用 Ring Algorithm。其核心思想是：

1. 将所有 GPU 组织成一个环
2. 每个 GPU 只与其相邻的 GPU 通信
3. 通过多轮次传递实现全局数据交换
4. 数据被分成多个 chunk 进行并行传输

优点：
- 充分利用 GPU 间的带宽
- 通信负载均衡
- 易于扩展

## Tree Algorithm

在[torch-distributed 的后记](../torch-distributed/readme.md#ring-all-reduce-and-tree-all-reduce)中已经介绍过了。

## 通信协议

NCCL 实现了三种通信协议：

1. **Simple**: 基础协议
2. **LL(Low Latency)**: 低延迟协议，适用于小数据量
3. **LL128**: 在 NVLink 环境下的优化协议，可达到 93.75% 的有效带宽


## NCCL 与其他通信库的对比

1. **与 MPI 的区别**:
   - NCCL 专注于 GPU 通信优化
   - MPI 更通用但性能可能较低
   - 可以结合使用：MPI 管理进程，NCCL 负责 GPU 通信

2. **与 Gloo 的区别**:
   - Gloo 支持 CPU 和 GPU
   - NCCL 在 GPU 通信性能上更优
