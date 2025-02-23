# KV Cache 代码解析
本文档提供了对 SGLang中KV Cache管理的代码实现的全面概述，追踪Request的生命周期以及涉及的关键组件，包括 `Scheduler`、 `Radix Cache`、 `Attention Backend` 和过程中维护的全局状态。

为便于说明，我们将在示例中做出以下假设：
- 我们使用 `FlashInfer` 作为`Attention Backend`
- 我们使用最长前缀（Longest Prefix）作为`waiting_queue`

## 全局状态

本节简要概述了跨Request维护的一些重要全局状态。

### KV Cache与内存池

KV Cache是服务器中最重要的全局状态，因为它可能占用 GPU 内存的很大一部分。KV Cache的管理采用了两级内存池。

#### `req_to_token_pool`
- **用途：** `req_to_token_pool` 将Request映射到其token的 KV Cache 索引。
- **布局：** 最大允许Request数 * 最大允许 token 数
- **访问：**
    - Dim0: `req_pool_indices`
    - Dim1: Request中 token 的位置，从 0、1、2 开始...
    - 值：token 的 `out_cache_loc`
  
#### `token_to_kv_pool`
- **用途：** `token_to_kv_pool` 将 token 的 KV Cache索引映射到其 KV Cache数据，`token_to_kv_pool` 有特定于模型的实现，如 `MHA`、 `MLA`、 `DoubleSparse`。
- **布局：** 层数 * 最大允许 token 数 * HEAD数 * HEAD维度
- **访问：**
    - Dim0: `layer_id`
    - Dim1: `out_cache_loc`
    - Dim2: HEAD
    - Dim3: HEAD维度
    - 值：`k_buffer` 的 `cache_k` 和 `v_buffer` 的 `cache_v`

#### `tree_cache`
- **用途：** 树结构，用于增强跨Request的前缀KV Cache复用。
- **访问：**
  - 键：Token ID
  - 值：Token 的 KV 索引

### Active Request管理

`Scheduler` 组件负责管理Active Request。以下核心全局状态用于在 `Scheduler` 中维护这些Active Request。

#### `waiting_queue`
- **用途：** `waiting_queue`是一个数据结构，设计用于存放Active Request。它根据优先级（Request的最长前缀）或可用内存动态重新排序这些Request，以优化批处理任务。
- **一些额外要点**
  - **入队**
    - 新到达的Request被加入 `waiting_queue`。
    - 从 `retract_decode` 返回的Request。
  - **出队** 当前有最高优先级的Request从队列中出队以形成批次。

#### `new_batch`
- **用途：**：一批准备好进行prefill/extend阶段的Request。
- **一些额外要点**
  - **分块预填充**：如果Request所需的 token 数超过可用内存（`remaining_tokens`），可能会被分块成较小的部分。
  - `new_batch` 中的Request将经历prefill/extend。
  - prefill/extend后，`new_batch` 将过渡到 **全局批次（Global Batch）**，用于下一次迭代。

#### `running_batch`
- **用途：**：一批准备好进行decode阶段的Request。
- **一些额外要点**
  - **Retracted**：如果decode期间可用内存不足，`Scheduler`可能会通过 `retract_decode` 从 `running_batch` 中撤回某些Request，将其返回到 `waiting_queue` 以供后续处理。

#### `cur_batch`
- **用途：**：`Scheduler` 主循环（`run_batch` 函数）中当前正在处理的Request批次。
- **一些额外要点**
  - `cur_batch` 在 `event_loop_normal` 中分配。
  - 形成 `cur_batch` 的逻辑是：如果本轮有准备好预填充的Request（`new_batch`），则使用 `new_batch` 作为 `cur_batch`。否则，`cur_batch` 将处理准备好decode的Request，因此使用 `running_batch` 作为 `cur_batch`。

## `Scheduler`概览

本节提供 `Scheduler` Request管理过程的高层概述。

### **`Scheduler`**
![alt text](scheduler_overview.png)
该图展示了 **`Scheduler`** 如何将Request从 `waiting_queue` 过渡到 `new_batch`（用于prefill/extend阶段），然后进入 `running_batch`（用于decode阶段）。

1. **新Request到达**：`Scheduler` 持续调用 `recv_requests` 以收集新到达的Request，验证它们并将其放入 `waiting_queue`。在我们的示例中，`Req 7` 被接收并入队。

2. **合并批次**：在为本轮形成新批次之前，`Scheduler` 会将上一轮的 `cur_batch` 合并到 `running_batch` 中。在图中，上一轮的 `cur_batch` 显示为 `cur_batch(i-1)`，`running_batch` 显示为 `running_batch(i-1)`。在
我们的示例中，`Req 0` 和 `Req 1` 将合并到新的 `running_batch` 中。**合并批次** 还会移除上一轮的 `being_chunked_request`。`being_chunked_request` 是在 `get_new_batch_prefill` 过程中生成的分块预填充Request。（在图中，有已完成的 `being_chunked_request`，如 `Req 5a`，表示 `Req 5` 的第一部分），我们会移除它，因为我们不希望它们进入decode阶段。）

3. **形成新批次**：`Scheduler` 会检查是否可以形成一个 `new_batch`（在 `get_new_batch_prefill` 中），所有能适应可用内存的Request将被打包到批次中。如果最后一个放入批次的Request大小超过剩余可用内存，该Request将被分块为 `being_chunked_request`。在我们的示例图中，`Scheduler` 从 `waiting_queue` 中拉取Request并创建一个 `new_batch`（如 `Req 6`、`Req 5b`，`Req 5b` 是 `being_chunked_request`），并将 `new_batch` 用作 `cur_batch`。图中未展示，但如果没有 `new_batch`，`running_batch` 将被过滤（例如，`Req 1`、`Req 0` 将被保留，而 `Old Finished Req` 将被移除），然后用作 `cur_batch`。此外，如果 GPU 内存不足，某些decode Request可能会根据特定策略被retracted。在 `retract_decode` 阶段，图中 `Req 0` 被撤回。

4. **运行批次**：一旦确定了 **全局批次**，调用 `run_batch` 执行一次前向传递。

5. **结果处理**：在 `run_batch` 之后，`Scheduler` 调用 `process_batch_result` 来确定哪些Request已完成，哪些继续进行。在我们的示例中，`Req 6` 完成并变为灰色，`Req 5b` 仍未完成。

6. **迭代**：循环重复，直到所有Request最终完成。如果遇到内存不足，Request可能会被chuncked（prefill/extend）或在retracted（decode），然后重新插入 `waiting_queue` 以供后续处理。