# SGLang 代码讲解 (SGLang Code Walk Through)

本文档为开发者提供 SGLang 后端代码的开发级指导，下图简要介绍了请求处理的路径：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./sglang-architecture.svg" alt="SGLang 架构图" style="width: 100%; height: auto;">
</div>

具体而言，请求的处理流程如下：

1. 用户启动服务器，初始化 FastAPI 应用、TokenizerManager、DetokenizerManager 和 Scheduler，每个组件均运行无限循环的事件处理程序。

2. 用户向 FastAPI Server 发送 `/v1/chat/completions` 请求，服务器通过 `v1_chat_completions` 端点将请求路由到 TokenizerManager。

3. `v1_chat_completions` 函数将请求转换为 `ChatCompletionRequest`，再转换为 `GenerateReqInput`，并调用 TokenizerManager 的 `generate_request` 方法。

4. TokenizerManager 对请求进行分词处理，并以 Python 对象（`pyobj`）形式将其转发给 Scheduler，同时调用 `_wait_one_response`。

5. Scheduler 在其无限循环 `event_loop_normal` 中处理请求：
   - Scheduler 通过 `recv_requests` 接收请求，调用 `process_input_requests` 处理输入，通过 `handle_generate_request` 管理生成逻辑，并将其加入 `waiting_queue`。
   - 从 `waiting_queue` 中，Scheduler 使用 `get_next_batch_to_run` 为即将处理的请求创建 `ScheduleBatch`。
   - Scheduler 执行 `run_batch` 函数，将 `ScheduleBatch` 转换为 `ModelWorkerBatch`。
   - Scheduler 调用 TpModelWorker 的 `forward_batch_generation`，等待 `logits_output` 和 `next_token_ids`。
   - TpModelWorker 初始化 `ForwardBatch`，将其转发至 ModelRunner，并等待 `logits_output`。
   - ModelRunner 处理 `ForwardBatch`，调用 `forward_extend` 执行模型的前向计算。
   - 模型通过 `AttentionBackend` 加速生成 logits，返回给 ModelRunner，进而返回给 TpModelWorker。
   - TpModelWorker 从 ModelRunner 接收 `logits_output`，调用 ModelRunner 的 `sample` 方法生成 `next_token_ids`，并将其发送回 Scheduler。
   - Scheduler 通过 `process_batch_result` 处理批次结果，并通过 `check_finished` 检查完成状态。
   - 对于已完成的请求，`process_batch_result` 函数将其加入缓存（`tree_cache.cache_finished_req(req)`），并将输出通过 Scheduler 的 `stream_output` 发送。
   - 在 `stream_output` 中，Scheduler 处理输出，将其包装成 `BatchTokenIDOut`，并发送给 DetokenizerManager。

6. DetokenizerManager 在其事件循环中接收 `BatchTokenIDOut`，处理后生成 `BatchStrOut` 并返回给 TokenizerManager。

7. TokenizerManager 在其事件循环中接收结果，通过 `handle_loop` 处理并更新内部状态，然后将响应返回给服务器。

8. FastAPI Server 打包响应并将其返回给用户。

## 致谢与许可证

本文基于版本 [v0.4.0](https://github.com/sgl-project/sglang/tree/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751) 的代码编写。特别感谢 [Chenyang Zhao](https://zhaochenyang20.github.io/Chayenne/)、[Wenxuan Tan](https://github.com/Edenzzzz)、[Simon Veitner](https://simveit.github.io/)、[Shuai Shi](https://shuaills.github.io/)、[Shizhe Diao](https://shizhediao.github.io/)、[Shending Hu](https://shengdinghu.github.io/)、[Xiaoyu Zhang](https://github.com/BBuf)、[agiping](https://github.com/agiping)、[Zhizhou Sha](https://jamessand.github.io/) 对本文档的贡献。

**注意：本文档仍在编写中，以下部分将在后续加入：**

1. 基于 Attention Backend 的 Radix Cache 管理。
2. `get_next_batch_to_run`：如何为每批次请求提取和写入 KV 缓存。
3. `get_model_worker_batch`。
4. `write_req_to_token_pool_trition`。
5. 使用 CUDA Graphs 优化 Attention Backend。
6. 重叠调度策略。



## 启动服务器 (Launch Server)

SGLang 提供 SRT（SGLang Runtime）服务器用于[服务 HTTP 请求](https://sgl-project.github.io/start/send_request.html)以及一个用于[离线模型执行的引擎](https://sgl-project.github.io/backend/offline_engine_api.html)。核心函数 [`launch_server`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L507) 和 [`launch_engine`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L418) 均定义在 [server.py](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py) 中。其中，`launch_engine` 函数负责初始化核心 SRT 服务器组件。

1. 配置环境：设置日志、服务器参数、CUDA/NCCL 环境变量以及进程间通信端口，下载模型和分词器。
2. 启动 Scheduler 进程：每个 Scheduler 启动一个 TpModelWorker 以进行预填充 (prefill) 和解码 (decode)，管理 Radix 缓存，并在无限事件循环中处理来自 TokenizerManager 的请求。如果 `dp_size > 1`，则运行 `run_data_parallel_controller_process`；否则，为每个 `tp_rank` 初始化一个 Scheduler。
3. 作为子进程运行 TokenizerManager 和 DetokenizerManager：前者将数据分词以供 Scheduler 使用，后者将 Scheduler 的输出去分词并返回给服务器前端。在多节点推理中（例如，在两个节点上使用 8 张 H100 部署 Llama 3.1 405B），TokenizerManager 和 DetokenizerManager 仅在第一个节点运行。
4. 如果指定了应用聊天模板，等待 Scheduler 进程发出准备就绪信号，并收集其配置信息。

需要注意的是，在版本 0.4.0 中，[DataParallelController](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/data_parallel_controller.py#L52) 用于在多个数据并行副本之间以轮询方式调度请求。未来，我们计划将其更换为 [SGLang Router](https://sgl-project.github.io/router/router.html) 来实现这个功能。

## 转发请求 (Forward Requests From Server)

服务器使用 FastAPI 应用定义 API 端点，通过 [v1_chat_completions](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/openai_api/adapter.py#L1101) 将 [`/v1/chat/completions`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L354) 请求转发至 TokenizerManager。

1. 从 `raw_request` 中解析 JSON 数据为 `ChatCompletionRequest`，将其转换为 `GenerateReqInput`，并通过 `v1_chat_generate_request` 配置 `sampling_params`。
2. 调用 TokenizerManager 的 `generate_request` 方法，根据 `stream` 参数处理流式 (streaming) 或非流式 (non-streaming) 响应。
3. 对于流式响应，使用 `generate_stream_resp` 逐步处理 `generate_request` 的输出；对于非流式响应，等待结果并通过 `v1_chat_generate_response` 转换为 `ChatCompletionResponse`。

## TokenizerManager 中的请求生成 (Generate Request In TokenizerManager)

[TokenizerManager](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tokenizer_manager.py#L88) 由服务器主进程中的 [`launch_server`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L507) 初始化，用于对请求进行分词处理。

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tokenizer_manager.py#L88)

1. 设置 [ZeroMQ](https://libzmq.readthedocs.io/en/latest/) 进行进程间通信，包括用于与 DetokenizerManager 和 Scheduler 交互的套接字。
2. 配置 `server_args`，启用 `metrics`，并初始化 `model_config`、`tokenizer` 以及多模态图像处理器的占位符。

### [generate_request](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tokenizer_manager.py#L173)

1. 如果尚未初始化事件循环，则创建一个。
2. 如果模型权重正在通过 `update_weights_from_disk` 或 `update_weights_from_distributed` 更新，则暂停处理。
3. 验证请求是否与模型的 `is_generation` 设置兼容。
4. 使用 `normalize_batch_and_arguments` 对请求进行标准化处理，以管理批处理、并行采样以及默认参数。
5. 对单个请求，通过 `_tokenize_one_request` 进行分词处理，将请求发送至 Scheduler，并通过 `_wait_one_response` 等待响应。
6. 对批处理请求，通过 `_handle_batch_request` 方法进行处理：分词输入、管理并行采样、与 Scheduler 交互，并在流式和非流式模式下生成响应。


## Scheduler 接收请求以及批处理 (Scheduler Receive Requests and Process Batches)

[Scheduler](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L97) 作为服务器的子进程运行，通过 `run_scheduler_process` 初始化，并通过 `event_loop_normal` 或 `event_loop_overlap` 执行无限事件循环。

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L97)

1. 配置 [ZeroMQ](https://libzmq.readthedocs.io/en/latest/) 用于与 TokenizerManager 的通信以及响应处理。
2. 设置 `server_args`、`port_args`、`model_config`、`sessions`，并根据重叠调度方式初始化 TpModelWorker 或 TpModelWorkerClient。
3. 初始化分词器和处理器，使用 ChunkCache 或 RadixCache 进行缓存管理，配置 SchedulePolicy。
4. 配置 Chunk 预填充参数，并为请求处理初始化 GrammarBackend。

### [Event Loop](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L376)

Scheduler 不断执行事件循环，在 `process_input_requests`、`get_next_batch_to_run`、`run_batch` 和 `process_batch_result` 之间循环。

### [process_input_requests](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L508)

遍历接收到的请求，识别其类型并将其分派给相应的处理程序。

### [get_next_batch_to_run](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L768)

1. 如有必要，将 `last_batch` 与 `running_batch` 合并，并通过 `get_new_batch_prefill` 优先处理预填充批次。
2. 如果没有预填充批次，则更新用于解码批次的 `running_batch`，包括过滤请求、管理内存和调整解码参数。

### [run_batch](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L956)

1. 对于生成模型，使用 TpModelWorker 的 `forward_batch_generation` 进行标记预测，或在空闲任务中使用 `forward_batch_idle`，并将结果返回至 `event_loop_normal`。
2. 对于嵌入或奖励模型，执行 `forward_batch_embedding`，并返回嵌入结果。

### [Processing and Finalizing Results](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L987)

在推理引擎中，LLM 推理通常分为预填充（Prefill）和解码（Decode）阶段，分别适用于不同的计算特性。关于预填充和解码的概念，可以参考 HuggingFace 的 [这篇文章](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi)。在 SGLang 中，大多数情况下使用 [扩展模式（extend mode）](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py) 代替预填充模式。

- **预填充模式（Prefill）：** 初始化新请求的 KV-Cache，通常使用 Paged KV-Cache。
- **扩展模式（Extend）：** 增量更新现有的 KV-Cache，常利用 Ragged Tensors 提高效率，特别适用于长序列或多轮任务。

### [process_batch_result](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L987)

在执行完 `run_batch` 后，Scheduler 在 `event_loop_normal` 中处理批量结果：

1. **Decode 模式**：处理输出，更新请求状态，处理标记和概率数据，管理内存，并记录统计信息。
2. **Extend 模式**：处理预填充结果，处理输入标记，并为进一步解码或嵌入做准备。
3. 已完成的请求通过 `cache_finished_req` 缓存，并流式传输到 DetokenizerManager。未完成的请求会被更新，并循环回 `get_next_batch_to_run` 进行进一步处理，直至完成。

需要注意的是，LLM 推理通常分为预填充（Prefill）和解码（Decode）阶段，因其计算特性不同。你可以参考 HuggingFace 的[这篇文章](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi)以了解预填充和解码的概念。在 SGLang 中，大多数情况下使用的是 [extend mode](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py)，而不是预填充模式。预填充模式为新请求初始化 KV-Cache，通常使用 Paged KV-Cache。而 Extend 模式则增量更新现有的 KV-Cache，通常利用 Ragged Tensors 提高效率，这使其非常适合处理长序列或多轮任务。

## TpModelWorker 管理前向计算和标记采样 (TpModelWorker Manage Forward and Token Sampling)

[TpModelWorker](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tp_worker.py#L40) 负责管理 ModelRunner 对由 Scheduler 调度的批量请求的前向计算和采样操作。

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tp_worker.py#L40)

1. 初始化分词器、模型配置和 ModelRunner。
2. 配置设备设置和内存池限制。

### [forward_batch_generation](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tp_worker.py#L148)

1. 创建一个 `ForwardBatch`，通过 ModelRunner 的 `forward` 计算 logits，并使用 `sample` 进行下一个标记的采样。
2. 将 `logits_output` 和 `next_token_ids` 返回给 Scheduler，用于 `process_batch_result`。

### [forward_batch_embedding](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tp_worker.py#L160)

1. 创建一个 `ForwardBatch`，通过 ModelRunner 的 `forward` 获取 `logits_output` 和 `embeddings`。
2. 跳过 `forward_batch_generation` 中的采样过程，直接将 `embeddings` 返回给 Scheduler。

## ModelRunner 管理模型执行 (ModelRunner Manages Model Execution)

[ModelRunner](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/model_executor/model_runner.py#L66) 初始化注意力后端并管理加载的模型，以执行生成和嵌入任务的前向传递。

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/model_executor/model_runner.py#L66)

初始化分布式环境，加载模型，应用张量并行性，并设置内存池和注意力后端。

### [forward](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/model_executor/model_runner.py#L675)

`forward` 函数根据 `forward_mode` 决定适当的前向模式来处理批次：

1. `forward_decode`：初始化前向元数据并调用模型的 `forward`，传入输入 ID 和位置。
2. `forward_extend`：初始化前向元数据并调用模型的 `forward` 进行生成或嵌入任务。
3. `forward_idle`：当前向模式为空闲时，管理前向传递。


## 模型加载权重并执行前向传递 (Model Load Weights and Perform Forward)

ModelRunner 的 `self.model` 是模型类的一个实例。所有 [支持的模型](https://sgl-project.github.io/references/supported_models.html) 都可以在 [python/sglang/srt/models](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/models) 中找到。我们以 [Qwen2ForCausalLM](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/models/qwen2.py#L269) 为例。

[`Qwen2ForCausalLM`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/models/qwen2.py#L269) 的结构如下：

* `model`：用于前向传递的权重。
* `embed_tokens`：将 `input_ids` 转换为 `embeddings`。
* `lm_head`：将隐藏状态映射回词汇空间。
* `logits_processor`：操作 `logits`，执行任务如采样和归一化。
* `pooler`：用于提取嵌入或计算奖励的池化机制。

### [forward](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/models/qwen2.py#L289)

`Qwen2ForCausalLM` 中的 `forward` 函数处理输入序列，生成用于下一个 token 预测的 logits，或用于奖励/嵌入请求的嵌入：

1. 使用 `embed_tokens` 将 `input_ids` 转换为嵌入。将嵌入依次通过 Qwen2DecoderLayer 层进行前向传递。
2. 如果 `get_embedding` 为 True，则通过 `pooler` 返回嵌入；否则，使用 `logits_processor` 计算 `logits` 并返回。

加速的关键来自于 `forward_batch` 与 AttentionBackend 之间的交互。

## AttentionBackend 加速模型前向传递 (AttentionBackend Accelerate Model Forward)

SGLang 支持多个 [Attention Backends](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention)，这些后端加速模型的前向传递和键值缓存重用。我们以 [FlashInferBackend](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py) 为例。

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L48)

1. 配置滑动窗口和跨注意力场景的包装器。
2. 分配必要的工作空间和键值索引缓冲区。
3. 为高效的注意力计算准备前向元数据。
4. 集成 CUDA 图支持以优化执行路径。

### [init_forward_metadata](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L130)

1. 解码模式：使用 `indices_updater_decode` 更新解码的索引，并设置 `forward_metadata` 以使用 `decode_wrappers`。
2. 扩展模式：根据 token 数量和包装器数量确定是否需要稀疏前向传递，然后使用 `indices_updater_prefill` 更新索引。
3. 元数据分配：设置 `forward_metadata`，为稀疏使用和前缀扩展设置标志。

### [forward_extend](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L223) 和 [forward_decode](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L277)

1. 选择合适的包装器，并根据情况在 `forward_extend` 中决定使用稀疏或分页注意力，或者在 `forward_decode` 中选择解码包装器。
2. 计算注意力，管理键值缓存，并返回 reshaped 后的输出。

## DetokenizerManager 进行解码并发送至 TokenizerManager (DetokenizerManager Detokenize and Send to TokenizerManager)

[DetokenizerManager](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/detokenizer_manager.py#L53) 由 [`launch_server`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L507) 初始化为 Server 的子进程，用于解码请求。

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/detokenizer_manager.py#L53)

设置通信套接字和分词器。使用 `LimitedCapacityDict` 管理解码状态。

### [event_loop](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/detokenizer_manager.py#L93) 和 [trim_eos](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/detokenizer_manager.py#L78)

1. 接收来自 Scheduler 的处理请求，直接转发 `BatchEmbeddingOut` 或处理 `BatchTokenIDOut` 进行解码。
2. 将 token ID 拆分为 `read_ids` 和 `surr_ids`。使用 `batch_decode` 将 token ID 转换为文本。更新 `DecodeStatus`，包括新的偏移量和解码文本。
3. 在停止序列处修剪输出，将解码后的文本与元数据合并成 `BatchStrOut`，并发送给 TokenizerManager。

## [FastAPI 整理并输出 (FastAPI Wraps the Output)](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L287)

1. DetokenizerManager 通过 [ZeroMQ](https://libzmq.readthedocs.io/en/latest/) 将 `BatchStrOut` 发送到 TokenizerManager。
2. TokenizerManager 更新请求状态并为 FastAPI 准备解码后的文本。
3. 最后，在 FastAPI 中，对于流式传输，使用异步生成器和 `StreamingResponse` 将响应发送给用户。
4. 对于非流式传输，收集并使用 `ORJSONResponse` 发送完整响应，并返回给用户。
