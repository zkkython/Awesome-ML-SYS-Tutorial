# veRL Server：基于 HTTP Server 的 rollout 接口

【disclaim】这篇文章是 yitianlian 参与 SGLang RL 的工作成果，全程有 jhinpan 和 fzyzcjy 的合作，最后 zhaochenyang20 完成了 review，感谢每位参与者的贡献。

为了配合 agentic LLM 的训练，在现有的 PPO/GRPO 算法的基础上，从 single turn rollout 改动为和环境交互的 multi-turn rollout 是非常自然的选择。考虑到这一过程中，由于 enviroment 交互的延迟，turn 之间的等待时间很长，一直用 Engine 做 rollout 的话（`engine.generate`），可能连 continuous batching 都组不起来，所以，改用 server 来通过 https 做 rollout 的需求就呼之欲出了。除此之外，考虑到 enviroment 的交互也常常是通过 https 请求完成的，比如众多 sandbox，就是 enviroment 自己启动一个 sandbox 然后往里面发请求实现的。为了在 training engine，rollout 和 enviroment 三个子进程中保持良好的通讯和交互，避免通过同意，选择 server 势在必行。

## 设计思路一（已废弃）

为实现这一目标，我们将 SGLang 的 `launch_server` 函数改写为 `launch_server_from_verl_engine`，允许我们在已有 `VerlEngine` 初始化的基础上，复用其 `TokenizerManager` 和 `SchedulerInfo`，从而避免重复创建通信管道或资源冲突。
在SGLang系统中，TokenizerManager和SchedulerInfo已经建立了一套完整的进程间通信(IPC)机制，包括ZMQ socket通道，如果不复用这些现有资源而是重新创建，就会建立重复的通信管道，这些额外的通信管道会消耗系统资源(内存、文件描述符等)，并增加系统负担。
当VerlEngine和HTTP Server分别创建自己的TokenizerManager和通信机制时，会出现争用相同资源的情况，下文出现的"第二次调用update_weights_from_tensor卡住"问题，就是资源冲突的具体表现。

## 测试流程

启动新的虚拟环境，这里我们不用 docker，但是还是使用 uv。

```bash
cd ~
python3 -m venv ~/.python/veRL-server
source ~/.python/veRL-server/bin/activate
python3 -m pip install uv

# 安装 sglang
git clone https://github.com/yitianlian/sglang-fork.git
cd sglang-fork
git checkout feature/http_server_engine

# 重要：安装修改后的包
cd python
pip install .
pip install torch_memory_saver

# 测试 veRL Server
cd ../test/srt
python test_verl_engine_server.py
```

我们之前遇到的 `BrokenPipeError: [Errno32] Broken Pipe` 错误已经找到根本原因。这个问题不是由端口冲突或服务器限制引起的，而是由于我们修改了SGLang的verl_Engine.py和http_server.py文件后，没有重新安装更新后的包所导致的。

## 开发思路

### 增加 `launch_server_from_verl_engine`

该函数与 [`launch_server`](https://github.com/sgl-project/sglang/blob/ef9a378a209d970e0b5c48ae3eac6f2660d43faf/python/sglang/srt/entrypoints/http_server.py#L659) 类似，但允许外部传入已有的 `tokenizer_manager` 和 `scheduler_info`，并从 `VerlEngine` 内部启动 HTTP Server。【这么设计的意义是什么，为什么要外部传入？不这么设计的坏处是什么？】

```python
def launch_server_from_verl_engine(
    tokenizer_manager: TokenizerManager,
    scheduler_info: Dict,
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
        )
    )

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            _global_state.tokenizer_manager.image_token_id,
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        warmup_thread.join()



```

### 修改 `VerlEngine.__init__`

我在 `tp_rank == 0` 的进程中，启动了一个新的线程来运行 `launch_server_from_verl_engine`，从而不阻塞主线程的初始化逻辑：

```python

class VerlEngine:
    def __init__(
        self,
        device_mesh_cpu: DeviceMesh,
        nnodes: int = 1,
        **kwargs,
    ):
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._tp_size = device_mesh_cpu.size()
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = Engine(
                **kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes
            )
        else:
            self._engine = None

        if self._tp_rank == 0:
            import copy

            new_server_args = copy.deepcopy(self._engine.server_args)
            new_server_args.port = 30000 + self._tp_rank
            print(f"launch_server_from_verl_engine {new_server_args.port}")

            def server_thread_wrapper(tokenizer_manager, scheduler_info, server_args):
                print(f"Server thread begin")
                launch_server_from_verl_engine(
                    tokenizer_manager=tokenizer_manager,
                    scheduler_info=scheduler_info,
                    server_args=server_args,
                )

            server_thread = threading.Thread(
                target=server_thread_wrapper,
                args=(
                    self._engine.tokenizer_manager,
                    self._engine.scheduler_info,
                    new_server_args,
                ),
                daemon=True,
            )
            server_thread.start()

        dist.barrier(group=self._device_mesh_cpu.get_group())
```

并通过设置 `server_args.port` 为 `30000 + tp_rank` 避免端口冲突。



## 设计思路一遇到的问题

目前的实现中，模型加载和 Server 启动都能成功，第一个 `update_weights_from_tensor` 调用也能顺利完成参数更新。然而在第二次调用该方法时，程序会**卡住不动，最终报出 NCCL 超时错误**。

经过调试发现：

- `scheduler` 在处理完更新任务后，调用了 `send_to_tokenizer.send_pyobj(output)`。
- 但 `tokenizer_manager` 的 `handler_loop` 虽然还在运行，却无法收到该消息，进而造成主进程阻塞。
- 如果注释掉 Server 的启动逻辑（在 `VerlEngine` 初始化时不调用 `launch_server_from_verl_engine`），上述问题完全消失，说明是 server 的某些组件影响了原有的通信逻辑。

## 初步分析

我们推测多线程设计方案存在严重问题，导致无法可靠使用：

1. 线程安全问题：
    - TokenizerManager可能不是线程安全的，当同时被多个线程访问时会导致竞争条件。

2. IPC通道干扰：
    - Server线程(FastAPI/Uvicorn)创建的事件循环与主线程的ZMQ通信通道产生相互干扰。
    - 这会导致pipe的一端被意外关闭，正如上文提到的 `BrokenPipeError: [Errno32] Broken Pipe` 这个报错。

3. GIL限制下的阻塞：
    - Python的GIL(全局解释器锁)在处理I/O密集型任务时，线程切换不及时。
    - 当Server线程长时间占用GIL，会导致TokenizerManager无法及时响应ZMQ消息。

4. 资源分配冲突：
    - 两个线程同时操作网络资源导致端口竞争，这可能与Atlas/Novita服务器上出现的broken pipe错误有关。

## 设计思路二 (现在采用版本)

在这个设计中，存在一个多层委托的调用链：

1. **调用链分析**:
   - test_verl_engine_server.py中调用`engine.update_weights_from_tensor()`
   - 这个engine实际上是VerlEngine的实例
   - VerlEngine内部在初始化时创建了HttpServerEngineAdapter作为其`_engine`属性
   - 当调用VerlEngine的update_weights_from_tensor时，它内部会调用`self._engine.update_weights_from_tensor()`

2. **关键代码连接点**:
   在verl_engine.py中，VerlEngine的初始化有这样一段代码：
   ```python
   if "launch_server" in kwargs and kwargs["launch_server"]:
       # 构建server_args...
       if self._tp_rank == 0:
           self._engine = HttpServerEngineAdapter(server_args)
   ```

   而在test_verl_engine_server.py中启动VerlEngine时有：
   ```python
   engine = VerlEngine(
       # 其他参数...
       launch_server=True
   )
   ```

3. **HTTP服务器的启动和通信**:
   - 当传入`launch_server=True`时，VerlEngine会创建一个HttpServerEngineAdapter
   - HttpServerEngineAdapter会启动一个HTTP服务器进程
   - VerlEngine的update_weights_from_tensor方法会收集所有节点的张量数据
   - 在主节点(tp_rank=0)上，它通过HttpServerEngineAdapter发送HTTP请求来更新权重

4. **分布式协作机制**:
   ```python
   # VerlEngine中的update_weights_from_tensor
   if self._tp_rank == 0:  # 只有主节点发送HTTP请求
       self._engine.update_weights_from_tensor(
           named_tensors=[(name, LocalSerializedTensor(values=gathered_serialized_tensors))],
           # 其他参数...
       )
   ```

这样的设计实现了一个完整的客户端-服务器架构：
- 服务器端是HttpServerEngineAdapter启动的HTTP服务器进程
- 客户端是VerlEngine通过HttpServerEngineAdapter发送的HTTP请求


### 为什么我们不采用 `update_weights_from_distributed` 来更新 Server 参数

尽管 NCCL 通信在多数场景下具备极高的性能，我们在本版本的实现中依然选择不使用 `update_weights_from_distributed`，而是通过 `update_weights_from_tensor` 接口来完成 Server 参数的更新。主要原因如下：

1. **兼容 VerlEngine 现有逻辑**  
   为了与当前 VerlEngine 的实现保持完全兼容，我们需采用 `update_weights_from_tensor` 接口进行参数更新。这一方式可以无缝对接现有的框架，避免对主逻辑产生不必要的干扰。

2. **HTTP 传输性能无需担忧**  
   起初我们担心通过 HTTP 传输 tensor 会成为性能瓶颈。但据与 fzyzcjy 的沟通确认，`update_weights_from_tensor` 实际上传输的仅为 meta data，而非完整 tensor 数据。因此，该方式在性能上也能满足需求，传输效率并不构成实际障碍。

3. **`update_weights_from_distributed` 与 Verl 框架存在设计冲突**  
   当前 `update_weights_from_distributed` 的实现逻辑是：在 rank 0 上保存模型参数，并通过 TCP 将参数广播至其他 ranks（如 rank 1、rank 2）。然而，在 Verl 框架中，HybridEngine 会将 training 与 inference 部署在同一资源池上。这就导致同一个 rank 的同一个端口需同时承担发送与接收任务，进而产生端口冲突。因此，该方法与 VerlEngine 的资源调度方式不兼容，无法直接采用。

综上，`update_weights_from_tensor` 在兼容性、性能和设计合理性方面均更符合当前实现的需求，因此我们选择了这一方案。