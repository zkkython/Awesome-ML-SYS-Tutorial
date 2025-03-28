# veRL Server：基于 HTTP Server 的 rollout 接口

【disclaim】这篇文章是 yitianlian 参与 SGLang RL 的工作成果，全程有 jhinpan 和 fzyzcjy 的合作，最后 zhaochenyang20 完成了 review，感谢每位参与者的贡献。

为了配合 agentic LLM 的训练，在现有的 PPO/GRPO 算法的基础上，从 single turn rollout 改动为和环境交互的 multi-turn rollout 是非常自然的选择。考虑到这一过程中，由于 enviroment 交互的延迟，turn 之间的等待时间很长，一直用 Engine 做 rollout 的话（`engine.generate`），可能连 continuous batching 都组不起来，所以，改用 server 来通过 https 做 rollout 的需求就呼之欲出了。除此之外，考虑到 enviroment 的交互也常常是通过 https 请求完成的，比如众多 sandbox，就是 enviroment 自己启动一个 sandbox 然后往里面发请求实现的。为了在 training engine，rollout 和 enviroment 三个子进程中保持良好的通讯和交互，避免通过同意，选择 server 势在必行。

为实现这一目标，我们将 SGLang 的 `launch_server` 函数改写为 `launch_server_from_verl_engine`，允许我们在已有 `VerlEngine` 初始化的基础上，复用其 `TokenizerManager` 和 `SchedulerInfo`，从而避免重复创建通信管道或资源冲突。【这里能解释下什么是通信管道浪费和资源冲突么？可能和 tom 老师之前的神之一笔有关？分享失败经验是非常重要的😂】

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
python3 -m uv pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# 测试 veRL Server

cd test/srt
python test_verl_engine_server.py
```

【在 atlas H100 和 novita H20 上全是 broken pipe，但是 SGLang CI 可以过，很奇怪？】

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



## 当前遇到的问题

目前的实现中，模型加载和 Server 启动都能成功，第一个 `update_weights_from_tensor` 调用也能顺利完成参数更新。然而在第二次调用该方法时，程序会**卡住不动，最终报出 NCCL 超时错误**。

经过调试发现：

- `scheduler` 在处理完更新任务后，调用了 `send_to_tokenizer.send_pyobj(output)`。
- 但 `tokenizer_manager` 的 `handler_loop` 虽然还在运行，却无法收到该消息，进而造成主进程阻塞。
- 如果注释掉 Server 的启动逻辑（在 `VerlEngine` 初始化时不调用 `launch_server_from_verl_engine`），上述问题完全消失，说明是 server 的某些组件影响了原有的通信逻辑。

## 初步分析

我推测该问题可能与多线程环境下资源抢占或 ZMQ 通信冲突有关。例如：

- Server 启动后，FastAPI 或 Uvicorn 可能创建了新的事件循环或通信通道，影响了 `TokenizerManager` 原有的 IPC 通道。
- `TokenizerManager` 虽然仍在运行，但其内部 ZMQ socket 的消息接收能力可能受到了主线程资源或 GIL 的竞争影响。

在经历了多线程的问题之后，我们换了一种新的解决方法：

# 全新设计

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

