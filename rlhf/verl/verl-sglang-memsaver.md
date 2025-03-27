# VeRL-SGLang-memsaver Installation and Testing Guide
## 背景
SGLang进入训练框架之后，需要保持`enable_memory_saver=True`开启以便在rollout结束后释放显存给别的模块使用

## verl 完整的memory saver PR需要
- [x] memory saver找不到头文件安装失败 https://github.com/fzyzcjy/torch_memory_saver/pull/2
- [x] 跨进程传tensor错误 https://github.com/sgl-project/sglang/pull/4565
- [ ] 更新verl-sglang镜像
- [x] 在verl engine中默认开启memory saver
- [ ] 更新verl对sglang rollout的依赖

verl的相关pr，持续更新中：
- https://github.com/volcengine/verl/pull/756

## 环境配置

### 创建新的 Python 虚拟环境

```bash
python3 -m venv ~/.python/veRL-mem-saver
source ~/.python/veRL-mem-saver/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade uv
```

## 安装和配置

### 安装 veRL 和依赖

```bash
# 克隆 veRL 代码库
git clone https://github.com/SwordFaith/verl.git
cd verl

# 安装 veRL 的SGLang依赖
python3 -m uv pip install -r requirements_sglang.txt

# 安装最新的SGLang主分支
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

### 创建数据集

```bash
# 创建 GSM8K 数据集
cd ~/verl  # 确保你在 verl 目录下
python examples/data_preprocess/gsm8k.py
```

### 修改 sglang_rollout.py 文件

找到 `sglang_rollout.py` 文件（通常在 `verl/verl/workers/rollout/` 目录下），在 VerlEngine 初始化部分添加 `enable_memory_saver=True` 参数：

```python
self.inference_engine = VerlEngine(
    model_path=actor_module,
    dtype=config.dtype,
    mem_fraction_static=config.gpu_memory_utilization,
    device_mesh_cpu=device_mesh_cpu["tp"],
    base_gpu_id=0,
    gpu_id_step=1,
    enable_memory_saver=True,  # 添加此行
    # NOTE(Chenyang): if you want to debug the sglang engine
    # please set the following parameters
    # Otherwise, it will make the engine run too slow
    # log_level="INFO",
    # log_requests=True,
    # log_requests_level=2,
    # max_running_requests=1,
)
```

## 运行测试

使用以下命令运行测试脚本：

```bash
DATA_DIR=$HOME/data/gsm8k
TIME=$(date +"%Y-%m-%d-%H-%M")
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.name=sglang \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=1 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2-7B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=16 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 2>&1 | tee verl_demo.log
```

运行脚本后，日志文件 `verl_demo.log` 将保存在当前目录中。您可以使用以下命令查找它：

```bash
find / -name verl_demo.log 2>/dev/null
```

## 常见问题

Q: 如下报错，成因是某个rank崩溃导致进程关闭

```shell
ray.exceptions.RayTaskError(RuntimeError): ray::WorkerDict.actor_rollout_generate_sequences() (pid=42249, ip=100.106.32.178, actor_id=a6552c4ca07c0202f4b6705701000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7f77f43bca30>)
  File "/data/gpu-use/sw-verl/verl/single_controller/ray/base.py", line 419, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/data/gpu-use/sw-verl/verl/single_controller/base/decorator.py", line 404, in inner
    return func(*args, **kwargs)
  File "/data/gpu-use/sw-verl/verl/workers/fsdp_workers.py", line 500, in generate_sequences
    with self.rollout_sharding_manager:
  File "/data/gpu-use/sw-verl/verl/workers/sharding_manager/fsdp_sglang.py", line 91, in __enter__
    self.inference_engine.update_weights_from_tensor([(k, v) for k, v in params.items()], load_format=None)
  File "/data/gpu-use/sw-verl/.venv/lib/python3.10/site-packages/sglang/srt/entrypoints/verl_engine.py", line 112, in update_weights_from_tensor
    dist.gather_object(
  File "/data/gpu-use/sw-verl/.venv/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 83, in wrapper
    return func(*args, **kwargs)
  File "/data/gpu-use/sw-verl/.venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 2825, in gather_object
    all_gather(object_size_list, local_size, group=group)
  File "/data/gpu-use/sw-verl/.venv/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 83, in wrapper
    return func(*args, **kwargs)
  File "/data/gpu-use/sw-verl/.venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 3346, in all_gather
    work.wait()
RuntimeError: [../third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [100.106.32.178]:22676
```
A: 确认是否安装了最新的SGLang main branch。特别需要注意要在安装的SGlang的`sglang/srt/entrypoints/verl_engine::VerlEngine`里找到`monkey_patch_torch_reductions`


<img width="628" alt="Screenshot 2025-03-28 at 1 32 31 AM" src="https://github.com/user-attachments/assets/61f0b74b-6362-491f-8063-908a90fd5808" />

