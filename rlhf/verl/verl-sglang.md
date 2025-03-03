# veRL-SGLang Dev Release

## 问题背景

对于 veRL 而言，inference engine 需要支持 SPMD，具体的 motivation 可以参考此[链接](https://github.com/vllm-project/vllm/issues/11400)。

这是  veRL 团队和 SGLang 团队开发的 dev release，旨在将 SGLang 接入 veRL 的训练流程中。目前虽然落后主分支有一定距离，但是会在近期完成合并，欢迎大家尝鲜、体验并且提供反馈。

## 环境配置

### 使用新的虚拟环境

```bash
python3 -m venv ~/.python/verl-sglang
source ~/.python/verl-sglang/bin/activate
```

### 安装 dev 分支的 veRL

```bash
git clone https://github.com/ocss884/verl verl-sglang
cd verl-sglang
git checkout dev_sglang
git pull --no-ff
pip install .
```

### 安装 dev 分支的 SGLang

注意`--config-settings`需要更新到最近的`pip`  
```bash
pip install --upgrade pip
cd ..
git clone https://github.com/sgl-project/sglang.git
cd verl-sglang
pip install -e ../sglang/python "sglang[all]" --config-settings editable_mode=strict --find-links "https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python/" 
```

这个过程可能出现若干问题，这里列出一些常见问题和解决方法：

1. **vllm dependency 冲突**

`ERROR: pip’s dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. verl 0.2 requires vllm<=0.6.3, but you have vllm 0.6.4.post1 which is incompatible.`

可以直接忽视。

2. **flash-attn 不存在**

可能虚拟环境不一定有 wheel 和 packaging，前一步 flash_attn 安装失败。这里需要手动再安装一次：

```bash
pip install wheel, packaging
pip install flash-attn --no-build-isolation --no-deps
```

3. **安装 flash_attn 时出现 CUDA ERROR**

如果出现 `CUDA ERROR`，尝试修改 `CUDA_HOME` 和 `LD_LIBRARY_PATH` 到本地的 cuda，我这里是 `12.4`。

```
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=“/usr/local/cuda-12.4”
```

成功安装后，可以检测下相关库的配置：

- sglang 0.4.3.post2 
- torch2.5.1+cu124  
- flashinfer 0.2.2.post1
- verl 0.2 
- ray 2.42.1  
- flash-attn 2.7.4.post1  

### 安装 megatron 作为 veRL 的 training engine

veRL 目前也支持使用 Megatron 作为 training engine，使用下面的命令安装 dev 版本的 megatron：

```bash
# 安装 Megatron-LM 到当前路径
git clone -b core_v0.4.0_verl https://github.com/eric-haibin-lin/Megatron-LM

# 将 Megatron-LM 添加到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/Megatron-LM
```

需要配套安装 [Transformer Engine 1.7](https://github.com/NVIDIA/TransformerEngine)：

```bash
pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7
```

安装时编译可能遇到一些问题：

1. **could not find cudnn**

```bash
CMake Error at /tmp/pip-req-build-s96o7cy6/3rdparty/cudnn-frontend/cmake/cuDNN.cmake:3 (find_path):
  Could not find CUDNN_INCLUDE_DIR using the following files: cudnn.h
Call Stack (most recent call first):
  CMakeLists.txt:33 (include)
```

[官方的 find path 函数](https://github.com/NVIDIA/cudnn-frontend/blob/1b0b5eac540b7f8fd19b18f1e6b8427c95503348/cmake/cuDNN.cmake)可以看到具体可用的查找方式，手动指定 `cudnn` 的安装路径给 `CUDNN_PATH` 即可，例如：

```bash
export CUDNN_PATH=/usr/local/cuda/cuda-12/cudnn/v8.9.7.29
```

`CUDNN_PATH` 路径下需要可以找到 `include/cudnn.h`。

2. **GCC版本大等于8.1**

参考[这个issue](https://github.com/NVIDIA/TransformerEngine/issues/1270)。编译需要支持 C++17 的 filesystem 头文件，transformer engine 团队内部使用 GCC 13.2.0 进行编译，可以参考下面的命令安装 GCC 13：

```bash
sudo apt update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-13 g++-13
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 60
```

## 测试 PPO 功能

首先构造数据集，默认保存至 `~/data`。

```bash
python3 examples/data_preprocess/gsm8k.py
python3 examples/data_preprocess/math_dataset.py
```

可以直接运行 `bash test_sglang.sh` 测试 SGLang 的 PPO 功能。具体运行的命令如下：

<details>
<summary>运行 PPO 的命令</summary>

```bash
DATA_DIR=$HOME/data/gsm8k
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 2>&1 | tee verl_demo.log
```
</details>


小对拍需要准备一台8卡机器，注意小对拍默认会使用wandb和环境变量WANDB_API_KEY记录训练metrics，如不想记录则删除`trainer.logger=['console','wandb']`中的wandb。该实验预计在8xH100上运行3h40m。命令修改自`examples/ppo_trainer/run_qwen2-7b_seq_balance.sh`。  
可以直接运行`bash run_sgl_qwen2-7b_seq_balance.sh`来启动对拍，具体命令如下
<details>
<summary>SGLag小对拍命令</summary>
set -x
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet
train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"
TIME=$(date +"%Y-%m-%d-%H-%M")

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=2048 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24000 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2-7B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=98304 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_example_gsm8k' \
    trainer.experiment_name="qwen2-7b_sglang_0.4.3.post2_function_rm_bsz8k_p4k_r4k_seq_packing-${TIME}" \
    trainer.n_gpus_per_node=8 \
    +trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
</details>