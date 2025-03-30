## 背景
目前的 main 上的并行写法等价于限制了 `tp<=8`，为了支持 deepseek 相关的训练，需要 verl engine 的跨 node 支持。

## Proposal

https://gist.github.com/fzyzcjy/01b851e045b970f16e63580b12dbf7ab

## Step by step setup

**1.** 登陆 novita_h20_1

```bash
ssh novita_hwo_1
docker exec -it {YOUR CONTAINER NAME} /bin/zsh
```

**2.** 安装 verl-sglang 需要的依赖

```bash
apt install python3.10-venv

git clone https://github.com/ocss884/verl.git verl-sglang
cd verl-sglang
git switch sglang_multinode
python3 -m venv .venv --upgrade-deps
source .venv/bin/activate

# 在 venv 中需要补充安装的依赖
pip install torch torchvision torchaudio build wheel

# 安装 verl
pip install ".[sglang]"
pip install ".[gpu]"
```

**3.** 设置 Huggingface 的环境变量

novita的磁盘空间不大，注意用以下配置共享模型权重和数据集文件：

```bash
# Huggingface
export HF_DATASETS_CACHE="/model/shared/.cache/huggingface/datasets"
export HF_HOME="/model/shared/.cached/huggingface"
```

**3.** 在 **novita_h20_1** 启动 ray

```bash
ray start --head --dashboard-host=0.0.0.0
```

运行结束后会看到下面的信息：

<img src=../img/gcs-address.png />

记住上面的 GCS address

**4.** 在在 **novita_h20_2** 加入 ray cluster

在 novita_h20_2 重复上面的 1-3 配置好环境后，运行下面的命令，令 node2 加入 cluster

```bash
ray start --address='10.94.16.4:6379'
```

看到下面的信息说明加入成功：

<img src=../img/ray-runtime-start.png />

运行完毕后回到 novita_h20_1，运行：

```bash
ray status
```
可以看到此时有两个 node 加入了！CHEERS！

<img width="506" src=../img/multi-node-status.png />

**5.** 运行实验

多机实验启动脚本，本次实验使用 Qwen2-7B-Instruct，2 机 16 卡。verl 的配置脚本如下，彻底走通还需要 debug：
 
 - rollout DP=4，TP=4
 - actor FSDP

```bash
  set -x

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2-7B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_multinode' \
    trainer.experiment_name='Qwen2-7B-SGLang-0.4.4' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@
```

## 常见问题

**1.** `pip install .[sglang]` 失败  
   这里错误种类很多，不全部列举，只写几个我印象比较深的错误：
  - Cargo 不存在
  - pip resolve deps 耗时特别久后突然 fail  
  ...

请运行下面的命令将你的 `pip` 升级到最新的 `25.0.1`。目前使用旧版（如 py3.10 自带的 pip==22.0）安装 verl-sglang 可能会出现 pip 无法正确解析依赖的情况：

```bash
pip install --upgrade pip
```
