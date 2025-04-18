# veRL-multiturn-rollout Dev Release

## 环境配置

### 创建新的 docker

```bash
# 如果你的系统没有配置过 HF_TOKEN 和 WANDB_API_KEY，请先配置好
# 这里的 cache 映射路径是在 atlas 集群上，如果需要使用自己的路径，请自行修改
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v /.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

以后每次从 docker 里面 `exit` 出来，再用这个指令可以重启：

```bash
docker start -i h100_{your_name}
```

### 更新 python

```bash
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
```

### 使用虚拟环境

```bash
# 创建虚拟环境
python3 -m venv ~/.python/veRL-multiturn-rollout

# 激活虚拟环境
source ~/.python/veRL-multiturn-rollout/bin/activate

# 安装 uv
python3 -m pip install uv
```

### clone 龙老师的分支

```bash
cd ~
git clone -b feat/add_async_sglang_multi_turn_support https://github.com/SwordFaith/verl.git
cd verl
```

### 配置 python 环境

先安装 veRL：

```bash
python3 -m uv pip install .
python3 -m uv pip install -r ./requirements_sglang.txt
```

安装 SGLang：

```bash
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

手动安装 flash-attn：

```bash
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

## 测试 SGLang

```bash
wandb login

# Set GPUs and run with better log organization
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_fsdp_multiturn.sh
```
