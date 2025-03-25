# veRL-multiturn-rollout Dev Release

## 环境配置

### 创建新的 docker

```
docker run -it --name h100_{your_name} --gpus all \
    --shm-size 32g \
    -v /opt/dlami/nvme/.cache:/root/.cache \
    --env "HF_TOKEN={your_hf_token}" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

以后每次从 docker 里面 `exit` 出来，再用这个指令可以重启：

```
docker start -i h100_{your name}
```

### 更新 python

```
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
```

### 使用虚拟环境

```
# 创建虚拟环境
python3 -m venv ~/.python/veRL-multiturn-rollout

# 激活虚拟环境
source ~/.python/veRL-multiturn-rollout/bin/activate

# 安装 uv
python3 -m pip install uv
```

### clone veRL-multiturn-rollout 并切换到 async-tp 分支

```
cd ~
git clone https://github.com/zyzshishui/veRL-multiturn-rollout.git
cd veRL-multiturn-rollout
git checkout async-tp
```

### 配置python环境

#### 从github安装最新的 SGLang main branch

```
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

手动安装 `flash-attn`：

```
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

#### 安装requirements.txt

```
python3 -m uv pip install -r ./requirements.txt
```

## 4 卡测试 SGLang

### 使用前需要配置好 `WANDB_API_KEY`

可以参考[这个过程](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。

```
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# 定义这个时间戳函数
function now() {
    date '+%Y-%m-%d-%H-%M'
}
```

### 运行

```
nohup python3 -m verl.trainer.main_ppo trainer.experiment_name=qwen7b_sft2_$(now) > log_$(now).txt
```

