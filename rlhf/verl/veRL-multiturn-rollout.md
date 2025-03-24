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

### docker 里面安装 nccl

如果遇到 nccl 问题，可能考虑如下方法安装。我们的 ubuntu 是 22.04，cuda 是 12.4，这里需要手动安装 nccl

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

sudo apt update

sudo apt install libnccl2=2.25.1-1+cuda12.4 libnccl-dev=2.25.1-1+cuda12.4
```

### 更新 python

```
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

#### 安装 dev 分支的 veRL

```
cd ~
git clone https://github.com/ocss884/verl verl-sglang
cd verl-sglang
git checkout dev_sglang
git pull --no-ff
python3 -m uv pip install .
```

#### 从github安装最新的 SGLang main branch

```
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

按照上述流程，很有可能缺少 `flash-attn`，这里建议手动安装：

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

### 下载模型

```
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir /root/models/qwen2-7b-instruct --local-dir-use-symlinks False
```

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
nohup python3 -m verl.trainer.main_ppo trainer.experiment_name=qwen7b_sft2_$(now)
```

