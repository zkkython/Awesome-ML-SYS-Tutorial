# veRL-multiturn-rollout Dev Release

还在调试，这个在 sglang 的 atals 集群上运行，尚未完成开发。

## 环境配置

### 创建新的 docker

```bash
docker run -it --name h100_verl_multiturn --gpus all \
    --shm-size 32g \
    -v /.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

以后每次从 docker 里面 `exit` 出来，再用这个指令可以重启：

```bash
docker start -i h100_verl_multiturn
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

### clone veRL-multiturn-rollout 并切换到 async-tp 分支

PS：这个分支还是 private 的，需要先我们内部邀请加入，不然就下载下源代码，直接传到 docker 里面。我其实自己在 atals 的 `/.cache` 里面偷偷塞了一个，所以可以直接用：

```bash
cd ~
cp -r /root/.cache/veRL-multiturn-rollout .
cd veRL-multiturn-rollout
git checkout async-tp
```

### 安装 dev 分支的 veRL

```bash
cd ~
git clone https://github.com/volcengine/verl.git verl-sglang
cd verl-sglang
git checkout dev_sglang
git pull --no-ff
python3 -m uv pip install .
```

### 从 github 安装最新的 SGLang main branch

```bash
python3 -m uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git/@main#egg=sglang&subdirectory=python" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

按照上述流程，很有可能缺少 `flash-attn`，这里建议手动安装：

```bash
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

#### 安装 requirements.txt

```bash
cd ~/veRL-multiturn-rollout
python3 -m uv pip install -r ./requirements.txt
```

## 4 卡测试 SGLang

### 下载模型

```bash
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir /root/models/qwen2-7b-instruct --local-dir-use-symlinks False
```

### 使用前需要配置好 `WANDB_API_KEY`

可以参考[这个过程](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。

```bash
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# 定义这个时间戳函数
function now() {
    date '+%Y-%m-%d-%H-%M'
}
```

### 运行

```bash
# change the visible devices on your own
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python3 -m verl.trainer.main_ppo trainer.experiment_name=qwen7b_sft2_$(now)
```

