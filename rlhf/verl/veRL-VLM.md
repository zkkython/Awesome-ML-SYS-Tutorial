# Qwen2.5VL GRPO using Sglang

## 环境配置

### 创建新的 docker

```bash
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v /.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```

以后每次从 docker 里面 exit 出来，再用这个指令可以重启：

```bash
docker start -i h100_{your_name}
```

### 基于源码安装 SGLang

配置 python 环境

```bash
mkdir -p /tmp
chmod 1777 /tmp
apt update
apt install -y python3.10 python3.10-venv
python3 -m ensurepip --upgrade
python3 -m venv ~/.python/verl-sglang
source ~/.python/verl-sglang/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade uv
```

安装 SGLang

```bash
cd ~
git clone https://github.com/sgl-project/sglang.git
cd sglang

python3 -m uv pip install --upgrade pip
python3 -m uv pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

安装 veRL

```bash
cd ~
git clone https://github.com/volcengine/verl.git
cd verl
python3 -m uv pip install .
python3 -m uv pip install -r ./requirements.txt
```

## 4 卡启动 Qwen2.5VL GRPO 训练脚本，并且使用 SGLang 作为 rollout 引擎

使用前需要配置好 `WANDB_API_KEY`

参考[这个过程](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。

```bash
cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# 拉取并预处理 geo3k 数据集
python examples/data_preprocess/geo3k.py --local_dir ~/data/geo3k

# 启动 GRPO 训练脚本， 记得去掉 examples/grpo_trainer/run_qwen2_5_vl-7b.sh 结尾的 $@
bash examples/grpo_trainer/run_qwen2_5_vl-7b.sh sglang
```
