# Qwen2.5VL GRPO using Sglang
## 环境配置
### 创建新的docker
```
docker run -it --name h100_verl_{your_name} --gpus all \
    --shm-size 32g \
    -v /.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash
```
以后每次从 docker 里面 exit 出来，再用这个指令可以重启：
```
docker start -i h100_{your_name}
```
如果docker容器内部遇到问题，可以先备份:
```
# 这个会创建一个镜像，之后你可以基于它重新运行容器
docker commit h100_{your_name} h100_{your_name}_backup

# 停止并删除原容器
docker stop h100_{your_name}
docker rm h100_{your_name}

# 重新启动同名新容器
docker run -it --name h100_{your_name} --gpus all \
    --shm-size 32g \
    -v /.cache:/root/.cache \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    h100_{your_name}_backup \
    /bin/bash
```

### 基于源码安装Sglang（这样好做修改，并用git diff查看）
```
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```

### 拉取最新的veRL
```
git clone https://github.com/volcengine/verl.git
```

### 配置python环境
```
cd verl
python3 -m pip install -r ./requirements.txt
```

## 4卡启动Qwen2.5VL GRPO训练脚本，并且使用Sglang作为 rollout 引擎
使用前需要配置好`WANDB_API_KEY`
参考[这个过程](https://community.wandb.ai/t/where-can-i-find-the-api-token-for-my-project/7914)。
```
cd verl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# 拉取并预处理geo3k数据集
python3 examples/data_preprocess/geo3k.py --local_dir ~/data/geo3k

# 启动GRPO训练脚本， 记得去掉examples/grpo_trainer/run_qwen2_5_vl-7b.sh中的“$@”
bash examples/grpo_trainer/run_qwen2_5_vl-7b.sh sglang
```
