# Test log

1. 基于verl所给的dockers image构建container

   ```
   docker run \
       -it \
       --shm-size 32g \
       --gpus all \
       -v /models/shared/.cache:/root/.cache \
       --ipc=host \
       --network=host \
       --privileged \
       --name verl_sppo_{your-name} \
       ocss884/verl-sglang:ngc-th2.5.1-cu126-sglang0.4.4.post4 \
       /bin/bash
   ```

2. 安装 verl

   ```
   git clone https://github.com/yhyang201/verl.git && cd verl
   git checkout sppo
   pip3 install -e .[sglang]
   ```

3. wandb login

   ```
   wandb login
   ```

4. download dataset and model

   ```
   python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
   huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct 
   ```

5. run bash (tested on h20x4)

   ```
   cd recipe/sppo
   bash run_qwen2.5-7b_rm.sh
   ```
