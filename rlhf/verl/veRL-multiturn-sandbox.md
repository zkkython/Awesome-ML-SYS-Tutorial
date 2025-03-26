# veRL-multiturn Sandbox

当前版本的 `veRL` 在 `swedev` 任务下与环境交互的文档，尽可能还原了源码里 sandbox 的调用方式，用于快速搭建一个 sandbox for veRL

## 建立 session

- URL: `http://60.165.239.98:5000/start_instance`
- 方法: `POST`
- 请求参数
  - <string> `instance_hash` veRL 端根据训练数据中的 `instance_id` 传递过来，用于标识要处理的项目/任务/PR。
- 返回参数
  - <string> `sid` session id，用于标识当前会话。后续的 `/process_action`、`/postprocess`、`/compute_reward` 都会带上这个 `sid`，告诉 sandbox 是哪一个任务。

- 功能
  1. Sandbox 可以在此时建立一个 session 上下文并存到内存或数据库里：`context_map[sid] = {...}`；
  2. 后续的接口就可以通过 `sid` 找回任务状态。

**example**

- request

```json
{
    "instance_hash": "3864552457764042195"
}
```

- response

```json
{
  "sid": "5671949450826943757"
}
```

**对应源码**

```python
# verl.utils.swedev_utils.py
async def initialize_runtime(instance_id):
    url = get_api(type="start")
    payload = {"instance_hash": instance_id}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=600) as response:
                result = await response.json()
                return result
    except Exception as e:
        print(f"Initializing - API call failed: {e}")
        return None
```

```python
# 调用点：verl.workers.agentic.async_rollout.py AsyncRollout.generate_sequence.swedev_start()
async def swedev_start(index):
            try:
                result = await initialize_runtime(prompts.batch['instance_id'][index // n].item())
                print(result)
                return {
                    "prompt_ids": _pre_process_inputs(tokenizer.pad_token_id, input_ids[index]),
                    "sid": result["sid"],
                    "sids": int(result["sid"]), # will be treated as a obs metric, thus, will be gathered into batch, and later used in reward acquisition
                }
            except Exception as e:
                # TODO: return true for handle api instead of raising an error
                print(f"Error processing instance: {e}")
                # in original logic, mismatched sids count and instance_ids count will cause error eventually, better raise now
                raise
```

## 处理 action

- URL: `http://60.165.239.98:5000/process_action`
- 方法: `POST`
- 请求参数
  - <string> `sid` 
  - <string> `content` prompt
- 返回参数
  - <string> `content` response

- 功能
  1. sandbox 把 `content` 视为一次action提交，如将用户/模型产生的代码写进临时文件并运行测试。

**example**

- request

```json
{
  "sid": "5671949450826943757",
  "content": "def solve():\n    pass"
}
```

- response

```json
{
  "content": "No errors found in test."
}
```

**对应源码**

```python
# verl.utils.swedev_utils.py
async def call_observation_api(sid, text: str):
    if isinstance(sid, torch.Tensor):
        sid = sid.item()
    url = get_api(type="action")
    payload = {
        "sid": sid,
        "content": text,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                return await response.json()
    except Exception as e:
        print(f"Observation - API call failed: {e}")
        return None  
```

```python
# 调用点：verl.workers.agentic.tasks.py
async def swe_dev_obs(action_ids, sid, tokenizer, **kwargs):
    action = tokenizer.decode(action_ids, skip_special_tokens=False)
    if is_stop(action):
        print(f"Action stop: {action}")
        return {"done": True, "ids": [], "observation_times": 0}

    result = call_observation_api(sid, action)
    # TODO(haoran): handle here
    try:
        obs = result["content"]
    except:
        obs = "Error"
    return {"done": False, "ids": tokenizer.encode(obs), "observation_times": 1}
```

## 后处理

- URL: `http://60.165.239.98:5000/postprocess`
- 方法: `POST`
- 请求参数
  - <string> `sid` session id
- 返回参数
  - 自定义

1. sandbox 接收请求，处理文本内容（如翻译、解析等）。
2. sandbox 返回处理后的文本内容（batch）。

**example**

- request

```json
{
  "sid": "5671949450826943757"
}
```

- response

```json
{
}
```

- 功能
  1. 多轮对话结束后收尾，sandbox可在这里执行“清理资源”“停止容器”“合并最终日志”等。
  2. 返回 JSON 的内容不参与后续对话，但可记录到日志。

**对应源码** 

```python
# verl.utils.swedev_utils.py
async def call_postprocess_api(sid: str):
    url = get_api(type="postprocess")
    if isinstance(sid, torch.Tensor):
        sid = sid.item()
    payload = {"sid": sid}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=600) as response:
                return await response.json()
    except Exception as e:
        print(f"Postprocess - API call failed: {e}")
        return None
```

```python
# 调用点：verl.workers.agentic.tasks.py
async def swe_dev_end(sid, _done):
    await asyncio.to_thread(call_postprocess_api, sid)
```

## 计算 reward

- URL: `http://60.165.239.98:5000/compute_reward`
- 方法: `POST`
- 请求参数
  - <string> `sid` session id
- 返回参数
  - <int> `reward`
  - <int> `f2p_count` (Optional)
  - <int> `f2p_total` (Optional)

- 功能
  1. 根据前面 `/process_action` 的累计结果，执行自动化测试或其他判断，然后得出 reward 分数。

**example**

- request

```json
{
  "sid": "5671949450826943757"
}
```

- response

```json
{
  "reward": 0.9,
  "f2p_count": 9,
  "f2p_total": 10
}
```

**对应源码**

```python
# verl.workers.reward_manager.swedev.py SWEDevRewardManager.fetch_reward()
async def fetch_reward(self, sid: torch.Tensor, session: aiohttp.ClientSession) -> float:
        """Fetch reward from API for a single instance"""
        try:
            payload = {"sid": sid.item()}
            async with session.post(get_api(type="reward"), json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    result = await response.json()
                    return float(calc_reward(result))
                else:
                    print(f"fetch_reward - API request failed with text {response.text} for sid: {sid}")
                    return 0.0
        except Exception as e:
            print(f"fetch_reward - Error fetching reward for sid {sid}: {e}")
            return 0.0
```

```python
# verl.utils.swedev_utils.py 
def calc_reward(reward_json):
    # patch_is_None
    # patch_exists
    # patch_succesfully_applied
    # resolved
    # f2p_count
    # f2p_total
    # p2p_count
    # p2p_total
    
    if 'reward' in reward_json:
        return reward_json['reward']
    else:
        try:
            return reward_json['f2p_count'] / reward_json['f2p_total']
        except:
            return 0.0
```

```python
# verl.utils.swedev_utils.py 
def get_api(type):
    base_url = random.sample([
        "http://60.165.239.98:5000",
        "http://60.165.239.99:5000"
    ], 1)[0]
    # TODO: only support shouyun1 now
    base_url = "http://60.165.239.98:5000"
    assert type in ["reward", "action", "start", "postprocess"]
    if type == "reward":
        return f"{base_url}/compute_reward"
    elif type == "action":
        return f"{base_url}/process_action"
    elif type == "start":
        return f"{base_url}/start_instance"
    elif type == "postprocess":
        return f"{base_url}/postprocess"
```

## 流程图

```
train.py
  └── main() / run_ppo()
        └── main_task()
              ├── 构造 reward_fn = SWEDevRewardManager(...)
              ├── 初始化 RayPPOTrainer(..., reward_fn=reward_fn)
              │     └── trainer.fit()
              │           └── rollout_wg.generate_sequences()
              │                 └── AsyncRollout.generate_sequences()
              │                       └── swedev_start()（/start_instance）
              │                       └── swe_dev_obs()（/process_action）
              │                       └── swe_dev_end()（/postprocess）
              │
              └── reward_fn(data)：
                    └── SWEDevRewardManager.__call__()
                          └── asyncio.run(fetch_reward())
                                └── POST /compute_reward
```

## bkp (写多了，这里用不上)

### d-r task 中的 observation
- URL: `http://172.16.65.43:8888/observation_kilt/`
- 方法: `POST`
- 请求参数
  - <string> `content` prompt
  - <bool> `translate`: 是否进行翻译（bool）
- 返回参数
  - <list[string]> `content` 模型生成的文本内容

- 功能
  1. sandbox接收请求，处理文本内容（如翻译、解析等）。
  2. sandbox返回处理后的文本内容（batch）。

#### example

- request

```json
{
  "content": "Tell me about the Eiffel Tower",
  "translate": true
}
```

- response

```json
[
  {
    "content": "The Eiffel Tower is in Paris."
  },
  {
    "content": "It was completed in 1889."
  }
]
```

#### 对应源码

```python
# verl/workers/agentic/async_rollout.py AsyncRollout.generatr_sequences.dr_obs()
async def dr_obs(action_ids, sid, tokenizer, **_):
            # find <|observation|> token part
            sid = sid % len(input_ids)
            dr_storage_sid2seq[sid].extend(action_ids)
            stop_id = action_ids[-1]
            stop_token = tokenizer.decode([stop_id])
            print(f"stop token: [{stop_id}] - [{stop_token}]")

            # only finish with <|observation|> token can be multi-turn
            if not stop_token.strip() == '<|observation|>':
                return {"done": True, "ids": [], "observations_times": 0, "failed_times": 0}
            action = tokenizer.decode(action_ids, skip_special_tokens=False)
            text = action.split("<|observation|>")[0].strip()

            # call api part
            url = "http://172.16.65.43:8888/observation_kilt/"
            payload = {"content": text, "translate": True}
            failed = 0
            for i in range(5):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(url, json=payload)
                        ret = response.json()
                        break
                except Exception as e:
                    print(f"API call failed: {e}")
                    await asyncio.sleep(1 + i)
            else:
                ret = [{"content": "API call failed, you may try again."}]
                failed = 1

            # combine part
            obv_combined = ['\n' + obv['content'].strip() for obv in ret]
            obs_text = f"{'<|observation|>'.join(obv_combined)}<|assistant|>\n"
            ret_ids = tokenizer.encode(obs_text)
            dr_storage_sid2seq[sid].extend(ret_ids)

            if torch.distributed.get_rank() == 0:
                print(f"nodedup {torch.distributed.get_rank()=} dr_obs: {obs_text=}")

            return {"done": False, "ids": ret_ids, "observations_times": 1, "failed_times": failed}
```