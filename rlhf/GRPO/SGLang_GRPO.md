# GRPO

This is the dev-log from SGLang team to support GRPO in [TRL](https://huggingface.co/docs/trl/index). Group Relative Policy Optimization (GRPO),  a variant of Proximal Policy Optimization (PPO), enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO. This doc will first introduce how GRPO works and then show how we add SGLang as an alternative inference backend for TRL, specifically the [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer), which also functions in the [Open-R1](https://github.com/huggingface/open-r1) project.

## How GRPO works

Compared with PPO, GRPO doesn't have the **value/critic model** to estimate total value. The algorithm computes the normalized reward for each output to derive advantages and updates the **reward model** to enhance training performance. In the context of GRPO, the term "update" refers specifically to the parameter updates of the policy model using gradients computed from the loss function. This means each training iteration adjusts the model's weights to maximize the advantage while maintaining proximity to a reference policy.

GRPO is composed of four steps:

- Generating completions
- Computing the advantage
- Estimating the KL divergence
- Computing the loss

<div align="center">
<img src="GRPO_Images/grpo-main.png" width="100%">
</div>


### Generating completions

At each training step, we sample a batch of prompts and generate a set of $G$ completions for each prompt (denoted as $o_{i = 1, 2, ..., G}$).

### Computing the advantage

For each of the $G$ sequences, we compute the reward using a **reward model**. To align with the comparative nature of reward models—typically trained on datasets of comparisons between outputs for the same question—the advantage is calculated to reflect these relative comparisons. It is normalized as follows:

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

This approach gives the method its name: **Group Relative Policy Optimization (GRPO)**, since it uses the relative reward to compute the advantage.

Hugging Face uses the above equation to compute advantages. In [GRPO paper](https://arxiv.org/abs/2402.03300), the author named it Outcome Supervision RL with GRPO. The author also found another method named Process Supervision RL with GRPO.

**Supervision Reinforcement Learning (Supervision RL)** combines traditional reinforcement learning with explicit guidance from labeled or structured data during training. It leverages direct supervision to enhance learning efficiency and model performance.

**Process Supervision RL** specifically uses intermediate step-wise rewards rather than evaluating only final outcomes. It provides detailed, step-by-step feedback to the model, thus facilitating fine-grained optimization of policy decisions at every generation step.

We can also leverage the information in each step. Formally, given the question *q* and *G* sampled outputs {$o_1$, $o_2$, … , $o_G$}, a process reward model is R = {{$r_1^{index(1)}$, …, $r_1^{index(K_1)}$}, … , {$r_G^{index(1)}$, …, $r_G^{index(K_G)}$}}, where  $index(K_j)$ is the end token index of $K_j$-th completion.

Normalize each reward:

$$
\tilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - mean(R)}{std(R)}
$$

The advantages of each output in each step:

$$
\hat{A}_{i,t} = \sum_{index(j)\geq t}\tilde r_i^{index(j)}
$$

### Estimating the KL divergence

KL divergence is estimated using the approximator, which is defined as follows:

$$
\mathrm{D}_{\mathrm{KL}}[\pi_{\theta}||\pi_{\mathrm{ref}}] = \frac{\pi_{\mathrm{ref}}(O_{i,t} \mid q, O_{i,<t})}{\pi_{\theta}(O_{i,t} \mid q, O_{i,<t})} - \log \frac{\pi_{\mathrm{ref}}(O_{i,t} \mid q, O_{i,<t})}{\pi_{\theta}(O_{i,t} \mid q, O_{i,<t})} - 1
$$

## Computing the loss

The objective is to **maximize the advantage** while ensuring that the model **remains close to the reference policy**. Consequently, the loss is defined as follows:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \frac{\pi_{\theta}(O_{i,t} \mid q, O_{i,<t})}{\pi_{\text{ref}}(O_{i,t} \mid q, O_{i,<t})} \right]_{\text{nograd}} \hat{A}_{i,t} - \beta D_{\text{KL}}[\pi_{\theta} \parallel \pi_{\text{ref}}]
$$

- the first term represents the scaled advantage
- the second term penalizes deviations from the reference policy through KL divergence.

In the original paper, this formulation is generalized to account for multiple updates after each generation by leveraging the **clipped surrogate objective**:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min \left( \frac{\pi_{\theta}(O_{i,t} \mid q, O_{i,<t})}{\pi_{\text{old}}(O_{i,t} \mid q, O_{i,<t})} \hat{A}_{i,t}, \mathrm{clip} \left( \frac{\pi_{\theta}(O_{i,t} \mid q, O_{i,<t})}{\pi_{\text{old}}(O_{i,t} \mid q, O_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right) \right] - \beta D_{\text{KL}}[\pi_{\theta} \parallel \pi_{\text{ref}}]
$$

where $clip(⋅,1−\epsilon,1+\epsilon)$ ensures that updates stay close to the reference policy by keeping the policy ratio between $1−\epsilon$ and $1+\epsilon$. However, since TRL follows the original paper in performing only one update per generation, we can simplify the loss to the first form.

## Customized GRPO

To support GRPO in TRL, we need to customize the GRPO Trainer. First, we can take a look at how inference engines are used in the [GRPO Trainer](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py).


### Conditional Setup and Import

The code conditionally imports vLLM modules and uses a configuration flag (`args.use_vllm`) to enable/disable vLLM's generation engine. This is controlled through the `self.use_vllm` flag in the trainer's constructor.

### Initialization and Dedicated GPU Selection

- **Dedicated Device Assignment**

When `use_vllm` is True, the trainer (but only on the main process) determines a dedicated GPU for generation. For example:

```python
if self.accelerator.is_main_process:
    vllm_device = self.args.vllm_device
    if vllm_device == "auto":
        if torch.cuda.device_count() == 1:
            vllm_device = "cuda:0"
        else:
            vllm_device = f"cuda:{self.accelerator.num_processes}"
```

This logic assigns one GPU (or a GPU outside of those used for training) exclusively for the vLLM generation task, thereby detaching generation work from the training GPUs.

- **Patching for Compatibility**

Since vLLM isn't inherently designed to work with the distributed setup from `accelerate`, the code applies two patches:

```python
world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
with world_size_patch, profiling_patch:
    self.llm = LLM(
        model=model.name_or_path,
        device=vllm_device,
        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
        dtype=self.args.vllm_dtype,
        enable_prefix_caching=True,
        max_model_len=self.args.vllm_max_model_len,
    )
self.sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=self.max_completion_length,
)
```

These patches make sure that:

1. The vLLM engine sees a **"world size"** of 1 (since it's running on a dedicated device).
2. Certain profiling checks that aren't applicable in this setting are bypassed.

- **Synchronization**

After setting up vLLM, the main process calls `self.accelerator.wait_for_everyone()` to ensure all processes are synchronized.When using vLLM, the main process is responsible for loading the model weights. This can cause process desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, this function ensures **all processes are synchronized** after the dedicated generation device is set up.

### Using vLLM for Generation During Training

- **Moving Weights to vLLM**

Before generating completions, the trainer calls **`_move_model_to_vllm(self)`**. This method extracts the model's state (merging adapters if needed) and loads the weights into the vLLM engine's GPU worker:

```python
if self.accelerator.is_main_process:
    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
```

This transfer is key because vLLM is responsible for **generating text**, and its internal model must reflect **the latest training weights**.

The method `llm_model.load_weights(state_dict.items())` is initially used to load model weights into vLLM during setup. Subsequent updates during training are also done by updating vLLM weights from disk.

- **Generation Workflow**

In the `_prepare_inputs` method, when using vLLM:

1. The main process gathers **all prompt texts** across GPUs.
2. It then **generates completions** with `outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)`.
3. The completions are **extracted**, and the results are **broadcast** to all processes so that every GPU gets its corresponding slice of generated text.

- **Detach Generation from Training**

By detaching the generation task to a dedicated GPU placement group, the training GPUs remain fully occupied with gradient computations.

## Support SGLang in GRPO Trainer

Here, we support SGLang as an inference backend alongside other inference engines in the `GRPOTrainer`. Unlike other inference engines—which run as an in-process engine using dedicated classes (e.g. `LLM`, `SamplingParams`) and requires patching of distributed methods—SGLang is deployed as a standalone server with HTTP endpoints (compatible with OpenAI's APIs). As a result, our integration leverages HTTP requests to manage weight updates and generate completions.

To substitute other inference engines with **SGLang**, we must account for the differences in API and internal architecture. The following steps outline the necessary modifications:

- **Checkpoint-based Updates**

    - Update the `GRPOConfig` by adding a `checkpoint_path` parameter.

    - Write model checkpoints at regular intervals.

    - Use the existing `/update_weights_from_disk` endpoint provided by the SGLang server.

This approach avoids modifying SGLang’s internal initialization routines and leverages its existing, stable checkpoint-loading capabilities.

### Import and Availability Check

Instead of importing in-process engine classes for SGLang, we introduce a configuration flag (`use_sglang`) in our arguments (e.g., in `GRPOConfig`). This flag signals that generation should be offloaded to a SGLang server. Since SGLang is accessed via HTTP calls, there's no need to import objects like `SGLangEngine`. Instead, we ensure that the SGLang server is reachable (or launch it within our code) and then use helper utilities (from `sglang.utils`) to manage the server lifecycle.

### Initialization on a Dedicated GPU

- **Server Launch and Device Assignment**

Rather than creating an in-process generation engine, we launch the SGLang server as an external process on a dedicated GPU. For example, in the trainer's `__init__`, we added:

```python
if self.args.use_sglang:
    # Assign a dedicated GPU for the SGLang server (e.g., "cuda:1")
    sglang_gpu_id = self.args.sglang_device.split(':')[-1]
    sglang_command = (
        f"CUDA_VISIBLE_DEVICES={sglang_gpu_id} python -m sglang.launch_server "
        f"--model-path {model_id} --host 0.0.0.0"
    )
    from sglang.utils import launch_server_cmd, wait_for_server
    self.server_process, port = launch_server_cmd(sglang_command)
    wait_for_server(f"http://localhost:{port}")
    self.sglang_server_url = f"http://localhost:{port}"
```

This command dedicates one GPU exclusively for the SGLang server, which will handle all generation requests.

- **Process Synchronization**

After launching the server, we call:

```python
self.accelerator.wait_for_everyone()
```

to ensure that all distributed processes are synchronized before proceeding.

### Generation and Weight Synchronization

- **Weight Synchronization**

In vLLM, update weights are done in-process via a helper like `_move_model_to_vllm()`. For SGLang, weight updates occur externally. We implement a helper function `_update_sglang_weights()` that calls SGLang's `/update_weights_from_disk` API to refresh the server's model state:

We revised `_update_sglang_weights` to robustly update model weights on the SGLang server by calling its /update_weights_from_disk API. This function now:
- Checks if the checkpoint exists.
- Sends an HTTP POST with a timeout.
- Checks for a success flag in the response.
- Optionally flushes the cache after the update.

- **About SGLang update weights**

【这里得全部改了】

**We meet some issue when trying to revise this function. To fix this, we must choose one of two paths:**

1. **Initialize the weight update group on the SGLang server** so that it supports distributed updates (and then continue using `/update_weights_from_distributed`). This means modifying the server's initialization (in `ModelRunner`) to call its `init_weights_update_group` function.

2. **Add a checkpointing mechanism in the training loop and use the disk-based update endpoint** `/update_weights_from_disk` (which doesn't require a weight update group). For this, update GRPOConfig to include a checkpoint_path and ensure that a checkpoint is written before calling the update.


In our current workflow we load the model directly from Hugging Face – which means we never had a "checkpoint". To use the SGLang `/update_weights_from_disk` endpoint (our "second choice"), we need to supply a checkpoint file. One straightforward solution is to add a new field (say, checkpoint_path) to our `GRPOConfig` and then, before training starts, save the current model weights to that location. Then, when `_update_sglang_weights()` is called, it will have a valid file from which the SGLang server can reload the weights.

```python
def _update_sglang_weights(self):
    """
    Update the model weights on the SGLang server via its API.
    This function assumes that the training loop writes the latest checkpoint to self.args.checkpoint_path.
    It performs additional checks and error handling to ensure the server successfully updates its weights.
    """
    import os
    import requests

    checkpoint = self.args.checkpoint_path
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint path {checkpoint} does not exist.")

    payload = {"model_path": checkpoint}
    try:
        response = requests.post(
            f"{self.sglang_server_url}/update_weights_from_disk",
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Weight update request failed: {e}")

    res_json = response.json()
    if not res_json.get("success", False):
        raise RuntimeError(f"Failed to update weights on SGLang server: {res_json.get('message', 'No message provided')}")

    # Optionally flush the cache after updating weights
    try:
        flush_response = requests.post(f"{self.sglang_server_url}/flush_cache", timeout=30)
        if not flush_response.json().get("success", True):
            print(f"Warning: Cache flush failed: {flush_response.json().get('message', 'No message provided')}")
    except requests.RequestException as e:
        print(f"Warning: Cache flush request failed: {e}")

    print(f"SGLang weights updated successfully: {res_json.get('message')}")
```

This function is called whenever the training step advances (e.g., if global_step changes).

- **Generation Call**

In the `_prepare_inputs()` method, we replace the in-process generation call with an HTTP request to SGLang's `/generate` endpoint:


```python
if self.use_sglang:
    # Update weights if the training step has advanced.
    if self.state.global_step != self._last_loaded_step:
        self._update_sglang_weights()
        self._last_loaded_step = self.state.global_step

    # Gather all prompt texts from all processes.
    all_prompts_text = gather_object(prompts_text)
    if self.accelerator.is_main_process:
        import requests
        payload = {
            "text": all_prompts_text,
            "sampling_params": self.sglang_sampling_params,
        }
        response = requests.post(f"{self.sglang_server_url}/generate", json=payload)
        generated_texts = response.json().get("text", [])
        completion_ids = [self.processing_class.encode(text) for text in generated_texts]
    else:
        completion_ids = [None] * len(all_prompts_text)

    # Broadcast and slice the generated completions.
    completion_ids = broadcast_object_list(completion_ids, from_process=0)
    process_slice = slice(
        self.accelerator.process_index * len(prompts),
        (self.accelerator.process_index + 1) * len(prompts),
    )
    completion_ids = completion_ids[process_slice]
    completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
    completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
```

- **Broadcasting Results**

We then broadcast the generated completions from the main process across all processes using `broadcast_object_list()` and slice the results according to each process's index.

### **API and Integration Adjustments**

- **Parameter Differences**

Since SGLang's API mimics OpenAI's endpoints, we pass sampling parameters as a **JSON payload** (e.g., `temperature`, `max_new_tokens`) to the `/generate` endpoint.

- **Error Handling and Device Checks**

We update error messages and checks for SGLang without needing to patch distributed functions—since SGLang runs as an external service, it operates independently of the training environment.

### **Overall Code Changes Recap**

- **Backend Flags**

Added `use_sglang` (and retained `use_vllm`) in configuration to let users choose the inference backend.

- **Server Initialization**

In the `__init__` method, if `use_sglang` is True, launch the SGLang server on a dedicated GPU and set `self.sglang_server_url`.

- **Weight Updates**

Implemented a robust **`_update_sglang_weights()`** function that ensures the SGLang server updates its model weights from the latest checkpoint, with error handling and cache flushing.

- **Generation Branch**

Modified **`_prepare_inputs()`** to branch based on the selected backend:

- **SGLang Branch**

Uses HTTP calls to SGLang's `/generate` endpoint, then converts returned texts to token IDs, broadcasts, and postprocesses.

### Testing and Next Steps

1. **Test the SGLang Server**

Ensure that the SGLang server launches correctly on the dedicated GPU and that its `/generate` and `/update_weights_from_disk` endpoints respond as expected.

2. **Integration Test**

Run the modified GRPOTrainer on a small dataset and verify that:

- Weights are updated on the SGLang server when the training step advances.
- Generation results are correctly broadcast and postprocessed.
- The overall training loop runs without errors.

## Summary

To support SGLang in TRL, we:

1. Introduce a configuration flag (e.g. use_sglang) and assign a dedicated GPU for the SGLang server.
2. Launch the SGLang server externally using a command (with appropriate GPU assignment) and wait for it to initialize.
3. Replace the in-process generation call in _prepare_inputs() with an HTTP POST to the SGLang /generate endpoint.
4. Implement a helper to update model weights on the SGLang server via its /update_weights_from_disk API.
5. Adjust parameter names, error handling, and synchronization logic to match SGLang's external server API.

# References

- SGLang Documentation – https://docs.sglang.ai/backend
- HuggingFace GRPO Trainer Documentation - https://huggingface.co/docs/trl/main/en/grpo_trainer
- Speeding Up Training - https://huggingface.co/docs/trl/main/en/speeding_up_training?vllm+examples=GRPO#vllm-for-fast-generation-in-online-methods
- GRPO Trainer in TRL - https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py