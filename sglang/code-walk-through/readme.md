# SGLang Code Walk Through

**[Warning: This is a work in progress, and the content is not yet complete. We are still polishing the content and will update it soon.]**

This doc serve as a developer-level guidance and provide a brief code walkthrough of SGLang's backend, tracing the path of how requests are processed, as shown in the following figure.

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./sglang-architecture.svg" alt="SGLang Architecture" style="width: 100%; height: auto;">
</div>

Specifically, requests flow through the following process to get responses:

1. The user launches the Server, initializing the FastAPI app, TokenizerManager, DetokenizerManager, and Scheduler, each running with its infinite event loop.

2. The user sends `/v1/chat/completions` requests to the FastAPI app, which routes them to TokenizerManager via the `v1_chat_completions` endpoint.

3. The `v1_chat_completions` function converts the incoming requests into a `ChatCompletionRequest`, further transforming them into a `GenerateReqInput` before invoking TokenizerManager's `generate_request` method.

4. TokenizerManager tokenizes the requests and forwards them to the Scheduler as Python objects (`pyobj`) while calling `_wait_one_response`.

5. The Scheduler operates its infinite `event_loop_normal` to handle the requests:
    - The Scheduler receives the requests via `recv_requests`, processes them through `process_input_requests`, handles the generation logic with `handle_generate_request`, and adds them to the `waiting_queue`.
    - From the `waiting_queue`, the Scheduler uses `get_next_batch_to_run` to create a `ScheduleBatch` for the upcoming requests.
    - The Scheduler executes the `run_batch` function, converting the `ScheduleBatch` into a `ModelWorkerBatch`.
    - The Scheduler calls `TpModelWorker`'s `forward_batch_generation`, awaiting the `logits_output` and `next_token_ids`.
    - TpModelWorker initializes a `ForwardBatch`, forwards it to ModelRunner, and waits for the `logits_output`.
    - The ModelRunner processes the `ForwardBatch` by classifying it and invoking `forward_extend` to execute the model's forward pass.
    - The model, accelerated by `AttentionBackend`, generates logits, which are returned to ModelRunner and subsequently to TpModelWorker.
    - TpModelWorker receives the `logits_output` from ModelRunner, invokes its `sample` method to generate `next_token_ids`, and sends them back to the Scheduler.
    - The Scheduler processes the batch results using `process_batch_result` and checks the completion status via `check_finished`.
    - If requests are completed, the `process_batch_result_decode` function adds them to the cache using `tree_cache.cache_finished_req(req)` and sends their outputs to `stream_output`.
    - In `stream_output`, the outputs are processed, wrapped into `BatchTokenIDOut`, and sent to the DetokenizerManager.

6. The DetokenizerManager, running its own event loop, receives `BatchTokenIDOut`, processes it, and sends `BatchStrOut` back to TokenizerManager.

7. The TokenizerManager, within its event loop, receives the results, processes them via `handle_loop`, updates the internal state, and yields the response to the server.

8. The FastAPI app packages the response and sends it back to the user.


Note that all the discussions are based on release [v0.4.0](https://github.com/sgl-project/sglang/tree/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751). We sincerely appreciate [Chenyang Zhao](https://zhaochenyang20.github.io/Chayenne/), [Wenxuan Tan](https://github.com/Edenzzzz),  [Simon Veitner](https://simveit.github.io/) and [Shuai Shi](https://shuaills.github.io/) for their contribution to this document.

## Launch Server


SGLang features an SRT (SGLang Runtime) Server for [serving online HTTP requests](https://sgl-project.github.io/start/send_request.html) and an Engine for [offline model execution](https://sgl-project.github.io/backend/offline_engine_api.html). Key functions, [`launch_server`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L507) and [`launch_engine`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L418), are in [server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server.py). The `launch_engine` function initializes core SRT Server components.

1. Set up configs (logger, server args, CUDA/NCCL env, inter-process ports) and download the model and tokenizer.
2. Run Scheduler processes: Each Scheduler runs TpModelWorker for prefill and decode, manage radix cache, and handles TokenizerManager requests in infinite event loop. If `dp_size > 1`, `run_data_parallel_controller_process`; otherwise, initialize a Scheduler for each `tp_rank`.
3. Run TokenizerManager and DetokenizerManager as subprocesses: the former tokenizes data for the Scheduler, and the latter detokenizes Scheduler outputs for the server frontend. For multi-node inference (e.g., Llama 3.1 405B), TokenizerManager and DetokenizerManager only run on the first node.
4. Apply chat templates via (if specified) and wait for Scheduler processes to signal readiness while collecting their configuration.

## Forward Requests From Server

The Server employs a FastAPI app to define API endpoints, forwarding [`/v1/chat/completions`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L354) requests to TokenizerManager via [v1_chat_completions](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/openai_api/adapter.py#L1101).


1. Parse JSON from `raw_request` into a `ChatCompletionRequest`, then convert it to `GenerateReqInput` and configure `sampling_params` using `v1_chat_generate_request`.
2. Invoke TokenizerManager `generate_request` and handle streaming or non-streaming responses based on the `stream` parameter.
3. For streaming, process `generate_request` output incrementally with `generate_stream_resp`; for non-streaming, await the result and convert it to a `ChatCompletionResponse` via `v1_chat_generate_response`.

## Generate Request In TokenizerManager

TokenizerManager runs in Server's main weprocess, handling request tokenization. It is initialized in `launch_server`, with details in [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py).

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tokenizer_manager.py#L88)

1. Set up ZeroMQ for inter-process communication, including sockets for DetokenizerManager and Scheduler.
2. Configure `server_args`, enable `metrics`, initialize `model_config`, `tokenizer`, and placeholders for multi-modal image processors.

### [generate_request](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tokenizer_manager.py#L173)

1. Create an event loop if not already initialized.
2. Pause processing if model weights are updating via `update_weights_from_disk` or `update_weights_from_distributed`.
3. Validate request compatibility with the model's `is_generation` setting.
4. Normalize requests using `normalize_batch_and_arguments` to manage batching, parallel sampling, and default parameters.
5. Process single requests with `_tokenize_one_request`, send to the scheduler, and yield responses from `_wait_one_response`.
6. Process batch requests with `_handle_batch_request`, tokenize inputs, manage parallel sampling, interact with the scheduler, and yield responses in both streaming and non-streaming modes.

## Scheduler Receive Requests and Process Batches

The Scheduler runs as Server’s subprocess, initialized via `run_scheduler_process` and executing its infinite event loop with `event_loop_normal` or `event_loop_overlap`. Details can be found in [scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py).

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L97)

1. Set up ZeroMQ for communication with TokenizerManager and response handling.
2. Configure `server_args`, `port_args`, `model_config`, `sessions`, and initialize TpModelWorker or TpModelWorkerClient based on overlap scheduling.
3. Initialize tokenizer and processor for multimodal or standard models based on server arguments.
4. Manage caching using ChunkCache or RadixCache and configure SchedulePolicy.
5. Set up chunk prefill parameters and GrammarBackend for request processing.

### [Event Loop](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L376)

The Scheduler continuously executes its event loop, alternating between `get_next_batch_to_run`, `run_batch` and `process_batch_result`.

### [get_next_batch_to_run](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L768)

1. Merge `last_batch` with `running_batch` if applicable and prioritize prefill batches with `get_new_batch_prefill` for immediate execution.
2. If no prefill batch, update `running_batch` for decode batch by filtering requests, managing memory, and adjusting decoding parameters.

### [run_batch](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L956)

1. For generation models, use TpModelWorker’s `forward_batch_generation` for token prediction or `forward_batch_idle` for idle tasks, returning results to `event_loop_normal`.
2. For embedding or reward models, assert token extension, execute `forward_batch_embedding`, and return embeddings.

### [Processing and Finalizing Results](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L987)

 In serving engines, LLM inference is usually broken into prefill and decode stages for their different compute charactor. You can check [this post](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi) from HuggingFace regarding the concept of Prefill and Decode. In SGLang, [extend mode](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py) is used instead of prefill mode most of the time. Prefill initializes KV-Cache for new requests, typically using Paged KV-Cache. Extend updates existing KV-Cache incrementally, often leveraging Ragged Tensors for efficiency, making it ideal for long sequences or multi-turn tasks.

After `run_batch`, Scheduler processes batch results in `event_loop_normal`:

1. Decode mode processes outputs, updates request states, handles token and probability data, manages memory, and logs statistics.
2. Extend mode handles prefill results, processes input tokens, and prepares for further decoding or embedding.
3. Finished requests are cached via `cache_finished_req`, streamed to DetokenizerManager. Unfinished requests are updated and looped back into `get_next_batch_to_run` for further processing until completion.

## TpModelWorker Manage Forward and Token Sampling

TpModelWorker manages ModelRunner’s forward and sampling for batches of requests scheduled by Scheduler. It's implementation can be found in [tp_worker.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py).

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tp_worker.py#L40)

1. Initializes tokenizer, model configuration and ModelRunner.
2. Configures device settings and memory pool limits.

### [forward_batch_generation](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tp_worker.py#L148)

1. Create a `ForwardBatch` with `model_worker_batch` and ModelRunner.
2. Compute logits and sample next tokens using ModelRunner’s `forward` and `sample`.
3. Return Results to Scheduler for process_batch_result.

### [forward_batch_embedding](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tp_worker.py#L160)

Embedding requests bypass token sampling and directly retrieve embeddings via ModelRunner’s `forward`.

## ModelRunner Manage AttnBackend and Model Execution

ModelRunner initialize the AttnBackend and managing the loaded model to perform various types of forward passes.

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/model_executor/model_runner.py#L66)

Initializes distributed environment, loads the model, applies tensor parallelism, and sets up memory pool and attention backends.

### [forward](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/model_executor/model_runner.py#L675)

The `forward` function determines the appropriate forward mode for processing batches based on the `forward_mode`:

1. `forward_decode`: Initializes metadata and processes the model's forward pass using input IDs and positions.
2. `forward_extend`: Initializes metadata and executes the forward pass for generation or embedding tasks.
3. `forward_idle`: Manages the forward pass when the forward mode is idle.

## Model Concert HuggingFace Models and Perform Forward

All [supported models](https://sgl-project.github.io/references/supported_models.html) can be found in [python/sglang/srt/models](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models). We take [Qwen2ForCausalLM](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2.py) and chat completion requests for example.

[`Qwen2ForCausalLM`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/models/qwen2.py#L269) is structured as follows:

* `model`: weights used for the forward pass.
* `embed_tokens`: maps `input_ids` into `embeddings`.
* `lm_head`: projects the hidden states back to the vocabulary space.
* `logits_processor`: manipulates `logits` to perform tasks such as sampling and normalization.
* `pooler`: pooling mechanism for extracting embeddings or computing rewards.

### [forward](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/models/qwen2.py#L289)

The forward function in Qwen2ForCausalLM processes input sequences to produce logits for next-token prediction of chat completion requests or embeddings for reward/embedding requests:

1. Converts `input_ids` to embeddings using `embed_tokens`. Sequentially processes embeddings through Qwen2DecoderLayer layers.
2. Returns embeddings via `pooler` if `get_embedding` is True; otherwise, computes logits using `logits_processor`.

The most import acceleration comes from the interaction between `forward_batch` and AttentionBackend. 

## AttentionBackend Accelerate Model Forward

SGLang supports several [AttentionBackends](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention) which accelerate model forward and KV cache reuse. We take [FlashInferBackend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py) for example.

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L48)

1. Configures wrappers for sliding window and cross-attention scenarios.
2. Allocates necessary buffers for workspace and key-value indices.
3. Prepares forward metadata for efficient attention computation.
4. Integrates CUDA graph support for optimized execution paths.

### [init_forward_metadata](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L130)

1. Decode Mode: Updates indices for decoding using `indices_updater_decode` and sets `forward_metadata` to use `decode_wrappers`.
2. Extend Mode: Determines if ragged forward is needed based on token count and wrapper count, then updates indices with `indices_updater_prefill`.
3. Metadata Assignment: Sets `forward_metadata` with flags for ragged use and prefix extension.

### [forward_extend](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L223) and [forward_decode](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L277)

1. Wrapper and Attention Type: Selects the appropriate wrapper and decides between ragged or paged attention for `forward_extend`, or picks the decode wrapper for `forward_decode`.
2. Attention Calculation: Computes attention, optionally managing key-value caching, and returns the reshaped output.

## DetokenizerManager Detokenize and Send to TokenizerManager

We can find the detokenizer manager in [detokenizer_manager.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/detokenizer_manager.py).

### [Initialization](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/detokenizer_manager.py#L53)

Sets up communication sockets and the tokenizer. Manages decoding states with `LimitedCapacityDict`.

### [event_loop](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/detokenizer_manager.py#L93) and [trim_eos](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/detokenizer_manager.py#L78)

1. Receives data from the Scheduler, forwarding `BatchEmbeddingOut` directly or processing `BatchTokenIDOut` for detokenization.
2. Splits token IDs into `read_ids` and `surr_ids`. Converts token IDs to text using `batch_decode`. Updates `DecodeStatus` with new offsets and decoded text.
3. Trims outputs at stop sequences, combines decoded text with metadata into `BatchStrOut`, and sends it to TokenizerManager.

## [FastAPI Wraps the Output](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L287)

1. DetokenizerManager sends `BatchStrOut` to TokenizerManager via ZeroMQ.
2. TokenizerManager updates request states and prepares decoded text for FastAPI.
3. For streaming, use an async generator and `StreamingResponse` in FastAPI.
4. For non-streaming, collect and send the complete response using `ORJSONResponse`.