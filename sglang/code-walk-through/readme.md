# SGLang Code Walk Through

This doc serve as a developer-level guidance and provide a brief code walkthrough of SGLang's backend, tracing the path of how requests are processed. Generally, requests flow through the following process to get responses:


1. User `launch_server`, init FastAPI app, TokenizerManager, DetokenizerManager and Scheduler with their infinite event loops. 
2. User sends `/v1/chat/completions` requests to FastAPI app and it’s routed to TokenizerManager through `v1_chat_completions`. 
3. `v1_chat_completions` convert requests to `ChatCompletionRequest` then to `GenerateReqInput` and call TokenizerManager’s `generate_request`. 
4. TokenizerManager tokenize the requests and  send them to Scheduler as pyobj and `_wait_one_response`. 
5. Scheduler run its infinite `event_loop_normal` and process the requests:
    * Scheduler `recv_requests`, `process_input_requests`, `handle_generate_request` into `Req` and add it to `waiting_queue`. 
    * From `waiting_queue`, Scheduler `get_next_batch_to_run` and get a `ScheduleBatch` for the coming requests. 
    * Scheduler `run_batch` and convert `ScheduleBatch` into `ModelWorkerBatch`. 
    * Scheduler call TpModelWorker’s `forward_batch_generation` and wait for the `logits_output` and `next_token_ids`. 
    * TpModelWorker init `ForwardBatch`, forward it to  `ModelRunner` and wait for the `logits_output`. 
    *  ModelRunner classify the `ForwardBatch` and `forward_extend` to call Model’s forward. 
    * Model is accelerated by AttentionBackend and return logits to ModelRunner then to TpModelWorker. 
    * TpModelWorker get the `logits_output` from ModelRunner and call it’s `sample` to get `next_token_ids`, return them them to Scheduler. 
    * Scheduler `process_batch_result` and `check_finished` for the requests. 
    * If requests are finished, in `process_batch_result_decode`, requests will be added into cache by `tree_cache.cache_finished_req(req)` and send requests’ outputs to `stream_output`. 
    * In `stream_output`, the outputs are processed, warped into `BatchTokenIDOut` and `send_to_detokenizer`. 
6. DetokenizerManager runs its event_loop, receive `BatchTokenIDOut` and send `BatchStrOut` to TokenizerManager. 
7. TokenizerManager’s event loop received the results and `handle_loop`, update state and yield response to the Server. 
8. FastAPI app warp the response and send it back to the user. 

This guide aims to succinctly describe the procedure of launching a SGLang server in a top-down fashion, as illustrated in the following figures.

[TODO A Figure]

## Launch Server


SGLang features an SRT (SGLang Runtime) Server for [serving online HTTP requests](https://sgl-project.github.io/start/send_request.html) and an Engine for [offline model execution](https://sgl-project.github.io/backend/offline_engine_api.html). Key functions, `launch_server` and `launch_engine`, are in [server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server.py). The `launch_engine` function initializes core SRT Server components.


1. Set up configs (logger, server args, CUDA/NCCL env, inter-process ports) and download the model and tokenizer.
2. Run Scheduler processes: Each Scheduler runs TpModelWorker for prefill and decode, manage radix cache, and handles TokenizerManager requests in infinite event loop. If `dp_size > 1`, `run_data_parallel_controller_process`; otherwise, initialize a Scheduler for each `tp_rank`.
3. Run TokenizerManager and DetokenizerManager as subprocesses: the former tokenizes data for the Scheduler, and the latter detokenizes Scheduler outputs for the server frontend. For multi-node inference (e.g., Llama 3.1 405B), TokenizerManager and DetokenizerManager only run on the first node.
4. Apply chat templates via (if specified) and wait for Scheduler processes to signal readiness while collecting their configuration.

## Forward Requests From Server

The Server employs a FastAPI app to define API endpoints, forwarding `/v1/chat/completions` requests to TokenizerManager via [v1_chat_completions](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/openai_api/adapter.py).


1. Parse JSON from `raw_request` into a `ChatCompletionRequest`, then convert it to `GenerateReqInput` and configure `sampling_params` using `v1_chat_generate_request`.
2. Invoke TokenizerManager `generate_request` and handle streaming or non-streaming responses based on the `stream` parameter.
3. For streaming, process `generate_request` output incrementally with `generate_stream_resp`; for non-streaming, await the result and convert it to a `ChatCompletionResponse` via `v1_chat_generate_response`.

## Generate Request In TokenizerManager

TokenizerManager runs in Server's main weprocess, handling request tokenization. It is initialized in `launch_server`, with details in [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py).

### Initialization

1. Set up ZeroMQ for inter-process communication, including sockets for DetokenizerManager and Scheduler.
2. Configure `server_args`, enable `metrics`, initialize `model_config`, `tokenizer`, and placeholders for multi-modal image processors.

### generate_request

1. Create an event loop if not already initialized.
2. Pause processing if model weights are updating via `update_weights_from_disk` or `update_weights_from_distributed`.
3. Validate request compatibility with the model's `is_generation` setting.
4. Normalize requests using `normalize_batch_and_arguments` to manage batching, parallel sampling, and default parameters.
5. Process single requests with `_tokenize_one_request`, send to the scheduler, and yield responses from `_wait_one_response`.
6. Process batch requests with `_handle_batch_request`, tokenize inputs, manage parallel sampling, interact with the scheduler, and yield responses in both streaming and non-streaming modes.

## Scheduler Receive Requests and Process Batches

The Scheduler runs as Server’s subprocess, initialized via `run_scheduler_process` and executing its infinite event loop with `event_loop_normal` or `event_loop_overlap`.

### Initialization

1. Set up ZeroMQ for communication with TokenizerManager and response handling.
2. Configure `server_args`, `port_args`, `model_config`, `sessions`, and initialize TpModelWorker or TpModelWorkerClient based on overlap scheduling.
3. Initialize tokenizer and processor for multimodal or standard models based on server arguments.
4. Manage caching using ChunkCache or RadixCache and configure SchedulePolicy.
5. Set up chunk prefill parameters and GrammarBackend for request processing.

###Event Loop

The Scheduler continuously executes its event loop, alternating between `get_next_batch_to_run`, `run_batch` and `process_batch_result`.

### get_next_batch_to_run

1. Merge `last_batch` with `running_batch` if applicable and prioritize prefill batches with `get_new_batch_prefill` for immediate execution.
2. If no prefill batch, update `running_batch` for decode batch by filtering requests, managing memory, and adjusting decoding parameters.

### run_batch

1. For generation models, use TpModelWorker’s `forward_batch_generation` for token prediction or `forward_batch_idle` for idle tasks, returning results to `event_loop_normal`.
2. For embedding or reward models, assert token extension, execute `forward_batch_embedding`, and return embeddings.

### Processing and Finalizing Results

 In serving engines, LLM inference is usually broken into prefill and decode stages for their different compute charactor. You can check [this post](https://huggingface.co/blog/martinigoyanes/llm-inference-at-scale-with-tgi) from HuggingFace regarding the concept of Prefill and Decode. In SGLang, [extend mode](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py) is used instead of prefill mode most of the time. Prefill initializes KV-Cache for new requests, typically using Paged KV-Cache. Extend updates existing KV-Cache incrementally, often leveraging Ragged Tensors for efficiency, making it ideal for long sequences or multi-turn tasks.

After `run_batch`, Scheduler processes batch results in event_loop_normal:

1. Decode mode processes outputs, updates request states, handles token and probability data, manages memory, and logs statistics.
2. Extend mode handles prefill results, processes input tokens, and prepares for further decoding or embedding.
3. Finished requests are cached via `cache_finished_req`, streamed to DetokenizerManager. Unfinished requests are updated and looped back into `get_next_batch_to_run` for further processing until completion.

## TpModelWorker Manage Forward and Token Sampling

TpModelWorker manages ModelRunner’s forward and sampling for batches of requests scheduled by Scheduler.

### Initialization

1. Initializes tokenizer, model configuration and ModelRunner.
2. Configures device settings and memory pool limits.

### forward_batch_generation

1. Create a `ForwardBatch` with `model_worker_batch` and ModelRunner.
2. Compute logits and sample next tokens using ModelRunner’s `forward` and `sample`.
3. Return Results to Scheduler for process_batch_result.

### forward_batch_embedding

Embedding requests bypass token sampling and directly retrieve embeddings via ModelRunner’s `forward`.

## ModelRunner Manage AttnBackend and Model Execution

ModelRunner initialize the AttnBackend and managing the loaded model to perform various types of forward passes.

### Initialization

Initializes distributed environment, loads the model, applies tensor parallelism, and sets up memory pool and attention backends.

### forward

The `forward` function determines the appropriate forward mode for processing batches based on the `forward_mode`:

1. `forward_decode`: Initializes metadata and processes the model's forward pass using input IDs and positions.
2. `forward_extend`: Initializes metadata and executes the forward pass for generation or embedding tasks.
3. `forward_idle`: Manages the forward pass when the forward mode is idle.

## Model Concert HuggingFace Models and Perform Forward

All [supported models](https://sgl-project.github.io/references/supported_models.html) can be found in [python/sglang/srt/models](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models). We take [Qwen2ForCausalLM](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2.py) and chat completion requests for example.

`Qwen2ForCausalLM` is structured as follows:

* `model`: weights used for the forward pass.
* `embed_tokens`: maps `input_ids` into `embeddings`.
* `lm_head`: projects the hidden states back to the vocabulary space.
* `logits_processor`: manipulates `logits` to perform tasks such as sampling and normalization.
* `pooler`: pooling mechanism for extracting embeddings or computing rewards.

### forward

The forward function in Qwen2ForCausalLM processes input sequences to produce logits for next-token prediction of chat completion requests or embeddings for reward/embedding requests:

1. Converts `input_ids` to embeddings using `embed_tokens`. Sequentially processes embeddings through Qwen2DecoderLayer layers.
2. Returns embeddings via `pooler` if `get_embedding` is True; otherwise, computes logits using `logits_processor`.

The most import acceleration comes from the interaction between `forward_batch` and AttentionBackend. 

## AttentionBackend Accelerate Model Forward

SGLang supports several AttentionBackends](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention) which accelerate model forward and KV cache reuse.

### Initialization

1. Configures wrappers for sliding window and cross-attention scenarios.
2. Allocates necessary buffers for workspace and key-value indices.
3. Prepares forward metadata for efficient attention computation.
4. Integrates CUDA graph support for optimized execution paths.

### init_forward_metadata

1. Decode Mode: Updates indices for decoding using `indices_updater_decode` and sets `forward_metadata` to use `decode_wrappers`.
2. Extend Mode: Determines if ragged forward is needed based on token count and wrapper count, then updates indices with `indices_updater_prefill`.
3. Metadata Assignment: Sets `forward_metadata` with flags for ragged use and prefix extension.

### forward_extend and forward_decode

1. Wrapper and Attention Type: Selects the appropriate wrapper and decides between ragged or paged attention for `forward_extend`, or picks the decode wrapper for `forward_decode`.
2. Attention Calculation: Computes attention, optionally managing key-value caching, and returns the reshaped output.

## DetokenizerManager Detokenize and Send to TokenizerManager

### Initialization

Sets up communication sockets and the tokenizer. Manages decoding states with `LimitedCapacityDict`.

### event_loop and trim_matched_stop

1. Receives data from the Scheduler, forwarding `BatchEmbeddingOut` directly or processing `BatchTokenIDOut` for detokenization.
2. Splits token IDs into `read_ids` and `surr_ids`. Converts token IDs to text using `batch_decode`. Updates `DecodeStatus` with new offsets and decoded text.
3. Trims outputs at stop sequences, combines decoded text with metadata into `BatchStrOut`, and sends it to TokenizerManager.

## FastAPI Wraps the Output

1. DetokenizerManager sends `BatchStrOut` to TokenizerManager via ZeroMQ.
2. TokenizerManager updates request states and prepares decoded text for FastAPI.
3. For streaming, use an async generator and `StreamingResponse` in FastAPI.
4. For non-streaming, collect and send the complete response using `ORJSONResponse`.
