# KV Cache

This document explains on a high level how the KV cache is managed following the lifecycle of a request.

## Overview
![alt text](scheduler_overview.png)
The figure illustrates how the **Scheduler** directs requests from the **Waiting Queue** into a **New Batch** (prefill phase) and then into the **Running Batch** (decode phase). It provides a high-level view of the batching lifecycle—covering prefill, decode, and the decision logic behind when to switch modes—alongside the main functions in the code (e.g., `recv_requests`, `get_new_batch_prefill`, `run_batch`, `process_batch_result`).

### Resources

1. Waiting Queue (Requests)  
- **New requests** arrive and are placed into the waiting queue.  
- Each request may include tokenized input IDs (e.g., text/image tokens) or embedding vectors.  
- The queue may be **reordered** (by priority or memory availability) before requests are pulled to form a batch.

2. Scheduler  
*(For brevity, only key steps are highlighted)*  
1. **Polling for requests**: Continuously retrieves new requests (`recv_requests`) and enqueues them if they are valid.  
2. **Prefill batch creation**: When feasible, combines as many waiting requests as possible into a single **New Batch** (`get_new_batch_prefill`). Checks available memory (`token_to_kv_pool`, `req_to_token_pool`) and stops if memory is insufficient or a batch limit is reached.  
3. **Prefill/Decode switching**: When no new prefill batch can be formed—or there are ongoing requests still in progress—triggers the decode phase (`update_running_batch`).  
4. **Running the batch**: Invokes `run_batch` to execute a forward pass (prefill or decode) on the current batch.  
5. **Processing results**: Calls `process_batch_result` to determine which requests have finished and which continue. Finished requests are handled via `cache_finished_req`, while incomplete requests are handled via `cache_unfinished_req`.

3. New Batch (for Prefill)  
- **Definition**: A group of requests pulled from the waiting queue via scheduling logic (e.g., priority, memory constraints) that will undergo the **prefill** stage.  
- **Existence**: Only exists during `get_new_batch_prefill`. Once that function finishes, if a new batch was created, it becomes the **Global Batch** for the upcoming iteration.  
- **Splitting large requests**: If a request requires more tokens than the available memory (`remaining_tokens`), it may be **chunked** into smaller parts (e.g., `Req 5b` and `Req 5c` in the figure).  
- **Prefill mode**: The scheduler prepares the input space—accounting for any prefixes already cached (e.g., in a `radixcache`)—and performs a forward pass on these new requests to initialize their hidden states, KV caches, etc.

4. Running Batch  
- **Definition**: Consists of requests that finished **prefill** but are not yet complete. These requests proceed to the **decode** phase to generate additional tokens.  
- **Decode mode**: The scheduler steps through token generation (one token at a time per request) using `prepare_for_decode` and `run_batch`.  
- **Memory constraints**: If available memory is insufficient during decode, the scheduler may **retract** certain requests (via `retract_decode`) from the running batch, returning them to the waiting queue for later processing.  
- **Completion & resource release**: Once decode ends or a request hits a stop condition, the request is flagged as finished, and its allocated memory is freed.

5. Global Batch  
- **Definition**: The batch that the Scheduler processes in each iteration of its main loop by calling `run_batch`.  
- **Selection**:  
  - If a **New Batch** was successfully created in the current cycle, that becomes the **Global Batch** (for prefill).  
  - Otherwise, the **Running Batch** is used (for decode).  
- **Execution**: Every time the Scheduler runs `run_batch` on the **Global Batch**, it performs a complete forward or decode step, updating each request’s status accordingly.

### Putting Them Together

1. **Continuous Polling**: The Scheduler loops, calling `recv_requests` to collect newly arrived requests, which are placed into the waiting queue.  
2. **Forming the Prefill Batch**: It attempts to build a **New Batch** (`get_new_batch_prefill`) by checking memory availability and packing as many requests as possible.  
3. **Prefill or Decode**:  
   - If a **New Batch** is formed, those requests enter the prefill phase.  
   - If no new prefill batch is formed—or existing requests are still in progress—decode begins or continues.  
4. **Running the Batch**: Once the **Global Batch** is determined (prefill vs. decode), `run_batch` is called to run a forward pass.  
5. **Result Processing**: After `run_batch`, the Scheduler calls `process_batch_result` to update request statuses. Finished requests go through `cache_finished_req`; others are retained via `cache_unfinished_req`.  
6. **Iteration**: The loop repeats until all requests are eventually completed. If insufficient memory is encountered, requests may be chunked (in prefill) or retracted (in decode), then reinserted into the waiting queue for later processing.

## Resources

There are two-level memory pools to manage KV cache. `req_to_token_pool` maps a request to its tokens' KV cache indices. `token_to_kv_pool` maps a token KV cache indices to its KV cache data, `token_to_kv_pool` has model-specific implementation like MHA, MLA, DoubleSparse.

### **req_to_token_pool**
- **Layout:** #Requests * #Tokens
- **Access:** 
    - Dim0: `req_pool_indices`
    - Dim1: token positions in req, starting from 0
    - Value: `out_cache_loc` for token
  
### **token_to_kv_pool**
- **Layout:** #Layers * #Tokens * #Head * Head Dimension
- **Access:** 
    - Dim0: `layer_id`
    - Dim1: `out_cache_loc`
    - Dim2: Head
    - Dim3: Head Dimension
    - Value: `cache_k` for k_buffer and `cache_v` for v_buffer

### **Radix Tree Cache**
A tree structure to enhance the reuse of prefix KV cache
- **Access:**
  - Key: Token ID
  - Value: Token's KV Indices

## Workflows
![alt text](kvcache-code-walkthrough.png)
Following the graph, this section provides a step-by-step walkthrough of the key functions that interact with the two resources. To facilitate this explanation, we will make a few assumptions:
- We use Flash Infer as backend

### Scheduler And Attention Backend ([scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)) ([schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)) ([Attention Batckend](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention))
<!-- 
- Must have detailed explanation for interaction between `Scheduler` and `Radix Cache` - init_next_runs, cache_unfinished, cache_finished
- Must have detailed event sequences for how 2 pools being updated in schedulers functions listed in the diagram
  - prepare_for_ext, prepare_for_dec, potentially process_batch_result?
- Could have ScheduleBatch -> Model Runner Batch -> Forward Batch flow 
-->
#### Prefill Batch
##### 1. Function `get_new_batch_prefill` 
  - Update prefix from radix tree cache for request, the `prefix_indices` will be updated too based on the prefix we get
  - Invoke `prepare_for_extend`
    - `req_to_token_pool`
      - Allocate to `req_pool_indices`
      - Add prefix to `req_to_token_pool`
    - `token_to_kv_pool`
      - Allocate (the sum of each reqs (number of input id tokens - number of prefix tokens)) `out_cache_loc`
      - For example: in above diagram, the batch size is 1
        - number of input id tokens = 3 -> A,B,C 
        - number of prefix tokens = 1 -> A
        - We will allocate 2 slots to `out_cache_loc` for token B, C
        
##### 2. Function `run_batch` 
Run `forward_extend` on the current batch, this will eventually invoke the Attention Backend, who is responsible for 
- Set the kv cache of extend tokens.
  - Set KV cache for extends token to `token_to_kv_pool` (Function `save_kv_cache`)
  - For example: In above step, we get 2 slots for token B, C in `out_cache_loc`, their corresponding K, V would be set to this 2 slots here.
- Run forward attention calculation, the input would be
  - Q = extend tokens, in our example token B, C
  - KV = All cached tokens from `req_to_token_pool` by `out_cache_loc` including A(prefix tokens), B, C(extend tokens) (Function `create_flashinfer_kv_indices_triton`).

##### 3. Function `process_batch_result_prefill`
  - If the request is finished, invoke `cache_finished_req` (refer to [PLACEHOLDER] for details of `cache_finished_req` )
  - else invoke `cache_unfinished_req` (refer to [PLACEHOLDER] for details of `cache_unfinished_req` )

#### Decode Batch
##### 1. Function `update_running_batch` 
  - Invoke `prepare_for_decode`
    - `req_to_token_pool` No change
    - `token_to_kv_pool`
      - Allocate (batch size * 1) slot to `out_cache_loc` because we only generate one token for each batch in decode mode
      - For example: in above diagram, the round that generate D from C
        - We will allocate 1 slots to `out_cache_loc` for token D

##### 2. Function `run_batch`
Run `forward_decode` on the current batch, this will eventually invoke the Attention Backend, who is responsible for 
- Save the kv cache of decode token.
  - Save KV cache for decode token to `token_to_kv_pool` (Function `save_kv_cache`)
  - For example: In above step, we get 1 slots for token D in `out_cache_loc`, it's corresponding K, V would be saved to this 1 slot here.
- Run forward, the input would be:
  - Q = decode token, in our example token D
  - KV = All cached tokens from `req_to_token_pool` by `out_cache_loc` including A, B, C(from previous round), D (Function `create_flashinfer_kv_indices_triton`)

##### 3. Function `process_batch_result_decode`
  - If the request is finished, invoke `cache_finished_req` (refer to [PLACEHOLDER] for details of `cache_finished_req` )
  - No operation for cache is needed for unfinished request

### Radix Cache (radix_cache.py)
<!-- 
- Must Have explanation on each functions based on their callers
- Must have diagram for radix tree updates
- Must have diagram for radix tree updates under multiple requests
- Could compare between chunked cache and radix cache 
-->
TODO
