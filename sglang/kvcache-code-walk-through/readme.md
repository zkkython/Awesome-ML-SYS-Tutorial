# KV Cache

This doc explain on a high level how the KV cache being managed following the lifecycle of a request

## Resources

There are 2 level memory pools to manage KV cache. `req_to_token_pool` maps a request to its token locations. `token_to_kv_pool` maps a token location to its KV cache data, `token_to_kv_pool` have model specific implementation like MHA, MLA, DoubleSparse.

### **req_to_token_pool**
- **Layout:** #Requests * #Tokens
- **Access:** 
    - Dim0: `req_pool_indices`
    - Dim1: token positions in req
    - Value: `out_cache_loc` for token
  
### **token_to_kv_pool**
- **Layout:** #Layers * #Tokens * #Head * Head Dimension
- **Access:** 
    - Dim0: `layer_id`
    - Dim1: `out_cache_loc`
    - Dim2: Head
    - Dim3: Head Dimension
    - Value: `cache_k` for k_buffer and `cache_v` for v_buffer

## Workflows
![alt text](kvcache-code-walkthrough.png)
This section would give a detailed explaination for each step in the workflow and how the 2 resources being accessed in each step.

### Scheduler ([scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)) ([schedule_batch.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py))
<!-- 
- Must have detailed explanation for interaction between `Scheduler` and `Radix Cache` - init_next_runs, cache_unfinished, cache_finished
- Must have detailed event sequences for how 2 pools being updated in schedulers functions listed in the diagram
  - prepare_for_ext, prepare_for_dec, potentially process_batch_result?
- Could have ScheduleBatch -> Model Runner Batch -> Forward Batch flow 
-->
We will go through the important functions that are interacting with the 2 resources by sequence.
#### Prefill Batch
##### 1. Function `get_new_batch_prefill` 
  - Get the prefix from radix tree cache for request
  - Invoke `prepare_for_extend`
    - `req_to_token_pool`
      - Allocate to `req_pool_indices`
      - Add prefix to `req_to_token_pool`
    - `token_to_kv_pool`
      - Allocate (the sum of each batch's (number of input id tokens - number of prefix tokens)) `out_cache_loc`
      - For example: in above diagram, the batch size is 1
        - number of input id tokens = 3 -> A,B,C 
        - number of prefix tokens = 1 -> A
        - We will allocate 2 slots to `out_cache_loc` for token B, C
##### 2. Function `run_batch` 
Run `forward_extend` on the current batch, this will eventually invoke the backend forward_extend, which is responsible for the attention calculation and also save the kv cache for the input token in `token_to_kv_pool`. 

For example: In above step, we get 2 slots for token B, C in `out_cache_loc`, their corresponding K, V would be saved to this 2 slots here.

##### 3. Function `process_batch_result_prefill`
  - If the request is finished, invoke `cache_finished_req` (refer to [PLACEHOLDER] for details of `cache_finished_req` )
  - elese invoke `cache_unfinished_req` (refer to [PLACEHOLDER] for details of `cache_unfinished_req` )

Decode Batch
1. `update_running_batch` 
  - Invoke `prepare_for_decode`
    - `req_to_token_pool` No change
    - `token_to_kv_pool`
      - Allocate (batch size * 1) slot to `out_cache_loc` becuase we only generate one token for each batch in decode mode
      - For example: in above diagram, the round that generate D from C
        - We will allocate 1 slots to `out_cache_loc` for token D
2. `run_batch`
Run `forward_decode` on the current batch, this will eventually invoke the backend forward_decode, which is responsible for the attention calculation and also save the kv cache for the input token in `token_to_kv_pool`. 

For example: In above step, we get 1 slots for token D in `out_cache_loc`, it's corresponding K, V would be saved to this 1 slot here.

3. `process_batch_result_decode`
  - If the request is finished, invoke `cache_finished_req` (refer to [PLACEHOLDER] for details of `cache_finished_req` )
  - No operation for cache is needed for unfinished request

### Backend ([attention backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/))
Using Flash Infer as example:
<!-- 
- Must have mention save KV and update token_to_kv_pool
- Could mention the interface abstraction, i.e among flash infer and triton 
-->
Step 4. `foward_extend`
In foward_extend, Attention Backend is responsible for 
- Save the kv cache of extend tokens.
  - Save KV cache for extends token to `token_to_kv_pool`
- Run forward with extend tokens as input.
  - Get all cached tokens' `out_cache_loc` from `req_to_token_pool` ([create_flashinfer_kv_indices_triton](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py#L856))

Step 6.8. `foward_decode`
In foward_decode, Attention Backend is responsible for 
- Save the kv cache of decode token.
  - Save KV cache for decode token to `token_to_kv_pool` ([save_kv_cache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py#L426))
- Run forward with decode token as input. 
  - Get all cached tokens' `out_cache_loc` from `req_to_token_pool` ([create_flashinfer_kv_indices_triton](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py#L856))

### Radix Cache (radix_cache.py)
<!-- 
- Must Have explanation on each functions based on their callers
- Must have diagram for radix tree updates
- Must have diagram for radix tree updates under multiple requests
- Could compare between chunked cache and radix cache 
-->
Step 2. `match_prefix`
