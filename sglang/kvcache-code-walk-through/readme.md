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
Step 1. `init_next_round_input` 
Step 3. `prepare_for_extend`

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
