# KV Cache Code Walkthrough
<!-- TODO(xiaotong):  CN -->
This document offers a comprehensive overview of the KV cache management system within the SGLang implementation, following lifecycle of requests and the key components that are involved in this lifecycle, including the Scheduler, Radix Cache, Attention Backend, and the global state that are maintained.

To facilitate this explanation, we will make a few assumptions in our examples:
- We use FlashInfer as backend.
- We use Longest Prefix as priority for requests in `waiting_queue` (e.g., the longest-prefix-match scheduling in SGLang paper).
- We don't consider jump forward decoding/speculative decoding.
- We don't enable `enable_mixed_chunck`.
- We use `Radix Cache` as `tree_cache`.

<!-- TODO:  这里概述下我们整体的叙述逻辑，类似于有一个 table of content，讲述先讲了什么，然后讲了什么，最后讲了什么 -->

## Global State

This section provides a brief overview for some of the important global state that are maintained across requests.

<!-- TODO(xiaotong):  add prefill vs extend's difference -->

### KV Cache & Memory Pools

KV Cache is among the most important global states in the server. 

<!-- TODO: 这里解释下 sglang 里面 kv cache 在何处被使用到 -->

There are two-level memory pools to manage KV cache. 

**`req_to_token_pool`**

`req_to_token_pool` maps a request to its tokens' KV cache indices. It's shape is `max-running-requests * max-total-tokens`, i.e., the maximum number of requests to run concurrently (or the maxium batch size) * the maximum number of tokens that can be stored into the KV cache. Use mainly for debugging.

- **Access:** 
    - Dim0: `req_pool_indices`
    - Dim1: token positions in req, starting from 0, 1, 2...
    - Value: `out_cache_loc` for token

  <!-- TODO：这里如果按照顺序来读的话，req_pool_indices 的意义还方便猜测，”大概是req_to_token_pool 的第一层 index，索引到这个 batch 里面的具体某个 req，然后第二层是 request 里面的 token 吧。但是 out_cache_loc 的意义呢？“-->
  
**`token_to_kv_pool`**

<!-- TODO: 这里解释下 req_to_token_pool 和 token_to_kv_pool 的联系，我理解是 req_to_token_pool 的 out_cache_loc 是 token_to_kv_pool 的 index，然后 token_to_kv_pool 的 cache_k 和 cache_v 是 token 具体的 KV cache 值？-->

Following `req_to_token_pool` which maps a request to its tokens' KV cache indices, `token_to_kv_pool` further maps a token's KV cache indices to its real KV cache data. Note that , for different attention implementation, like [`MHA`](https://arxiv.org/abs/1706.03762), [`MLA`](https://arxiv.org/abs/2405.04434), [`Double Sparsity`](https://arxiv.org/abs/2408.07092), `token_to_kv_pool` could have different implementation. The shape of `token_to_kv_pool` is `Number of Layers in the Model * Max Allowed Tokens Number Per Layers * Number of Head * Head Dimension`.

<!-- TODO: 我还是不理解 out_cache_loc 的含义-->

- **Access:** 
    - Dim0: `layer_id`
    - Dim1: `out_cache_loc`
    - Dim2: Head
    - Dim3: Head Dimension
    - Value: `cache_k` for k_buffer and `cache_v` for v_buffer

<!-- TODO(shuai):  confirm this Max Allowed Tokens Number Per Layers -->

**`tree_cache`**

<!-- TODO: 解释 tree_cache 和 req_to_token_pool、token_to_kv_pool 的关系 -->

`tree_cache` is a tree structure to enhance the reuse of prefix KV Cache across requests.

<!-- TODO: 我不太理解了，如果 key 是 token id 的话，同一个 token 在多个 request 里面都有的话，怎么唯一得到这个 token 在某个 request 里的 KV cache 呢？或者说，我觉得我们得先讲清楚 tree cache 和 kv cache pool 彼此的使用方式，查找方式，再讨论具体的数据结构。 -->

- **Access:**
  - Key: Token ID
  - Value: Token's KV Indices

### Active Requests Management

<!-- TODO: 如果让我来写的话，整篇文章我会先从 scheduler 开始。在 xxx 文章中，我们已经介绍了 SGLang 一个 request 的生命周期。我们接着从 scheduler 开始，先先从 request 开始梳理 scheduler 如何管理 requests，再细致介绍如何管理 kv cache。现在这个写法，前后链接有点不清晰。-->

The `Scheduler` manages active requests by utilizing these following core global states.

**`waiting_queue`**

<!-- TODO: 具体是什么数据结构，按名字来说是 queue，先进先出，但实际上不是先进先出的？-->

The Waiting Queue is a data structure designed to hold active requests. It dynamically changes the order of requests based on priority or available memory to optimize batch processing time.

Newly arrived requests are enqueued into the waiting queue. Requests that are returned from `retract_decode`. The highest priority requests (in our assumption, the requests with longest prefix) are dequeued from the queue to form a batch.

<!-- TODO: Requests that are returned from `retract_decode`. 这句话的上下文呢？没看懂什么意思-->

**`new_batch`**

A batch of requests that are ready for prefill/extend stage.

- **Chunked Prefill**: If a request requires more tokens than the available memory (`remaining_tokens`), it may be **chunked** into smaller parts.

- **Some Additional Key Points** 
  - **Chunked Prefill**: If a request requires more tokens than the available memory (`remaining_tokens`), it may be **chunked** into smaller parts.
  - Requests in `new_batch` would go through prefill/extend.
  - After prefill/extend, `new_batch` will transit to **Global Batch** for the upcoming iteration.  

#### `running_batch`  
- **Purpose:**: A batch of requests that are ready for decode stage.
- **Some Additional Key Points** 
  - **Retract**: If available memory is insufficient during decode, the scheduler may retract certain requests (via `retract_decode`) from the `running_batch`, returning them to the waiting queue for later processing.

#### `cur_batch`
- **Purpose:**: The batch of requests that are currently being processed in the main loop of Scheduler (`run_batch` function).  
- **Some Additional Key Points**
  - `cur_batch` is assigned in `event_loop_normal`.
  - The logic of forming `cur_batch`is: If there's requests ready for prefill (`new_batch`) in this cycle, use `new_batch` as `cur_batch`. Otherwise, `cur_batch` would process those that are ready for decode, thus use `running_batch` as `cur_batch`.  

## Scheduler Overview

This section provides a high-level overview of the `Scheduler`'s request management process.

### **Scheduler**
![alt text](scheduler_overview.png)
The figure illustrates how the **Scheduler** directs requests from the `waiting_queue` into a `new_batch` (for prefill/extend phase) and then into the `running_batch` (for decode phase).

#### 1. **New Request arrived**
The Scheduler continuously calls `recv_requests` to collect newly arrived requests, validate them and place them into the `waiting_queue`. In our example, `Req 7` are received and enqueued.

#### 2. **Merge Batches**
Before form the new batch for this round, Scheduler would merge the `cur_batch` from last round into `running_batch`. In the diagram, `cur_batch` from last round are shown as `cur_batch(i-1)` and `running_batch` are shown as `running_batch(i-1)`. In our example, `Req 0` and `Req 1` will be merged together into the new `running_batch`. 

During this process, the scheduler also removes the last round's `being_chunked_request`. These chunked requests (like `Req 5a`) are temporary fragments created solely for prefill operations to store KV cache, and are not meant to proceed to the decode phase. They serve their purpose once the KV cache is stored and should be removed.

To better understand the chunking process, let's look at how `Req 5` is handled:
1. Initially, `Req 5` enters the waiting queue as a single long request
2. During the first prefill attempt, due to memory constraints, it's split into `Req 5a` and `Req 5b`
3. After `Req 5a` is processed and its KV cache is stored, it's removed as a `being_chunked_request`
4. In the current round, when trying to process `Req 5b`, it still exceeds the resource limit
5. Therefore, `Req 5b` is further split into a new `Req 5b` (which goes into the global batch) and `Req 5c` (which stays in the waiting queue)

#### 3. **Forming the New Batch**:
The process of forming a new batch consists of three main steps:

1. **Reorder Waiting Queue**:
   - The scheduler reorders requests based on its scheduling policy
   - By default, it prioritizes requests with the longest prefix in the radix cache
   - This maximizes the reuse of existing KV cache entries

2. **Request Selection**:
   - Requests are selected one by one from the reordered queue
   - In our example:
     - `Req 6` is selected first as it has the longest prefix match in the cache
     - `Req 5b` is selected next as there's still available memory
     - However, `Req 5b` exceeds the memory limit and is further split into `Req 5b` and `Req 5c`
     - `Req 5b` is added to the new batch while `Req 5c` remains in the waiting queue

3. **Prepare for Extend**:
   - Once the new batch is formed, space is allocated for the selected requests
   - This includes allocating space in both req_to_token and token_to_kv pools

If no new batch can be formed, the scheduler will filter the running batch (removing finished requests like `Old Finished Req`) and use it as the current batch. Additionally, if GPU memory becomes insufficient, some decoding requests may be retracted according to the retraction policy. In our example, `Req 0` is retracted during this process.

#### 4. **Running the Batch**: 
Once the **Global Batch** is determined, `run_batch` is called to run a forward pass.

#### 5. **Result Processing**:
After `run_batch`, the Scheduler calls `process_batch_result` to to determine which requests have finished and which continue. In our example, `Req 6` is finished and turns grey, `Req 5b` remains unfinished.

#### 6. **Iteration**:
The loop repeats until all requests are eventually completed. If insufficient memory is encountered, requests may be chunked (in prefill) or retracted (in decode), then reinserted into the waiting queue for later processing.

<!-- TODO(mingyuan):  CN -->
## One Request Lifecycle
![alt text](kv-cache-request-lifecycle.png)

Following one request lifecycle, this section provides a step-by-step walkthrough of the key functions that updates the KV Cache & Memory Pools, we will use request ABC as an example.

#### Prefill Batch
##### Step 1. Function `get_new_batch_prefill` 
  - Update prefix from radix tree cache for request
    - When `ABC` come in, say the current radix cache have one node `AFG` thats currently being referenced.
    - `match_prefix` would try to find the prefix for `ABC` in the current radix cache, and it matched up to the first token `A` in the node.
    - Radix Cache Split the node into `A` and `FG`, node `A` is the current last node for this request.
  - Invoke `prepare_for_extend`
    - `req_to_token_pool`
      - Allocate to `req_pool_indices`
      - Add prefix to `req_to_token_pool`
    - `token_to_kv_pool`
      - Allocate (the sum of each reqs (number of input id tokens - number of prefix tokens)) `out_cache_loc`
      - In our example of request ABC, the batch size is 1
        - number of input id tokens = 3 -> A,B,C 
        - number of prefix tokens = 1 -> A
        - We will allocate 2 slots to `out_cache_loc` for token B, C
        
##### Step 2. Function `run_batch` 
Run `forward_extend` on the current batch, this will eventually invoke the Attention Backend, who is responsible for 
- Set the kv cache of extend tokens.
  - Set KV cache for extends token to `token_to_kv_pool` (Function `save_kv_cache`)
  - In our example: from above steps, we get 2 slots for token B, C in `out_cache_loc`, their corresponding K, V would be set to this 2 slots here.
- Run forward attention calculation, the input would be
  - Q = extend tokens, in our example token B, C
  - KV = All cached tokens from `req_to_token_pool` by `out_cache_loc` including A(prefix tokens), B, C(extend tokens) (Function `create_flashinfer_kv_indices_triton`).

##### Step 3. Function `process_batch_result_prefill`
  `cache_finished_req` and `cache_unfinished_req` are responsible for managing the KV cache in Radix Cache, ReqToTokenPool, and TokenToKVPool.
  - If the request is finished, invoke `cache_finished_req` (refer to [this secion](#radixcache-cache_finished_req--cache_finished_req) for details of `cache_finished_req` )
  - else invoke `cache_unfinished_req` (refer to [this secion](#radixcache-cache_finished_req--cache_finished_req) for details of `cache_unfinished_req` )

  In our example, `cache_unfinished_req` is invoked after extend/prefill phase, `BC` was added as a child node for `A`, both nodes `A` and `BC` increase the lock reference, node `BC` become the `last_node` for the request.

#### Decode Batch
##### Step 4. Function `update_running_batch` 
  - Invoke `prepare_for_decode`
    - `req_to_token_pool` No change
    - `token_to_kv_pool`
      - Allocate (batch size * 1) slot to `out_cache_loc` because we only generate one token for each batch in decode mode
      - For example: in above diagram, the round that generate D from C
        - We will allocate 1 slots to `out_cache_loc` for token D

##### Step 5. Function `run_batch`
Run `forward_decode` on the current batch, this will eventually invoke the Attention Backend, who is responsible for 
- Save the kv cache of decode token.
  - Save KV cache for decode token to `token_to_kv_pool` (Function `save_kv_cache`)
  - For example: In above step, we get 1 slots for token D in `out_cache_loc`, it's corresponding K, V would be saved to this 1 slot here.
- Run forward, the input would be:
  - Q = decode token, in our example token D
  - KV = All cached tokens from `req_to_token_pool` by `out_cache_loc` including A, B, C(from previous round), D (Function `create_flashinfer_kv_indices_triton`)

##### Step 6. Function `process_batch_result_decode`
  If the request is finished, invoke `cache_finished_req` (refer to [this secion](#radixcache-cache_finished_req--cache_finished_req) for details of `cache_finished_req` ). No operation for cache is needed for unfinished request in decode phase.
  
  In our example, `DE` is appended to node `BC`, and the lock reference for node `A` and `BCDE` got decreased. 

<!-- TODO(yangmin):  CN -->
#### RadixCache `cache_finished_req` & `cache_finished_req`
This section would go deeper on `cache_finished_req` & `cache_finished_req`'s flow.

##### **Quick Overview of `cache_finished_req()` vs. `cache_unfinished_req()`**
| Sequence | `cache_unfinished_req()` | `cache_finished_req()` |
|------|--------------------------|--------------------------|
| **1. Get `kv_indices`** from `req_to_token_pool.req_to_token` | - | - |
| **2. Update Radix Cache** (`insert()`) | - | - |
| **3. Free KV Cache** (`self.token_to_kv_pool.free()`) | - | - |
| **4. Handle `req_to_token_pool`** | **Writes and updates** `req_to_token_pool | **Releases** `req_to_token_pool` as the request is completed. |
| **5. Handle `req.last_node`** | **Increases** the reference count of `req.last_node` | **Decreases** the reference count of `req.last_node`, as `req` is finished. |

As we can observe, the core functionality is essentially the same for `cache_unfinished_req()` and `cache_finished_req()`, we are going to walk through the code for how req_to_token_pool, token_to_kv_pool and tree_cache are updated during `cache_unfinished_req()`, and explain the key difference between `cache_unfinished_req()` and `cache_finished_req()`.

##### `cache_unfinished_req`
1. Get KV Index: Get KV index from `req_to_token_pool.req_to_token`

2. Update Radix Cache
```python
new_prefix_len = self.insert(token_ids, kv_indices.clone())
```
This method inserts token_ids and their corresponding kv_indices into the Radix Cache. If successful, it returns a new prefix length (new_prefix_len).

3. `Free KV Cache` 

4. Update `prefix_indices` and `last_node`
  
    Calls `match_prefix()` to update the request’s `prefix_indices` and `last_node`. This is important as `prefix_indices` and `last_node` will be used in next iteration.
  - `prefix_indices` is used to calculate the needed size of extend/prefill tokens.
  - `last_node` is used to keep track of a requests last node in the radix tree so that it can be used to trace back towards root node when updating lock.

5. Lock Management for Memory Safety
To prevent unintended deletion of active cache nodes. Keeping a lock on a node shields it from being freed while still needed. After above state transition, the old `last_node` is unlocked using dec_lock_ref(), allowing it to be freed when no longer in use. The new `last_node` is locked, protecting it from deletion.

##### `cache_finished_req`

###### **Similar to `cache_unfinished_req()`, `cache_finished_req()` also has the following steps:**
1. When a request `req` is completed, its `token_ids` are stored in the **Radix Cache**. Update Radix Cache 
2. **Release** redundant **KV Cache space** in `token_to_kv_pool` (removing duplicates).
3. **Release `req_to_token_pool`** and **update `tree_cache`**.  
