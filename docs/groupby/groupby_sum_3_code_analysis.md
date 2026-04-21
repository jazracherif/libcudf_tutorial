# cuDF `groupby` + `SUM` — Part III: Code Analysis

> **Part of a three-document series:**
> - [Part I — Algorithm Overview](groupby_sum_1_algorithm_overview.md): high-level description of the hash groupby algorithm, data structures, the two-kernel aggregation strategy, and the life of `global_mapping_indices`.
> - [Part II — Nsight Analysis](groupby_sum_2_nsight_analysis.md): ground-truth kernel table and performance breakdown from an actual Nsight Systems capture on 100M rows.
> - **Part III — Code Analysis** *(this file)*: function-by-function walk-through of the cuDF, cuCollections, RMM, and CCCL source, with annotated call stack and library layer summary.

This document traces the exact call stack from the public `groupby::aggregate()` entry point down to the individual atomic SUM update, linking every step to the source line where it happens. It assumes familiarity with the algorithm described in Part I and uses the kernel labels (A, B, 1, D, E, …) established in Part II.

---

## Table of Contents

- [1. Public Entry Point: `groupby::groupby()` Constructor](#1-public-entry-point-groupbygroupby-constructor)
- [2. `groupby::aggregate()`](#2-groupbyaggregate)
- [3. `dispatch_aggregation()` — Hash vs. Sort Decision](#3-dispatch_aggregation--hash-vs-sort-decision)
- [4. `detail::hash::groupby()`](#4-detailhashgroupby)
- [5. `dispatch_groupby()` — Row Comparator & Hasher Setup](#5-dispatch_groupby--row-comparator--hasher-setup)
- [6. `compute_groupby()` — The `cuco::static_set` and Main Orchestration](#6-compute_groupby--the-cucostaticset-and-main-orchestration)
  - [6a. Hash caching (optional Thrust kernel)](#6a-hash-caching-optional-thrust-kernel)
  - [6b. `cuco::static_set` construction — the central hash table](#6b-cucostatic_set-construction--the-central-hash-table)
  - [6c. Call to `compute_single_pass_aggs()`](#6c-call-to-compute_single_pass_aggs)
  - [6d. Gather unique key rows into output table](#6d-gather-unique-key-rows-into-output-table)
- [7. `compute_single_pass_aggs()` — Two-Path Strategy](#7-compute_single_pass_aggs--two-path-strategy)
  - [7a. Pre-processing](#7a-pre-processing)
  - [7b. Path A — Shared Memory Aggregations](#7b-path-a--shared-memory-aggregations-preferred-for-low-group-cardinality)
  - [7c. Path B — Global Memory Fallback](#7c-path-b--global-memory-fallback-compute_global_memory_aggs)
- [8. `update_target_element<Source, SUM>` — The Atomic SUM](#8-update_target_elementsource-sum--the-atomic-sum)
- [9. Result Finalization](#9-result-finalization)
- [10. Library Layer Summary](#10-library-layer-summary)
- [11. Complete Call Stack](#11-complete-call-stack-for-sum-hash-path-shared-memory-sub-path)
- [12. Key RMM Allocations Along the Path](#12-key-rmm-allocations-along-the-path)
- [13. CCCL / libcudacxx / cuCollections Components](#13-cccl--libcudacxx--cucollections-components)
- [14. `cuco::static_set` Internals](#14-cucostatic_set-internals)

---

## 1. Public Entry Point: `groupby::groupby()` Constructor (user called)

**File**: [cudf/cpp/src/groupby/groupby.cu](../../cudf/cpp/src/groupby/groupby.cu#L40)

```
groupby::groupby(table_view const& keys, null_policy, sorted, ...)
```

- Stores `_keys`, `_include_null_keys`, `_keys_are_sorted`, `_column_order`, `_null_precedence`.
- No GPU work yet. The `table_view` is a lightweight non-owning reference to the device memory columns.

---

## 2. `groupby::aggregate()` - (user called)

**File**: [cudf/cpp/src/groupby/groupby.cu](../../cudf/cpp/src/groupby/groupby.cu#L238)

```
std::pair<unique_ptr<table>, vector<aggregation_result>>
    groupby::aggregate(host_span<aggregation_request const> requests, stream, mr)
```

Steps:
1. Validates that every `request.values.size() == _keys.num_rows()`.
2. Calls `verify_valid_requests()` — checks type/aggregation compatibility.
3. If `_keys.num_rows() == 0` → returns empty results immediately.
4. Falls through to **`dispatch_aggregation()`**.

---

## 3. `dispatch_aggregation()` — Hash vs. Sort Decision

**File**: [cudf/cpp/src/groupby/groupby.cu](../../cudf/cpp/src/groupby/groupby.cu#L60)

```cpp
if (_keys_are_sorted == sorted::NO and not _helper
    and detail::hash::can_use_hash_groupby(requests))
    → detail::hash::groupby(...)   // ← taken for SUM on fixed-width types
else
    → sort_aggregate(...)
```

### `can_use_hash_groupby()` check

**File**: [cudf/cpp/src/groupby/hash/groupby.cu](../../cudf/cpp/src/groupby/hash/groupby.cu#L165)

The set `hash_aggregations` includes: `SUM, SUM_WITH_OVERFLOW, SUM_OF_SQUARES, PRODUCT, MIN, MAX, COUNT_VALID, COUNT_ALL, ARGMIN, ARGMAX, MEAN, M2, STD, VARIANCE`.

For `SUM` on a numeric (fixed-width, atomic-capable) type → returns `true` → **hash path taken**.

---

## 4. `detail::hash::groupby()`

**File**: [cudf/cpp/src/groupby/hash/groupby.cu](../../cudf/cpp/src/groupby/hash/groupby.cu#L192)

```cpp
std::pair<unique_ptr<table>, vector<aggregation_result>>
    groupby(table_view const& keys, requests, include_null_keys, stream, mr)
```

1. Creates a `cudf::detail::result_cache` (host-side map from `(column_ptr, agg_kind)` → output column).
2. Calls `dispatch_groupby()` which returns `unique_keys` table.
3. Calls `extract_results(requests, cache, stream, mr)` to finalize results from cache.

---

## 5. `dispatch_groupby()` — Row Comparator & Hasher Setup

**File**: [cudf/cpp/src/groupby/hash/groupby.cu](../../cudf/cpp/src/groupby/hash/groupby.cu#L83)

```cpp
auto preprocessed_keys = cudf::detail::row::hash::preprocessed_table::create(keys, stream);
auto comparator        = cudf::detail::row::equality::self_comparator{preprocessed_keys};
auto row_hash          = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
auto d_row_hash        = row_hash.device_hasher(nullate::DYNAMIC{...});
```

- **`preprocessed_table`**: Builds a column-order-aware view of the keys table, used by both the hasher and comparator. Resolves nested structs/lists.
- **`row_hasher`** / **`d_row_hash`**: Wraps `cudf::hashing::detail::default_hash` (MurmurHash3_x86_32) into a device callable that hashes an entire row by index.
- **`self_comparator`** / **`d_row_equal`**: Device callable that compares two row indices for equality, handling nulls according to `null_keys_are_equal = EQUAL`.

> **libcudacxx / CCCL** connection: `d_row_hash` is wrapped in `row_hasher_with_cache_t` (defined in [helpers.cuh](../../cudf/cpp/src/groupby/hash/helpers.cuh#L56)), which is used as the hash function for the `cuco::linear_probing` scheme.

Dispatches to `compute_groupby<row_comparator_t>()` (no nested columns) or `compute_groupby<nullable_row_comparator_t>()`.

---

## 6. `compute_groupby()` — The `cuco::static_set` and Main Orchestration

**File**: [cudf/cpp/src/groupby/hash/compute_groupby.cu](../../cudf/cpp/src/groupby/hash/compute_groupby.cu#L46)

### 6a. Hash caching (optional Thrust kernel)

> **Not taken for this dataset.** The `label` key is a single non-nested `utf8` column, so `count_nested_columns` returns 1. Since `1 ≤ HASH_CACHING_THRESHOLD=4`, `cached_hashes` is returned as an empty zero-size `device_uvector` and no kernel fires. The `row_hasher_with_cache_t` wrapper receives a null pointer for the cache and falls back to recomputing the hash on every probe.

If the keys table has more than `HASH_CACHING_THRESHOLD = 4` total columns (nested included), each row's hash is pre-computed once and cached to avoid re-hashing during the two passes inside `mapping_indices_kernel`:

```cpp
// CUDA kernel launched via Thrust (CCCL)
rmm::device_uvector<hash_value_type> hashes(num_keys, stream);   // RMM allocation
thrust::tabulate(rmm::exec_policy_nosync(stream), hashes.begin(), hashes.end(),
    [d_row_hash, row_bitmask] __device__(size_type idx) { return d_row_hash(idx); });
```

> **RMM**: `rmm::device_uvector<hash_value_type>` allocates device memory via the stream-ordered allocator pool from [rmm/cpp/include/rmm/device_uvector.hpp](../../rmm/cpp/include/rmm/device_uvector.hpp#L124).
> **CCCL/Thrust**: `thrust::tabulate` → maps to a single CUDA kernel launch.

### 6b. `cuco::static_set` construction — **the central hash table**

**File**: [cudf/cpp/src/groupby/hash/compute_groupby.cu](../../cudf/cpp/src/groupby/hash/compute_groupby.cu#L128)  
**CCCL component**: `cuco/static_set.cuh` (cuCollections, part of CCCL)

```cpp
auto set =
    cuco::static_set{
        cuco::extent<int64_t>{static_cast<int64_t>(num_keys)},  // initial capacity
        cudf::detail::CUCO_DESIRED_LOAD_FACTOR,                 // 50% load factor → 2× capacity
        cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL}, // sentinel = INT32_MAX
        d_row_equal,                                            // row equality comparator
        probing_scheme_t{row_hasher_with_cache_t{d_row_hash, cached_hashes.data()}},
        cuco::thread_scope_device,
        cuco::storage<GROUPBY_BUCKET_SIZE>{},                   // 1 slot per bucket
        rmm::mr::polymorphic_allocator<char>{},                 // ← RMM allocator
        stream.value()};
```

**Data structure details** ([helpers.cuh](../../cudf/cpp/src/groupby/hash/helpers.cuh)):
- `global_set_t = cuco::static_set<size_type, extent<int64_t>, thread_scope_device, row_comparator_t, probing_scheme_t, rmm::mr::polymorphic_allocator<char>, storage<1>>`
- Stores **row indices** (`size_type`) as keys. The actual data lives in the cuDF columns.
- Probing: **linear probing** with `GROUPBY_CG_SIZE=1` (one thread per operation).
- Memory backend: `rmm::mr::polymorphic_allocator<char>` — at runtime this routes to whichever `device_async_resource_ref mr` the user passed (e.g., pool allocator, arena allocator from RMM).

### 6c. Call to `compute_single_pass_aggs()`

```cpp
auto const [key_gather_map, has_compound_aggs] =
    compute_single_pass_aggs(set, row_bitmask, requests, cache, stream, mr);
```

### 6d. Gather unique key rows into output table

```cpp
return cudf::detail::gather(keys, key_gather_map, ...);  // Thrust-backed gather kernel
```

---

## 7. `compute_single_pass_aggs()` — Two-Path Strategy

**File**: [cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh](../../cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh#L30)

### 7a. Pre-processing

1. `extract_single_pass_aggs(requests, stream)` ([extract_single_pass_aggs.cpp](../../cudf/cpp/src/groupby/hash/extract_single_pass_aggs.cpp#L116)):
   - For `SUM` → kept as-is (single-pass, `is_agg_intermediate=false`).
   - For `MEAN` → decomposed into `SUM` + `COUNT_VALID`.
   - For `M2/STD/VARIANCE` → decomposed into `SUM_OF_SQUARES` + `SUM` + `COUNT_VALID`.
   - Returns: `values` table (view of value columns), `agg_kinds` vector, `has_compound_aggs` flag.

2. `d_agg_kinds` copied to device via `cudf::detail::make_device_uvector_async()` — **RMM** allocation.

3. Compute `grid_size`:
   - `max_active_blocks_mapping_kernel` → `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
   - `max_active_blocks_shmem_aggs_kernel` → same
   - `grid_size = min(max, ceil(num_rows / GROUPBY_BLOCK_SIZE))`, `GROUPBY_BLOCK_SIZE = 128`

4. `is_shared_memory_compatible()`: checks if dynamic shared memory is sufficient:
   - Queries `get_available_shared_memory_size(grid_size)` → `cudaDeviceGetAttribute`
   - Each column needs `sizeof(T) * GROUPBY_CARDINALITY_THRESHOLD` bytes in shared memory.

### 7b. Path A — Shared Memory Aggregations (preferred for low GROUP cardinality)

#### KERNEL 1: `mapping_indices_kernel`

**File**: [cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh](../../cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh#L120)  
**Host launcher**: [cudf/cpp/src/groupby/hash/compute_mapping_indices.cu](../../cudf/cpp/src/groupby/hash/compute_mapping_indices.cu)

```
Grid: (grid_size) blocks × (GROUPBY_BLOCK_SIZE=128) threads
Dynamic shared memory: GROUPBY_SHM_MAX_ELEMENTS=256 slots (block-local hash set)
```

**Per block, phases in the kernel**:

1. Each block allocates a **block-scoped `cuco::static_set_ref`** in shared memory (`__shared__ size_type slots[valid_extent.value()]`).
2. Threads stride through input rows: for each row, call `find_local_mapping()`:
   - `shared_set.insert_and_find(idx)` — inserts into the block's private shared-memory set.
   - If first occurrence: `local_mapping_indices[idx] = atomic_increment(cardinality)` (block-rank).
   - If already seen: `local_mapping_indices[idx] = local_mapping_indices[matched_idx]`.
   - If `cardinality > GROUPBY_CARDINALITY_THRESHOLD=128`: sets `needs_global_memory_fallback` flag and exits.
3. After the row loop: `find_global_mapping()` — for each unique key found in the shared set, call `global_set.insert_and_find(input_idx)` → write result to `global_mapping_indices[block_id * THRESHOLD + local_rank]`.

After kernel 1, host checks `needs_global_memory_fallback`. If set → falls back to **Path B**.

**RMM allocations** (host side, before kernel launch):
- `rmm::device_uvector<size_type> local_mapping_indices(num_rows, stream)`
- `rmm::device_uvector<size_type> global_mapping_indices(grid_size * GROUPBY_CARDINALITY_THRESHOLD, stream)`
- `rmm::device_uvector<size_type> block_cardinality(grid_size, stream)`
- `rmm::device_scalar<cuda::std::atomic_flag> needs_global_memory_fallback(stream)` ← **libcudacxx** type

**CCCL / libcudacxx in this kernel**:
- `cuda::atomic_ref<size_type, cuda::thread_scope_block>` — block-scoped atomic increment of `cardinality`.
- `cuda::std::atomic_flag` with `test_and_set(memory_order_relaxed)` — signals fallback needed.
- `cooperative_groups::this_thread_block()` — for `block.sync()` and `block.thread_rank()`.
- `cuco::static_set_ref` (block-scoped, in shared memory) — uses `cuco::thread_scope_block`.

#### KERNEL 2: `single_pass_shmem_aggs_kernel`

**File**: [cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu](../../cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu#L207)

```
Grid: same (grid_size) blocks × (GROUPBY_BLOCK_SIZE=128) threads
Dynamic shared memory: available_shmem_size bytes for partial aggregation results + offset tables
```

**Phases** (column-batched loop to fit shmem):

1. **`calculate_columns_to_aggregate()`** — determine which aggregation columns fit in the current shared-memory budget.

2. **`initialize_shmem_aggregations()`** — each thread initializes its partial accumulator slots:
   - `dispatch_type_and_aggregation(..., initialize_shmem{}, target, target_mask, idx)`
   - For `SUM` → `get_identity<T, SUM>()` → `DeviceType(0)` via CCCL `cuco/detail` operator identity.

3. **`compute_pre_aggregations()`** ← **Phase 1 of reduction**:
   - Each thread reads its assigned source row (`source_idx = global_thread_id`).
   - Computes `target_idx = local_mapping_indices[source_idx] + agg_location_offset` (block-local rank).
   - Calls `shmem_element_aggregator{}(target, target_mask, target_idx, source_col, source_idx)`.
   - For `SUM` → `update_target_element<Source, SUM>` → **`cudf::detail::atomic_add()`** into shared memory buffer.

4. `block.sync()` — ensure all partial aggregations in shmem are complete.

5. **`compute_final_aggregations()`** ← **Phase 2 of reduction**:
   - Each thread reads block-local partial result from shmem.
   - Computes `target_idx = global_mapping_indices[block_id * THRESHOLD + local_rank]` (global output row).
   - Calls `gmem_element_aggregator{}(target_col, target_idx, source_col, source, source_mask, idx)`.
   - For `SUM` → `update_target_element<Source, SUM>` → **`cudf::detail::atomic_add()`** into global output column.

### 7c. Path B — Global Memory Fallback (`compute_global_memory_aggs`)

**File**: [cudf/cpp/src/groupby/hash/compute_global_memory_aggs.cuh](../../cudf/cpp/src/groupby/hash/compute_global_memory_aggs.cuh#L162)

**Sub-strategy** — chosen by `h_agg_kinds.size() > GROUPBY_DENSE_OUTPUT_THRESHOLD=2`:

**Sparse output + gather** (`compute_aggs_sparse_output_gather`):
```cpp
// CCCL/Thrust kernel ~ O(num_rows) threads
thrust::for_each_n(rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0), num_rows,
    compute_single_pass_aggs_sparse_output_fn{set.ref(insert_and_find), ...});
// Then cudf::detail::gather to compact sparse results
```

**Dense output** (`compute_aggs_dense_output`):
```cpp
// CCCL/Thrust kernel
thrust::tabulate(..., compute_matching_keys_fn{set.ref(insert_and_find), ...});
thrust::for_each_n(..., num_rows * num_agg_cols,
    compute_single_pass_aggs_dense_output_fn{target_indices, ...});
```

In both strategies every per-element functor calls:
```cpp
// For SUM: device_aggregators.cuh:115
update_target_element<Source, aggregation::SUM>::operator()(target_col, target_idx, src_col, src_idx)
    → cudf::detail::atomic_add(&target.element<Target>(target_idx),
                               static_cast<Target>(source.element<Source>(source_idx)));
```

---

## 8. `update_target_element<Source, SUM>` — The Atomic SUM

**File**: [cudf/cpp/include/cudf/detail/aggregation/device_aggregators.cuh](../../cudf/cpp/include/cudf/detail/aggregation/device_aggregators.cuh#L115)

```cpp
// For fixed-width, atomic-capable, non-fixed-point, non-timestamp Source:
__device__ void operator()(mutable_column_device_view target, size_type target_index,
                           column_device_view source, size_type source_index) const noexcept
{
    using Target = target_type_t<Source, aggregation::SUM>;
    cudf::detail::atomic_add(
        &target.element<Target>(target_index),          // pointer into RMM-managed output column
        static_cast<Target>(source.element<Source>(source_index)));
}
```

- `cudf::detail::atomic_add` — thin wrapper around CUDA's `atomicAdd` intrinsic (or `cuda::atomic_ref` for larger types).
- For `int64` input (`amount`): `Target = int64_t` — same type, no widening needed.
- For `int32` input: `Target = int64_t` (widened to avoid overflow); for `float32`: `Target = float64`.
- The output column's device memory is an `rmm::device_buffer` owned by `cudf::column`.

---

## 9. Result Finalization

### `finalize_output()` → `extract_single_pass_aggs::extract_results()`

**File**: [cudf/cpp/src/groupby/hash/extract_single_pass_aggs.cpp](../../cudf/cpp/src/groupby/hash/extract_single_pass_aggs.cpp)  
**File**: [cudf/cpp/src/groupby/hash/output_utils.cu](../../cudf/cpp/src/groupby/hash/output_utils.cu)

- Reads aggregation results from `agg_results` table.
- Stores them in the `result_cache` (`cache[{column_ptr, agg_kind}] = column`).

### `extract_results()` — `hash/groupby.cu`

- Iterates over user-requested aggregations.
- For `SUM` → directly fetches the SUM result column from cache.
- For compound aggs (`MEAN`, `STD`) → `hash_compound_agg_finalizer` computes final value from cached intermediate columns using `thrust::transform` (Thrust/CCCL kernel).

---

## 10. Library Layer Summary

| Step | cuDF function | CCCL / libcudacxx component | RMM component |
|------|--------------|----------------------------|---------------|
| Row hashing | `row::hash::row_hasher` | — | — |
| Row equality | `row::equality::self_comparator` | — | — |
| Hash table | `cuco::static_set` | **cuCollections** (CCCL) | `polymorphic_allocator<char>` → pool |
| Block-shared hash table | `cuco::static_set_ref` (shmem) | **cuCollections** (CCCL) | shared memory (no RMM) |
| Block atomic cardinality | `cuda::atomic_ref<thread_scope_block>` | **libcudacxx** (CCCL) | — |
| Fallback flag | `cuda::std::atomic_flag` | **libcudacxx** (CCCL) | `rmm::device_scalar` |
| Hash-caching tabulate | `thrust::tabulate` | **Thrust** (CCCL) | `rmm::device_uvector` |
| Per-element reduction (global) | `thrust::for_each_n` | **Thrust** (CCCL) | — |
| SUM atomic update | `cudf::detail::atomic_add` | CUDA `atomicAdd` / libcudacxx | output column `device_buffer` |
| Stream-ordered allocs | — | — | `rmm::device_uvector`, `rmm::device_scalar`, `rmm::device_buffer` |
| Exec policy | `rmm::exec_policy_nosync(stream)` | Thrust policy | — |
| Gather unique keys | `cudf::detail::gather` | **Thrust** (CCCL) | — |
| Cooperative groups | `cooperative_groups::this_thread_block()` | CUDA toolkit | — |

---

## 11. Complete Call Stack (for SUM, hash path, shared-memory sub-path)

<pre>
[user code]
  cudf::groupby::groupby(table_view{{key_col}})   // <a href="../../cudf/cpp/src/groupby/groupby.cu#L40">groupby.cu:40</a>  — constructor, no GPU
  .aggregate(requests, stream, mr)                 // <a href="../../cudf/cpp/src/groupby/groupby.cu#L223">groupby.cu:223</a>

    groupby::dispatch_aggregation()                // <a href="../../cudf/cpp/src/groupby/groupby.cu#L53">groupby.cu:53</a>
      detail::hash::can_use_hash_groupby()         // <a href="../../cudf/cpp/src/groupby/hash/groupby.cu#L165">hash/groupby.cu:165</a>  → true for SUM

      detail::hash::groupby()                      // <a href="../../cudf/cpp/src/groupby/hash/groupby.cu#L192">hash/groupby.cu:192</a>
        dispatch_groupby()                         // <a href="../../cudf/cpp/src/groupby/hash/groupby.cu#L83">hash/groupby.cu:83</a>
          row::hash::preprocessed_table::create()  // sets up row hash/compare ops
          cuco::static_set{...}                    // (inside <a href="../../cudf/cpp/src/groupby/hash/compute_groupby.cu#L128">compute_groupby.cu:128</a>)
          compute_groupby&lt;row_comparator_t&gt;()      // <a href="../../cudf/cpp/src/groupby/hash/compute_groupby.cu#L46">compute_groupby.cu:46</a>

            [optional KERNEL 0] thrust::tabulate   // hash caching — CCCL/Thrust
              → row hash cache kernel

            compute_single_pass_aggs()             // <a href="../../cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh#L30">compute_single_pass_aggs.cuh:30</a>
              extract_single_pass_aggs()           // <a href="../../cudf/cpp/src/groupby/hash/extract_single_pass_aggs.cpp#L116">extract_single_pass_aggs.cpp:116</a>
                → SUM kept as-is; MEAN→SUM+COUNT
              is_shared_memory_compatible()        // queries shmem size

              [KERNEL 1] mapping_indices_kernel    // <a href="../../cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh#L120">compute_mapping_indices.cuh:120</a>
                find_local_mapping()               //   insert into block-local cuco set (shmem)
                  cuco::static_set_ref::insert_and_find()  // cuCollections
                  cuda::atomic_ref&lt;thread_scope_block&gt;::fetch_add()  // libcudacxx
                find_global_mapping()              //   unique keys → global cuco set
                  global_set_ref::insert_and_find()         // cuCollections

              [fallback check] cudaMemcpyAsync + stream.synchronize()
                → if needs_fallback → compute_global_memory_aggs()

              extract_populated_keys(global_set, ...)   // <a href="../../cudf/cpp/src/groupby/hash/output_utils.cu#L134">output_utils.cu:134</a>
                cuco::static_set::retrieve_all()         // <a href="../../cuCollections/include/cuco/detail/static_set/static_set.inl#L568">static_set.inl:568</a>
                  open_addressing_impl::retrieve_all()   // <a href="../../cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh#L902">open_addressing_impl.cuh:902</a>
                    cub::DeviceSelect::If(dry run)       //   measures scratch memory (no kernel)
                    cub::DeviceSelect::If(stream)        //   stream-compaction → uses thrust iteratore + CUb filter function to gather unique key indices (non-sentinel values in the hash slots)

              [KERNEL 2] single_pass_shmem_aggs_kernel  // <a href="../../cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu#L207">compute_shared_memory_aggs.cu:207</a>
                initialize_shmem_aggregations()    //   set accumulator slots to identity(SUM)=0
                compute_pre_aggregations()         //   reduce to shmem
                  shmem_element_aggregator::operator()
                    update_target_element&lt;T,SUM&gt;()     // <a href="../../cudf/cpp/include/cudf/detail/aggregation/device_aggregators.cuh#L115">device_aggregators.cuh:115</a>
                      cudf::detail::atomic_add()       // → CUDA atomicAdd into shared mem
                compute_final_aggregations()       //   flush shmem → global output col
                  gmem_element_aggregator::operator()
                    update_target_element&lt;T,SUM&gt;()     // <a href="../../cudf/cpp/include/cudf/detail/aggregation/device_aggregators.cuh#L115">device_aggregators.cuh:115</a>
                      cudf::detail::atomic_add()       // → CUDA atomicAdd into global col

            finalize_output()                      // <a href="../../cudf/cpp/src/groupby/hash/output_utils.cu">output_utils.cu</a>
              store results in result_cache

          [KERNEL 3] cudf::detail::gather()        // <a href="../../cudf/cpp/src/groupby/hash/compute_groupby.cu#L118">compute_groupby.cu:118</a>
            → Thrust-based gather of unique key rows

        extract_results(requests, cache, ...)      // <a href="../../cudf/cpp/src/groupby/hash/groupby.cu">hash/groupby.cu</a>
          → returns aggregation result columns from cache
</pre>

---

## 12. Key RMM Allocations Along the Path

| Variable | Type | Size | Where |
|----------|------|------|-------|
| `cuco::static_set` storage | `rmm::mr::polymorphic_allocator<char>` | `2 × 100M × 4 = 800 MB` | `compute_groupby.cu:128` |
| `cached_hashes` | `rmm::device_uvector<hash_value_type>` | `100M × 4 = 400 MB` (**not allocated** — `label` is 1 column ≤ `HASH_CACHING_THRESHOLD=4`) | `compute_groupby.cu:91` |
| `local_mapping_indices` | `rmm::device_uvector<size_type>` | `100M × 4 = 400 MB` | `compute_single_pass_aggs.cuh:105` |
| `global_mapping_indices` | `rmm::device_uvector<size_type>` | `781,250 blocks × 128 × 4 ≈ 400 MB` | `compute_single_pass_aggs.cuh:107` |
| `block_cardinality` | `rmm::device_uvector<size_type>` | `781,250 × 4 ≈ 3 MB` | `compute_single_pass_aggs.cuh:116` |
| `needs_global_memory_fallback` | `rmm::device_scalar<cuda::std::atomic_flag>` | `1` byte | `compute_single_pass_aggs.cuh:119` |
| `agg_results` output (`amount` SUM) | `unique_ptr<table>` of `cudf::column` | `K × 8` bytes (`int64_t`, K = distinct labels) | `compute_single_pass_aggs.cuh:171` |
| `d_agg_kinds` | `rmm::device_uvector<aggregation::Kind>` | `1 × sizeof(Kind)` bytes (1 agg column) | `compute_single_pass_aggs.cuh:68` |

All `rmm::device_uvector` and `rmm::device_scalar` allocations go through the stream-ordered pool specified by the `device_async_resource_ref mr` passed by the caller.  
See [rmm/cpp/include/rmm/device_uvector.hpp](../../rmm/cpp/include/rmm/device_uvector.hpp#L124).

---

## 13. CCCL / libcudacxx / cuCollections Components

### cuCollections (`cuco`) — part of CCCL

| API | Location | Usage |
|-----|----------|-------|
| `cuco::static_set` | `cuco/static_set.cuh` | Global hash set storing unique-key row indices |
| `cuco::static_set_ref` | `cuco/static_set_ref.cuh` | Block-scoped shared-memory hash set in `mapping_indices_kernel` |
| `cuco::linear_probing<1, row_hasher_with_cache_t>` | `cuco/probing_scheme.cuh` | Probing strategy for both global and shared sets |
| `cuco::insert_and_find` tag | — | Used on every row: insert-or-find, returns iterator + was_inserted |
| `cuco::retrieve_all` | — | Extracts populated indices (unique keys) after insertion |

### libcudacxx — part of CCCL

| API | Location | Usage |
|-----|----------|-------|
| `cuda::atomic_ref<size_type, thread_scope_block>` | `cuda/atomic` | Block-scoped atomic increment of `cardinality` counter in shmem kernel |
| `cuda::std::atomic_flag` | `cuda/std/atomic` | Device-side flag to signal global-memory fallback needed |
| `cuda::std::memory_order_relaxed` | `cuda/std/atomic` | Memory ordering for all groupby atomics |
| `cuda::thread_scope_device` | `cuda/atomic` | Scope for `cuco::static_set` |
| `cuda::thread_scope_block` | `cuda/atomic` | Scope for shared-memory `cuco::static_set_ref` |
| `cuda::std::byte` | `cuda/std/cstddef` | Byte pointer type for shared-memory aggregation buffers |

### Thrust — part of CCCL

| API | Usage |
|-----|-------|
| `thrust::tabulate(rmm::exec_policy_nosync(stream), ...)` | Hash caching; key-index computation in global path |
| `thrust::for_each_n(rmm::exec_policy_nosync(stream), ...)` | Global memory aggregation path; compound agg finalization |
| `thrust::uninitialized_fill` | Initialize `global_mapping_indices` to sentinel |
| `thrust::make_counting_iterator` | Input iterators for Thrust algorithms |

### CUB — part of CCCL

CUB appears extensively as the **underlying implementation** of Thrust and cuCollections operations. Despite not being called directly by cuDF groupby code, it generates the majority of the observed kernel launches:

| CUB API | Caller | Observed Kernel |
|---------|--------|----------------|
| `cub::DeviceFor::Bulk` | `cuco::storage_.initialize_async()` | `for_each::static_kernel<initialize_functor>` — slot init |
| `cub::DeviceFor::Bulk` | `thrust::uninitialized_fill` | `for_each::static_kernel<uninitialized_fill::functor>` — mapping init |
| `cub::DeviceSelect::If` | `cuco::static_set::retrieve_all()` | `DeviceCompactInitKernel` + `DeviceSelectSweepKernel<get_slot, slot_is_filled>` |
| `cub::DeviceTransform` | Thrust transform / permutation | `transform::transform_kernel<permutation_iterator>` |
| `cub::DeviceScan` | `cudf::strings::gather` | `DeviceScanInitKernel` + `DeviceScanKernel` (string offset prefix scan) |

---

## 14. `cuco::static_set` Internals

This section documents the internal design of `cuco::static_set` — the GPU-accelerated hash set at the heart of the groupby key deduplication step.

### Source file layout

`cuco::static_set` is a **header-only** library. There is no compiled `.cu` or `.cpp` — all code lives in `.cuh` and `.inl` files that are included transitively:

| File | Purpose |
|------|---------|
| [`cuco/static_set.cuh`](../../cuCollections/include/cuco/static_set.cuh) | Public API — template class declaration and type aliases |
| [`cuco/detail/static_set/static_set.inl`](../../cuCollections/include/cuco/detail/static_set/static_set.inl) | Host member function bodies (constructor, `insert`, `retrieve_all`, …) — included at the bottom of `static_set.cuh` |
| [`cuco/detail/open_addressing/open_addressing_impl.cuh`](../../cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh) | Shared host-side implementation for all open-addressing containers (`static_set`, `static_map`, `static_multiset`) — kernel launches live here |
| [`cuco/detail/open_addressing/kernels.cuh`](../../cuCollections/include/cuco/detail/open_addressing/kernels.cuh) | The actual GPU kernels: `insert_if_n`, `insert_and_find`, `contains_if_n`, `find_if_n`, `count`, … |
| [`cuco/static_set_ref.cuh`](../../cuCollections/include/cuco/static_set_ref.cuh) | Device-side non-owning reference — trivially copyable, passed by value into GPU kernels |
| [`cuco/detail/static_set/static_set_ref.inl`](../../cuCollections/include/cuco/detail/static_set/static_set_ref.inl) | Member function bodies for `static_set_ref` |

> **Why `.inl`?** An `.inl` ("inline") file is just a plain C++ file meant to be `#include`d rather than compiled directly. It keeps the header readable while still providing full template definitions to every translation unit that includes it.

### Host/device split: `static_set` vs. `static_set_ref`

GPU kernels receive their arguments **by value** (copied into registers). This means a kernel cannot receive `static_set` directly — it owns RAII resources (device memory, allocators) and is not trivially copyable.

The solution is a two-tier design:

```
static_set          — host-side owner
  owns: device slot array (via RMM allocator)
  owns: allocator, probing scheme, sentinels
  NOT passed to kernels

static_set_ref      — device-side non-owning view
  contains: raw pointer to slot array
  contains: hash function, key comparator, empty sentinel
  trivially copyable → safe to pass by value to <<<kernel>>>
```

A ref is created from the host side with an operator tag:
```cpp
auto r = set.ref(cuco::op::insert_and_find);
my_kernel<<<grid, block>>>(r, ...);
```

### Open addressing with linear probing

cuDF's groupby configures `cuco::static_set` with `cuco::linear_probing<1, row_hasher_with_cache_t>`
([`probing_scheme.cuh`](../../cuCollections/include/cuco/probing_scheme.cuh)):

```cpp
// helpers.cuh
using probing_scheme_t = cuco::linear_probing<GROUPBY_CG_SIZE, row_hasher_with_cache_t>;
//                                             ^1               ^ MurmurHash3 row hasher
```

**How open addressing works:**

The hash table is a flat array of slots, each holding either a key or an empty sentinel (`INT32_MAX`). There are no linked lists or separate buffers — all keys live in the same contiguous array.

For each incoming row index `i`:

1. Compute the hash of row `i`: $h = H(\text{row}_i) \bmod \text{capacity}$
2. Start at slot $h$. If that slot holds the empty sentinel → insert `i` there (new group).
3. If that slot already holds some key `j` → compare row `i` to row `j` using `d_row_equal`.
   - Equal → same group, return existing slot.
   - Not equal → **linear probe**: move to slot $h+1$, $h+2$, … (wrapping around) until an empty or matching slot is found.

$$\text{slot}_k = (H(\text{row}_i) + k) \bmod \text{capacity}$$

This is **insert-or-find**: a single operation that either inserts the key (new group) or returns the existing slot (known group). The result is an iterator pointing to the slot, plus a boolean indicating whether insertion occurred — exactly what `insert_and_find` returns.

**Why linear probing here?**

cuDF uses `CGSize = 1` (one thread per key, `GROUPBY_CG_SIZE = 1`). With a single thread there is no benefit to cooperative slot scanning. Linear probing with a high-quality hash function (`MurmurHash3_x86_32` via `row_hasher_with_cache_t`) keeps the implementation simple while avoiding clustering in practice.

The table is sized at **2× the number of input rows** (50% load factor via `CUCO_DESIRED_LOAD_FACTOR = 0.5`) to keep probe chains short.

**Single-thread path in the kernel:**

In [`kernels.cuh`](../../cuCollections/include/cuco/detail/open_addressing/kernels.cuh), when `CGSize == 1` the kernel takes this branch:

```cpp
if constexpr (CGSize == 1) {
    auto const [iter, inserted] = ref.insert_and_find(insert_element);
    // Write to shared memory first to avoid L1 flushing extra L2→global traffic
    output_location_buffer[thread_idx] = output(iter);
    output_inserted_buffer[thread_idx] = inserted;
    block.sync();
    *(found_begin + idx)    = output_location_buffer[thread_idx];
    *(inserted_begin + idx) = output_inserted_buffer[thread_idx];
}
```

The shared-memory buffering before the global write avoids a known issue where `ld.relaxed.gpu` causes excess sector stores from L2 to global memory.

### `insert_and_find` — lock-free insert-or-find with CAS

The actual `insert_and_find` logic lives in [`open_addressing_ref_impl.cuh`](../../cuCollections/include/cuco/detail/open_addressing/open_addressing_ref_impl.cuh). In plain terms:

> For a given key: if it is already in the table, return a pointer to it. If it is not, insert it and return a pointer to the new slot. Either way, also return whether the insertion happened. All of this is done without any locks.

**Step by step (single-thread path, `CGSize == 1`):**

1. **Hash the key** → compute the starting slot index `h = H(key) % capacity`.

2. **Walk the probe chain** — starting at `h`, inspect each slot:
   - `EQUAL` → key already exists here. Return `{pointer_to_slot, false}` immediately.
   - `AVAILABLE` (empty sentinel) → this slot is a candidate for insertion. Attempt a CAS.
   - `UNEQUAL` → slot is occupied by a different key. Move to slot `h+1`, `h+2`, … (wrap around).

3. **Compare-And-Swap (CAS) on the empty slot:**
   ```
   atomically: if slot still == empty_sentinel  →  write new key, return SUCCESS
               if slot now  == our key          →  another thread inserted same key, return DUPLICATE
               if slot now  == different key     →  another thread stole this slot,   return CONTINUE
   ```
   - `SUCCESS` → we won the race. Return `{pointer_to_slot, true}`.
   - `DUPLICATE` → a concurrent thread inserted the same key just before us. Return `{pointer_to_slot, false}`.
   - `CONTINUE` → the slot was stolen by a different key. Resume probing from the next slot.

4. **Full-table check:** if the probe chain wraps all the way back to `h` without finding a match or empty slot, the table is full — return `{end(), false}`.

**Why CAS and not a mutex?**

On the GPU there are potentially millions of threads all inserting simultaneously. A mutex would serialize them entirely. CAS is a single atomic hardware instruction that either succeeds or fails instantly — no thread ever blocks waiting for another. The only cost of a collision is one extra probe step.

**The `group.ballot()` / `__ffs` / `group.shfl()` pattern (CG path) — avoiding divergence:**

When `CGSize > 1`, each thread in the tile independently checks one slot and arrives at a local result (`EQUAL`, `AVAILABLE`, or `UNEQUAL`). The challenge is: how do all threads agree on what to do next without branching differently from each other (which would cause warp divergence)?

The answer is three warp-level intrinsics that operate on **all threads simultaneously** — no loops, no shared memory, no `if` branching across lanes:

```
group.ballot(condition)
```
Every thread votes `true` or `false` on `condition` in a single hardware instruction. The result is a bitmask where bit `i` is set if lane `i` voted `true`. All threads receive the same bitmask — so they all see the same picture of the tile's state without any of them branching differently.

```
__ffs(mask)   // "find first set bit"
```
`__ffs(group_finds_equal)` returns the index of the lowest lane that voted `true` — identifying the "winning" lane (e.g., the first thread that found a match or an empty slot). This is a single hardware instruction, not a loop.

```
group.shfl(value, src_lane)
```
Broadcasts the `value` held by `src_lane` to all other threads in the tile in one instruction — no shared memory write needed. Used here to share the winning thread's slot pointer with all other threads so they can all return the same iterator.

**Putting it together — concrete example (key already exists):**

```
Thread 0: checks slot h+0 → UNEQUAL
Thread 1: checks slot h+1 → EQUAL   ← found it
Thread 2: checks slot h+2 → UNEQUAL
Thread 3: checks slot h+3 → UNEQUAL

group.ballot(state == EQUAL)  →  bitmask = 0b0010  (bit 1 set)
__ffs(0b0010) - 1             →  src_lane = 1
group.shfl(slot_ptr, 1)       →  all threads now hold thread 1's slot pointer
group.sync()                  →  all threads wait for thread 1 to finish
return {iterator, false}       ← all 4 threads return the same value, no divergence
```

The key insight: **all threads in the tile always take the same branch** because `ballot` + `shfl` turn "which thread found something?" from a per-thread question into a shared bitmask that every thread reads identically. Divergence only happens when threads in a warp execute different instructions — here they always execute the same ones.

### CRTP operator mixin design

`static_set_ref` uses the **Curiously Recurring Template Pattern (CRTP)** to selectively add methods at compile time via the `Operators...` variadic template parameter.

**The mechanism:**

```cpp
// Each operator is an empty tag type (operator.hpp)
struct insert_and_find_tag {} inline constexpr insert_and_find;
struct contains_tag        {} inline constexpr contains;

// static_set_ref inherits from one mixin per operator in the pack
class static_set_ref
  : public detail::operator_impl<Operators,
                                 static_set_ref<Key,...,Operators...>>...
//                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//     pack expansion: one base class per tag in Operators...
```

Each `operator_impl<Tag, Derived>` specialization contributes exactly one device method:
```cpp
template <typename Derived>
struct operator_impl<insert_and_find_tag, Derived> {
    __device__ auto insert_and_find(auto key) {
        return static_cast<Derived*>(this)->impl_.insert_and_find(key);
    }
};
```

**Why no virtual functions?** Virtual dispatch requires a vtable pointer — an extra global memory load on the GPU. Different threads following different virtual call paths also cause warp divergence (serial execution). CRTP resolves everything at compile time with zero runtime overhead.

**Practical effect:** A ref with only `op::contains` cannot call `.insert()` — it is a compile error. This makes illegal operations impossible rather than silently corrupting data at runtime.

```cpp
auto r_ro = set.ref(op::contains);          // read-only ref — no insert method
auto r_rw = set.ref(op::insert, op::find);  // read-write ref — both methods present
auto r_ia = set.ref(op::insert_and_find);   // insert-or-find ref — what cuDF uses
```

### `.inl` file convention

An `.inl` file is simply a C++ source file that is `#include`d by its parent header rather than compiled independently. The convention signals to readers: *"do not compile this file directly — it is part of `static_set.cuh`"*. The compiler sees one unified translation unit. The same convention is used with `.tpp` ("template implementation") and `.ipp` ("inline implementation") in other libraries.
