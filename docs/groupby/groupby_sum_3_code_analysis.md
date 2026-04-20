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
  - [6b. `cuco::static_set` construction — the central hash table](#6b-cucostaticset-construction--the-central-hash-table)
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

---

## 1. Public Entry Point: `groupby::groupby()` Constructor

**File**: [cudf/cpp/src/groupby/groupby.cu](cudf/cpp/src/groupby/groupby.cu#L40)

```
groupby::groupby(table_view const& keys, null_policy, sorted, ...)
```

- Stores `_keys`, `_include_null_keys`, `_keys_are_sorted`, `_column_order`, `_null_precedence`.
- No GPU work yet. The `table_view` is a lightweight non-owning reference to the device memory columns.

---

## 2. `groupby::aggregate()`

**File**: [cudf/cpp/src/groupby/groupby.cu](cudf/cpp/src/groupby/groupby.cu#L238)

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

**File**: [cudf/cpp/src/groupby/groupby.cu](cudf/cpp/src/groupby/groupby.cu#L60)

```cpp
if (_keys_are_sorted == sorted::NO and not _helper
    and detail::hash::can_use_hash_groupby(requests))
    → detail::hash::groupby(...)   // ← taken for SUM on fixed-width types
else
    → sort_aggregate(...)
```

### `can_use_hash_groupby()` check

**File**: [cudf/cpp/src/groupby/hash/groupby.cu](cudf/cpp/src/groupby/hash/groupby.cu#L165)

The set `hash_aggregations` includes: `SUM, SUM_WITH_OVERFLOW, SUM_OF_SQUARES, PRODUCT, MIN, MAX, COUNT_VALID, COUNT_ALL, ARGMIN, ARGMAX, MEAN, M2, STD, VARIANCE`.

For `SUM` on a numeric (fixed-width, atomic-capable) type → returns `true` → **hash path taken**.

---

## 4. `detail::hash::groupby()`

**File**: [cudf/cpp/src/groupby/hash/groupby.cu](cudf/cpp/src/groupby/hash/groupby.cu#L192)

```cpp
std::pair<unique_ptr<table>, vector<aggregation_result>>
    groupby(table_view const& keys, requests, include_null_keys, stream, mr)
```

1. Creates a `cudf::detail::result_cache` (host-side map from `(column_ptr, agg_kind)` → output column).
2. Calls `dispatch_groupby()` which returns `unique_keys` table.
3. Calls `extract_results(requests, cache, stream, mr)` to finalize results from cache.

---

## 5. `dispatch_groupby()` — Row Comparator & Hasher Setup

**File**: [cudf/cpp/src/groupby/hash/groupby.cu](cudf/cpp/src/groupby/hash/groupby.cu#L83)

```cpp
auto preprocessed_keys = cudf::detail::row::hash::preprocessed_table::create(keys, stream);
auto comparator        = cudf::detail::row::equality::self_comparator{preprocessed_keys};
auto row_hash          = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
auto d_row_hash        = row_hash.device_hasher(nullate::DYNAMIC{...});
```

- **`preprocessed_table`**: Builds a column-order-aware view of the keys table, used by both the hasher and comparator. Resolves nested structs/lists.
- **`row_hasher`** / **`d_row_hash`**: Wraps `cudf::hashing::detail::default_hash` (MurmurHash3_x86_32) into a device callable that hashes an entire row by index.
- **`self_comparator`** / **`d_row_equal`**: Device callable that compares two row indices for equality, handling nulls according to `null_keys_are_equal = EQUAL`.

> **libcudacxx / CCCL** connection: `d_row_hash` is wrapped in `row_hasher_with_cache_t` (defined in [helpers.cuh](cudf/cpp/src/groupby/hash/helpers.cuh#L56)), which is used as the hash function for the `cuco::linear_probing` scheme.

Dispatches to `compute_groupby<row_comparator_t>()` (no nested columns) or `compute_groupby<nullable_row_comparator_t>()`.

---

## 6. `compute_groupby()` — The `cuco::static_set` and Main Orchestration

**File**: [cudf/cpp/src/groupby/hash/compute_groupby.cu](cudf/cpp/src/groupby/hash/compute_groupby.cu#L46)

### 6a. Hash caching (optional Thrust kernel)

> **Not taken for this dataset.** The `label` key is a single non-nested `utf8` column, so `count_nested_columns` returns 1. Since `1 ≤ HASH_CACHING_THRESHOLD=4`, `cached_hashes` is returned as an empty zero-size `device_uvector` and no kernel fires. The `row_hasher_with_cache_t` wrapper receives a null pointer for the cache and falls back to recomputing the hash on every probe.

If the keys table has more than `HASH_CACHING_THRESHOLD = 4` total columns (nested included), each row's hash is pre-computed once and cached to avoid re-hashing during the two passes inside `mapping_indices_kernel`:

```cpp
// CUDA kernel launched via Thrust (CCCL)
rmm::device_uvector<hash_value_type> hashes(num_keys, stream);   // RMM allocation
thrust::tabulate(rmm::exec_policy_nosync(stream), hashes.begin(), hashes.end(),
    [d_row_hash, row_bitmask] __device__(size_type idx) { return d_row_hash(idx); });
```

> **RMM**: `rmm::device_uvector<hash_value_type>` allocates device memory via the stream-ordered allocator pool from [rmm/cpp/include/rmm/device_uvector.hpp](rmm/cpp/include/rmm/device_uvector.hpp#L124).
> **CCCL/Thrust**: `thrust::tabulate` → maps to a single CUDA kernel launch.

### 6b. `cuco::static_set` construction — **the central hash table**

**File**: [cudf/cpp/src/groupby/hash/compute_groupby.cu](cudf/cpp/src/groupby/hash/compute_groupby.cu#L128)  
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

**Data structure details** ([helpers.cuh](cudf/cpp/src/groupby/hash/helpers.cuh)):
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

**File**: [cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh](cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh#L30)

### 7a. Pre-processing

1. `extract_single_pass_aggs(requests, stream)` ([extract_single_pass_aggs.cpp](cudf/cpp/src/groupby/hash/extract_single_pass_aggs.cpp#L116)):
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

**File**: [cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh](cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh#L120)  
**Host launcher**: [cudf/cpp/src/groupby/hash/compute_mapping_indices.cu](cudf/cpp/src/groupby/hash/compute_mapping_indices.cu)

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

**File**: [cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu](cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu#L207)

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

**File**: [cudf/cpp/src/groupby/hash/compute_global_memory_aggs.cuh](cudf/cpp/src/groupby/hash/compute_global_memory_aggs.cuh#L162)

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

**File**: [cudf/cpp/include/cudf/detail/aggregation/device_aggregators.cuh](cudf/cpp/include/cudf/detail/aggregation/device_aggregators.cuh#L115)

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

**File**: [cudf/cpp/src/groupby/hash/extract_single_pass_aggs.cpp](cudf/cpp/src/groupby/hash/extract_single_pass_aggs.cpp)  
**File**: [cudf/cpp/src/groupby/hash/output_utils.cu](cudf/cpp/src/groupby/hash/output_utils.cu)

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

```
[user code]
  cudf::groupby::groupby(table_view{{key_col}})   // groupby.cu:40  — constructor, no GPU
  .aggregate(requests, stream, mr)                 // groupby.cu:223

    groupby::dispatch_aggregation()                // groupby.cu:53
      detail::hash::can_use_hash_groupby()         // hash/groupby.cu:165  → true for SUM

      detail::hash::groupby()                      // hash/groupby.cu:192
        dispatch_groupby()                         // hash/groupby.cu:83
          row::hash::preprocessed_table::create()  // sets up row hash/compare ops
          cuco::static_set{...}                    // (inside compute_groupby.cu:128)
          compute_groupby<row_comparator_t>()      // compute_groupby.cu:46

            [optional KERNEL 0] thrust::tabulate   // hash caching — CCCL/Thrust
              → row hash cache kernel

            compute_single_pass_aggs()             // compute_single_pass_aggs.cuh:30
              extract_single_pass_aggs()           // extract_single_pass_aggs.cpp:116
                → SUM kept as-is; MEAN→SUM+COUNT
              is_shared_memory_compatible()        // queries shmem size

              [KERNEL 1] mapping_indices_kernel    // compute_mapping_indices.cuh:120
                find_local_mapping()               //   insert into block-local cuco set (shmem)
                  cuco::static_set_ref::insert_and_find()  // cuCollections
                  cuda::atomic_ref<thread_scope_block>::fetch_add()  // libcudacxx
                find_global_mapping()              //   unique keys → global cuco set
                  global_set_ref::insert_and_find()         // cuCollections

              [fallback check] cudaMemcpyAsync + stream.synchronize()
                → if needs_fallback → compute_global_memory_aggs()

              [KERNEL 2] single_pass_shmem_aggs_kernel  // compute_shared_memory_aggs.cu:207
                initialize_shmem_aggregations()    //   set accumulator slots to identity(SUM)=0
                compute_pre_aggregations()         //   reduce to shmem
                  shmem_element_aggregator::operator()
                    update_target_element<T,SUM>()     // device_aggregators.cuh:115
                      cudf::detail::atomic_add()       // → CUDA atomicAdd into shared mem
                compute_final_aggregations()       //   flush shmem → global output col
                  gmem_element_aggregator::operator()
                    update_target_element<T,SUM>()     // device_aggregators.cuh:115
                      cudf::detail::atomic_add()       // → CUDA atomicAdd into global col

            finalize_output()                      // output_utils.cu
              store results in result_cache

          [KERNEL 3] cudf::detail::gather()        // compute_groupby.cu:118
            → Thrust-based gather of unique key rows

        extract_results(requests, cache, ...)      // hash/groupby.cu
          → returns aggregation result columns from cache
```

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
See [rmm/cpp/include/rmm/device_uvector.hpp](rmm/cpp/include/rmm/device_uvector.hpp#L124).

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
