
# cuDF `groupby` + `SUM` — Part II: Nsight Analysis

> **Part of a three-document series:**
> - [Part I — Algorithm Overview](groupby_sum_1_algorithm_overview.md): high-level description of the hash groupby algorithm, data structures, the two-kernel aggregation strategy, and the life of `global_mapping_indices`.
> - **Part II — Nsight Analysis** *(this file)*: ground-truth kernel table and performance breakdown from an actual Nsight Systems capture on 100M rows.
> - [Part III — Code Analysis](groupby_sum_3_code_analysis.md): function-by-function walk-through of the cuDF, cuCollections, RMM, and CCCL source, with annotated call stack and library layer summary.

This document validates the algorithm described in Part I against a real GPU trace. Every kernel listed here was observed in Nsight Systems during a single `groupby` + `SUM(amount)` call on the 100M-row `label`/`amount` dataset. Durations are exact values from the capture; no rounding.

---

## Table of Contents

- [1. CUDA Kernels Launched (Nsight Ground-Truth)](#1-cuda-kernels-launched-nsight-systems-ground-truth-sum-on-string-key-table)
- [2. Nsight Systems Observed Kernel Trace (Ground Truth)](#2-nsight-systems-observed-kernel-trace-ground-truth)
  - [2a. Raw Kernel Sequence](#2a-raw-kernel-sequence)
  - [2b. Performance Breakdown](#2b-performance-breakdown-exact-kernel-times-from-nsight)

---

## 1. CUDA Kernels Launched (Nsight Systems Ground-Truth, SUM on string-key table)

> **Source of truth**: Nsight Systems capture of the hash groupby phase.  
> Numbers reflect a single `utf8` string key column (`label`), one `int64` value column (`amount`), 100M rows, shared-memory sub-path.

| # | Kernel (short name) | Full Nsight Name (abbreviated) | Duration | cuDF/cuCo source | Purpose |
|---|---------------------|-------------------------------|----------|-----------------|---------|
| A | `for_each` / init | `cub::detail::for_each::static_kernel<initialize_functor<long,int>>` | **4.105 ms** | [storage/functors.cuh](cuCollections/include/cuco/detail/storage/functors.cuh#L54) | `cuco::static_set` ctor → `clear_async()` → `storage_.initialize_async()` → CUB DeviceFor writes empty-slot sentinel to all 2×N slots. |
| B | `for_each` / fill | `cub::detail::for_each::static_kernel<__uninitialized_fill::functor<int*,int>>` | **3.840 μs** | [compute_single_pass_aggs.cuh](cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh#L111) | `thrust::uninitialized_fill` on `global_mapping_indices`. CUB `DeviceFor` is the Thrust backend. |
| 1 | `mapping_indices_kernel` | `cudf::groupby::detail::hash::mapping_indices_kernel<...>` | **4.784 ms** | [compute_mapping_indices.cuh](cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh#L120) | Insert every row into global `cuco::static_set`; record local- and global-mapping-indices. |
| D | `DeviceCompactInitKernel` | `cub::detail::scan::DeviceCompactInitKernel<ScanTileState<int,true>, long*>` | **2.848 μs** | [open_addressing_impl.cuh](cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh#L932) | `extract_populated_keys()` → `retrieve_all()` → `cub::DeviceSelect::If` pass 1: init per-tile prefix-sum scratch. |
| E | `DeviceSelectSweepKernel` | `cub::detail::select::DeviceSelectSweepKernel<transform_iterator<get_slot,...>, slot_is_filled<false,int>>` | **3.439 ms** | [open_addressing_impl.cuh](cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh#L945) | Stream compaction: copies every filled slot's key to the output buffer (unique key row-indices). |
| F | `transform_kernel` (perm) | `cub::detail::transform::transform_kernel<permutation_iterator<int*,const int*>>` | **5.568 μs** | [output_utils.cu](cudf/cpp/src/groupby/hash/output_utils.cu#L158) | `compute_key_transform_map()` — remap unique key row-indices via a permutation. |
| G | `for_each` / remap | `cub::detail::for_each::static_kernel<op_wrapper_t<compute_single_pass_aggs::[lambda]>>` | **4.960 μs** | [compute_single_pass_aggs.cuh](cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh#L154) | `thrust::for_each_n` — applies `key_transform_map` to renumber `global_mapping_indices` to dense [0, num_unique_keys). |
| H | `transform_kernel` (fill) | `cub::detail::transform::transform_kernel<__return_constant<double>, double*>` | **1.312 μs** | [output_utils.cu](cudf/cpp/src/groupby/hash/output_utils.cu#L132) | `thrust::fill` / output offset initialisation for output column allocation. |
| 2 | `single_pass_shmem_aggs_kernel` | `cudf::groupby::detail::hash::<unnamed>::single_pass_shmem_aggs_kernel(...)` | **5.119 ms** | [compute_shared_memory_aggs.cu](cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu#L207) | Two-phase SUM accumulation: Phase 1 row→shmem accumulator, Phase 2 shmem→global output (atomic add). |
| I | `DeviceScanInitKernel` | `cub::detail::scan::DeviceScanInitKernel<ScanTileState<long,true>>` | **1.376 μs** | [strings/copying/gather.cu](cudf/cpp/src/strings/copying/gather.cu) | `cudf::strings::gather` — prefix-scan init over string character offsets. |
| J | `DeviceScanKernel` | `cub::detail::scan::DeviceScanKernel<Policy1000, transform_iterator<string_offsets_fn,...>>` | **10.464 μs** | [strings/copying/gather.cu](cudf/cpp/src/strings/copying/gather.cu) | Inclusive prefix scan over string character offsets → output offset buffer. |
| K | `gather_chars_fn_char_parallel` | `cudf::strings::detail::gather_chars_fn_char_parallel<32, transform_iterator<value_accessor,...>>` | **4.928 μs** | [strings/copying/gather.cu](cudf/cpp/src/strings/copying/gather.cu) | Copy character data for the string key column into the output gathered key table. |
| L | `valid_if_n_kernel` | `cudf::detail::valid_if_n_kernel<counting_iter, counting_iter, gather_bitmask_functor<INCLUDE,...>, 256>` | **2.592 μs** | [valid_if.cuh](cudf/cpp/include/cudf/detail/valid_if.cuh) | Build null mask for the gathered string key output column. |

> **Total dominating cost**: Kernel 2 (aggregate, 5.12 ms) + Kernel 1 (insert, 4.78 ms) + Kernel A (cuco init, 4.11 ms) + Kernel E (retrieve_all, 3.44 ms) ≈ **17.5 ms** total.  
> `DeviceSelectSweepKernel` (E) is now the **4th most expensive** kernel — behind `single_pass_shmem_aggs_kernel` (2), `mapping_indices_kernel` (1), and hash table init (A).


---

## 2. Nsight Systems Observed Kernel Trace (Ground Truth)

> Captured with Nsight Systems on a single groupby+SUM operation.  
> Key: `label` (1 `utf8` string column, col 2). Value: `amount` (1 `int64` column, col 4). Input: 100M rows. Path: shared-memory sub-path (cardinality ≤ 128).

### 2a. Raw Kernel Sequence

> Durations are exact values from the Nsight Systems capture.  
> `cudaMemset` (needs_global_memory_fallback) and `cudaMemcpy DtoH` (reads of needs_global_memory_fallback and h_num_out) do appear in the trace but as **memory operations**, not as kernel rows — they are omitted from this kernel-only list.

```
 #   Duration    Kernel Name (abbreviated)
──────────────────────────────────────────────────────────────────────────────
 A   4.105 ms    cub::detail::for_each::static_kernel
                   <policy_500_t, long, cuco::detail::initialize_functor<long, int>>
                 ← cuco::static_set ctor → clear_async() → storage_.initialize_async()
                 ← Source: cuCollections/include/cuco/detail/storage/functors.cuh:54

 B   3.840 μs    cub::detail::for_each::static_kernel
                   <policy_500_t, long, thrust::cuda_cub::__uninitialized_fill::functor<int*, int>>
                 ← thrust::uninitialized_fill on global_mapping_indices
                 ← Source: cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh:111

 1   4.784 ms    cudf::groupby::detail::hash::mapping_indices_kernel<
                   cuco::static_set_ref<int, thread_scope_block,
                   device_row_comparator, linear_probing<1, row_hasher_with_cache_t>,
                   bucket_storage_ref, insert_and_find_tag>>
                 ← Per-row insert into global cuco::static_set + record mapping indices
                 ← Source: cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh:120

 D   2.848 μs    cub::detail::scan::DeviceCompactInitKernel<ScanTileState<int,true>, long*>
                 ← cuco::static_set::retrieve_all() → cub::DeviceSelect::If (pass 1: scratch init)
                 ← Source: cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh:932

 E   3.439 ms    cub::detail::select::DeviceSelectSweepKernel<
                   Policy1000,
                   thrust::transform_iterator<get_slot<false, bucket_storage_ref>, counting_iter>,
                   NullType*, int*, long*, ScanTileState<int,true>,
                   slot_is_filled<false, int>, ...>
                 ← cub::DeviceSelect::If (pass 2: stream compaction of all filled slots)
                 ← Source: cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh:945
                   Functors: cuCollections/include/cuco/detail/open_addressing/functors.cuh

 F   5.568 μs    cub::detail::transform::transform_kernel<
                   policy1000, int, always_true_predicate, identity,
                   thrust::permutation_iterator<int*, const int*>,
                   thrust::counting_iterator<int>>
                 ← compute_key_transform_map() remaps unique key row-indices
                 ← Source: cudf/cpp/src/groupby/hash/output_utils.cu:158

 G   4.960 μs    cub::detail::for_each::static_kernel<
                   policy_500_t, int,
                   op_wrapper_t<int, compute_single_pass_aggs<...>::[lambda]>>
                 ← thrust::for_each_n updates global_mapping_indices via key_transform_map
                 ← Source: cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh:154

 H   1.312 μs    cub::detail::transform::transform_kernel<
                   policy1000, int, always_true_predicate,
                   thrust::cuda_cub::__return_constant<double>, double*>
                 ← thrust::fill / output offset initialisation
                 ← Source: cudf/cpp/src/groupby/hash/output_utils.cu:132

 2   5.119 ms    cudf::groupby::detail::hash::<unnamed>::single_pass_shmem_aggs_kernel(
                   int, const uint*, int*, int*, int*,
                   table_device_view, mutable_table_device_view,
                   const aggregation::Kind*, int, int)
                 ← Two-phase SUM: Phase 1 row→shmem, Phase 2 shmem→global (atomicAdd)
                 ← Source: cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu:207

 I   1.376 μs    cub::detail::scan::DeviceScanInitKernel<ScanTileState<long,true>>
                 ← cudf::strings::gather → prefix-scan on string offsets (init pass)
                 ← Source: cudf/cpp/src/strings/copying/gather.cu

 J   10.464 μs   cub::detail::scan::DeviceScanKernel<
                   Policy1000,
                   thrust::transform_iterator<string_offsets_fn<...>, counting_iter>,
                   sizes_to_offsets_iterator<int*, long>,
                   ScanTileState<long,true>, plus<void>, ...>
                 ← Inclusive prefix-sum of string character offsets → output offset buffer

 K   4.928 μs    cudf::strings::detail::gather_chars_fn_char_parallel<32,
                   thrust::transform_iterator<value_accessor<string_view>, counting_iter>,
                   input_indexalator>
                 ← Copy char data for string key column into gathered output table

 L   2.592 μs    cudf::detail::valid_if_n_kernel<counting_iter, counting_iter,
                   gather_bitmask_functor<INCLUDE, input_indexalator>, 256>
                 ← Build null mask for gathered string key output column
```

### 2b. Performance Breakdown (exact kernel times from Nsight)

| Phase | Kernels | Kernel Time |
|-------|---------|------------|
| Hash table init (`cuco::static_set` ctor → `clear_async`) | A | **4.105 ms** |
| `global_mapping_indices` init (`thrust::uninitialized_fill`) | B | 3.840 μs |
| Insert all rows (`mapping_indices_kernel`) | 1 | **4.784 ms** |
| Extract unique keys (`retrieve_all` → `cub::DeviceSelect::If`) | D, E | **3.442 ms** |
| Remap indices (transform + for_each + fill) | F, G, H | 11.8 μs |
| SUM aggregation (`single_pass_shmem_aggs_kernel`) | 2 | **5.119 ms** |
| String key gather (scan + chars + null mask) | I, J, K, L | 19.4 μs |
| **Total GPU kernel time** | | **≈ 17.5 ms** |

> **Note**: `single_pass_shmem_aggs_kernel` (2, 5.119 ms) is now the most expensive kernel, followed by `mapping_indices_kernel` (1, 4.784 ms), hash table init (A, 4.105 ms), and `DeviceSelectSweepKernel` (E, 3.439 ms). Kernel 1 improved by ~2.2 ms vs the previous run. The unique-key extraction step accounts for ~20% of total kernel time.


