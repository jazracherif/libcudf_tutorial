
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
| A | `for_each` / init | `cub::detail::for_each::static_kernel<initialize_functor<long,int>>` | **4.105 ms** | [storage/functors.cuh](../../cuCollections/include/cuco/detail/storage/functors.cuh#L54) | `cuco::static_set` ctor → `clear_async()` → `storage_.initialize_async()` → CUB DeviceFor writes empty-slot sentinel to all 2×N slots. |
| B | `for_each` / fill | `cub::detail::for_each::static_kernel<__uninitialized_fill::functor<int*,int>>` | **3.840 μs** | [compute_single_pass_aggs.cuh](../../cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh#L111) | `thrust::uninitialized_fill` on `global_mapping_indices`. CUB `DeviceFor` is the Thrust backend. |
| 1 | `mapping_indices_kernel` | `cudf::groupby::detail::hash::mapping_indices_kernel<...>` | **4.784 ms** | [compute_mapping_indices.cuh](../../cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh#L120) | Insert every row into global `cuco::static_set`; record local- and global-mapping-indices. |
| D | `DeviceCompactInitKernel` | `cub::detail::scan::DeviceCompactInitKernel<ScanTileState<int,true>, long*>` | **2.848 μs** | [open_addressing_impl.cuh](../../cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh#L944) | `extract_populated_keys()` → `retrieve_all()` → `cub::DeviceSelect::If` pass 2: init per-tile prefix-sum scratch. |
| E | `DeviceSelectSweepKernel` | `cub::detail::select::DeviceSelectSweepKernel<transform_iterator<get_slot,...>, slot_is_filled<false,int>>` | **3.439 ms** | [open_addressing_impl.cuh](../../cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh#L944) | Stream compaction: copies every filled slot's key to the output buffer (unique key row-indices). |
| F | `transform_kernel` (perm) | `cub::detail::transform::transform_kernel<permutation_iterator<int*,const int*>>` | **5.568 μs** | [output_utils.cu](../../cudf/cpp/src/groupby/hash/output_utils.cu#L158) | `compute_key_transform_map()` — remap unique key row-indices via a permutation. |
| G | `for_each` / remap | `cub::detail::for_each::static_kernel<op_wrapper_t<compute_single_pass_aggs::[lambda]>>` | **4.960 μs** | [compute_single_pass_aggs.cuh](../../cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh#L154) | `thrust::for_each_n` — applies `key_transform_map` to renumber `global_mapping_indices` to dense [0, num_unique_keys). |
| H | `transform_kernel` (fill) | `cub::detail::transform::transform_kernel<__return_constant<double>, double*>` | **1.312 μs** | [output_utils.cu](../../cudf/cpp/src/groupby/hash/output_utils.cu#L132) | `thrust::fill` / output offset initialisation for output column allocation. |
| 2 | `single_pass_shmem_aggs_kernel` | `cudf::groupby::detail::hash::<unnamed>::single_pass_shmem_aggs_kernel(...)` | **5.119 ms** | [compute_shared_memory_aggs.cu](../../cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu#L207) | Two-phase SUM accumulation: Phase 1 row→shmem accumulator, Phase 2 shmem→global output (atomic add). |
| I | `DeviceScanInitKernel` | `cub::detail::scan::DeviceScanInitKernel<ScanTileState<long,true>>` | **1.376 μs** | [strings/detail/gather.cuh](../../cudf/cpp/include/cudf/strings/detail/gather.cuh#L246) | `cudf::strings::gather` — prefix-scan init over string character offsets. |
| J | `DeviceScanKernel` | `cub::detail::scan::DeviceScanKernel<Policy1000, transform_iterator<string_offsets_fn,...>>` | **10.464 μs** | [strings/detail/gather.cuh](../../cudf/cpp/include/cudf/strings/detail/gather.cuh#L246) | Inclusive prefix scan over string character offsets → output offset buffer. |
| K | `gather_chars_fn_char_parallel` | `cudf::strings::detail::gather_chars_fn_char_parallel<32, transform_iterator<value_accessor,...>>` | **4.928 μs** | [strings/detail/gather.cuh](../../cudf/cpp/include/cudf/strings/detail/gather.cuh#L156) | Copy character data for the string key column into the output gathered key table. |
| L | `valid_if_n_kernel` | `cudf::detail::valid_if_n_kernel<counting_iter, counting_iter, gather_bitmask_functor<INCLUDE,...>, 256>` | **2.592 μs** | [valid_if.cuh](../../cudf/cpp/include/cudf/detail/valid_if.cuh#L144) | Build null mask for the gathered string key output column. |

> **Total dominating cost**: Kernel 2 (aggregate, 5.12 ms) + Kernel 1 (insert, 4.78 ms) + Kernel A (cuco init, 4.11 ms) + Kernel E (retrieve_all, 3.44 ms) ≈ **17.5 ms** total.  
> `DeviceSelectSweepKernel` (E) is now the **4th most expensive** kernel — behind `single_pass_shmem_aggs_kernel` (2), `mapping_indices_kernel` (1), and hash table init (A).


---

## 2. Nsight Systems Observed Kernel Trace (Ground Truth)

> Captured with Nsight Systems on a single groupby+SUM operation.  
> Key: `label` (1 `utf8` string column, col 2). Value: `amount` (1 `int64` column, col 4). Input: 100M rows. Path: shared-memory sub-path (cardinality ≤ 128).

### 2a. Raw Kernel Sequence

> Durations are exact values from the Nsight Systems capture. Significant memory operations are
> interleaved with kernels to show where wall time is actually spent. Minor allocations (< 1 MB),
> small memcpys, and host-only driver calls (cuLibraryGetKernel, cuKernelGetName, etc.) are omitted.

```
[tag]   duration     event / kernel
──────────────────────────────────────────────────────────────────────────────────────────────────
[MEM]  22.911 ms    cudaMalloc  800 MB  (cuco::static_set storage)
                    ← 2 × 100M slots × 4 B. The 2× load-factor cap keeps probe chains short.
                    ← This single alloc is 20% of the total aggregate wall time.

[ A ]   4.105 ms    static_kernel  <initialize_functor<long,int>>  grid=390625×1×1, block=256×1×1
                    ← Fills all 200M slots with the empty-slot sentinel.
                    ← cuCollections/include/cuco/detail/storage/functors.cuh:54

[MEM]   5.526 ms    cudaStreamSynchronize
                    ← Host blocks until kernel A completes (sync after static_set ctor).

[MEM]  11.266 ms    cudaMalloc  400 MB  (global_mapping_indices)
                    ← One int32 per input row (100M × 4 B). Records which unique-key slot
                    ← each row maps to; consumed by both kernel 1 and kernel 2.

[ B ]   3.840 μs    static_kernel  <__uninitialized_fill::functor<int*,int>>  grid=108×1×1
                    ← thrust::uninitialized_fill on global_mapping_indices.

[ 1 ]   4.784 ms    mapping_indices_kernel  grid=432×1×1, block=128×1×1
                    ← Insert every row into global cuco::static_set; record mapping indices.
                    ← cudf/cpp/src/groupby/hash/compute_mapping_indices.cuh:120

[MEM]   5.313 ms    cudaMemcpyAsync  DtoH  4 B  (h_num_out — unique-key count)
                    ← Root cause: data dependency — h_num_out is written by mapping_indices_kernel,
                    ← so the copy cannot complete until the kernel finishes (~4.784 ms of GPU work).
                    ← Secondary effect: h_num_out is a non-pinned stack variable, so CUDA enforces
                    ← an implicit stream sync *inside* the cudaMemcpyAsync call rather than in a
                    ← separate sync. With pinned memory the 5 ms would appear in cudaStreamSynchronize
                    ← instead — same total latency, different trace attribution.
                    ← The explicit cudaStreamSynchronize(corrId=14332) after this costs 2.912 μs.

[MEM]  11.502 ms    cudaMalloc  400 MB  (unique-key output buffer)
                    ← Worst-case N slots for retrieve_all() / extract_populated_keys().

[ D ]   2.848 μs    DeviceCompactInitKernel  grid=272×1×1, block=128×1×1
                    ← cub::DeviceSelect::If pass 1: initialise per-tile prefix-sum scratch.

[ E ]   3.439 ms    DeviceSelectSweepKernel  grid=34723×1×1, block=384×1×1
                    ← Stream compaction: copies all filled slot indices into the 400 MB buffer.
                    ← cuCollections/include/cuco/detail/open_addressing/open_addressing_impl.cuh:944

[MEM]   4.657 ms    cudaMemcpyAsync  DtoH  8 B  (confirmed unique-key count)
                    ← Same pattern: data dependency on DeviceSelectSweepKernel (E) writing the
                    ← unique count. Launched only 5 μs after E; blocks until E finishes (~3.439 ms).
                    ← Non-pinned destination again forces the implicit sync inside the API call.
                    ← The cuStreamSynchronize after this costs only 9.136 μs — stream already idle.

[MEM]   2.830 ms    cudaFree  400 MB  (global_mapping_indices)
                    ← Released once key_transform_map is built; no longer needed.

[MEM]  11.463 ms    cudaMalloc  400 MB  (aggregation output table)
                    ← One int64 SUM accumulator per unique key (worst-case 100M × 8 B).

[ F ]   5.568 μs    transform_kernel  <permutation_iterator>  grid=1×1×1
                    ← compute_key_transform_map(): remap unique key row-indices.

[ G ]   4.960 μs    static_kernel  <op_wrapper_t<...>>  grid=108×1×1
                    ← thrust::for_each_n: renumber global_mapping_indices to dense [0, K).

[ H ]   1.312 μs    transform_kernel  <__return_constant<double>>  grid=1×1×1
                    ← thrust::fill: output offset initialisation.

[ 2 ]   5.119 ms    single_pass_shmem_aggs_kernel  grid=432×1×1, block=128×1×1
                    ← Two-phase SUM: Phase 1 row→shmem, Phase 2 shmem→global (atomicAdd).
                    ← cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu:207

[MEM]   3.297 ms    cudaFree  400 MB  (unique-key output buffer)
[MEM]   2.715 ms    cudaFree  400 MB  (aggregation output table)
[MEM]   5.104 ms    cudaFree  800 MB  (cuco::static_set storage)
                    ← All three large RMM buffers freed synchronously on teardown.
                    ← The 800 MB free alone costs more than kernels D+E combined.

[ I ]   1.376 μs    DeviceScanInitKernel  grid=1×1×1, block=128×1×1
                    ← cudf::strings::gather: prefix-scan init over string char offsets.

[ J ]  10.464 μs    DeviceScanKernel  grid=1×1×1, block=224×1×1
                    ← Inclusive prefix-sum of string char offsets → output offset buffer.

[ K ]   4.928 μs    gather_chars_fn_char_parallel  grid=1×1×1, block=128×1×1
                    ← Copy char data for the string key column into the gathered output.
                    ← cudf/cpp/include/cudf/strings/detail/gather.cuh:156

[ L ]   2.592 μs    valid_if_n_kernel  grid=1×1×1, block=256×1×1
                    ← Build null mask for the gathered string key output column.
                    ← cudf/cpp/include/cudf/detail/valid_if.cuh:144
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
| **Non-kernel overhead** (cudaMalloc × 3 × 400 MB, cudaFree, cudaMemcpy, sync) | — | **≈ 96.8 ms** |
| **Total wall time (`libcudf:aggregate` NVTX region)** | | **≈ 114.3 ms** |

> **Note**: `single_pass_shmem_aggs_kernel` (2, 5.119 ms) is now the most expensive kernel, followed by `mapping_indices_kernel` (1, 4.784 ms), hash table init (A, 4.105 ms), and `DeviceSelectSweepKernel` (E, 3.439 ms). The unique-key extraction step accounts for ~20% of total kernel time.
>
> **Only ~15% of the total aggregate wall time is actual GPU kernel execution.** The remaining ~85% (~96.8 ms) is dominated by synchronous memory management — three `cudaMalloc` calls allocating 400 MB each for the hash table and output buffers, plus the corresponding `cudaFree` calls. This is the dominant cost for single-shot workloads and would amortise significantly with buffer reuse across repeated calls.


# NCU analysis

```bash
ncu --set full \
    --kernel-name-base function \
    --kernel-name regex:"mapping_indices_kernel|DeviceCompactInitKernel|DeviceSelectSweepKernel|DeviceScanInitKernel|DeviceScanKernel|gather_chars_fn_char_parallel|single_pass_shmem_aggs_kernel" \
    -o reports/libcudf_groupby_orders_100M_ncu \
    ./build/libcudf_tpch_orders_groupby --input ./data/orders_100M.parquet
```