# Analysis of NVIDIA `libcudf` Structure (Kernel-Focused, DBMS Lens)

## Scope

This report analyzes the [cpp/](../cudf/cpp/) `libcudf` codebase (especially CUDA kernel-backed components) and summarizes which database-system aspects are implemented.

Primary areas inspected:

- [cpp/src/*](../cudf/cpp/src/) kernel/operator implementations (`join`, `groupby`, `sort`, `io`, `ast`, etc.)
- [cpp/include/cudf/*](../cudf/cpp/include/cudf/) public/internal data model + API contracts
- [cpp/doxygen/*](../cudf/cpp/doxygen/) architecture/developer docs
- [python/cudf_polars/*](../cudf/python/cudf_polars/) only where physical/logical planning is relevant

---

## 1) High-level architecture

`libcudf` is a GPU-accelerated **columnar execution library**, not a full standalone DBMS.

- Core data model is columnar (`cudf::column`, `cudf::table`) with nullable semantics and nested types.
- Execution is operator-centric (joins, groupby, aggregations, sorting, filtering, transforms).
- Most heavy computation is in CUDA kernels (`.cu`/`.cuh`) and GPU-parallel primitives (Thrust, cuCollections/cuco).
- It behaves like a **vectorized query execution engine backend** that higher layers call.

Evidence:

- [cpp/doxygen/main_page.md](../cudf/cpp/doxygen/main_page.md)
- [cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md](../cudf/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md)
- [cpp/src/*](../cudf/cpp/src/)

---

## 2) Kernel code organization supporting cuDF features

The [cpp/src](../cudf/cpp/src/) layout is feature/operator modular and maps closely to DataFrame + relational operations:

- Relational: [cpp/src/join/](../cudf/cpp/src/join/), [cpp/src/groupby/](../cudf/cpp/src/groupby/), [cpp/src/sort/](../cudf/cpp/src/sort/), [cpp/src/search/](../cudf/cpp/src/search/), [cpp/src/partitioning/](../cudf/cpp/src/partitioning/)
- Expressions/compute: [cpp/src/ast/](../cudf/cpp/src/ast/), [cpp/src/binaryop/](../cudf/cpp/src/binaryop/), [cpp/src/unary/](../cudf/cpp/src/unary/), [cpp/src/transform/](../cudf/cpp/src/transform/), [cpp/src/reductions/](../cudf/cpp/src/reductions/)
- Nested/text analytics: [cpp/src/strings/](../cudf/cpp/src/strings/), [cpp/src/lists/](../cudf/cpp/src/lists/), [cpp/src/structs/](../cudf/cpp/src/structs/), [cpp/src/text/](../cudf/cpp/src/text/), [cpp/src/json/](../cudf/cpp/src/json/)
- Ingestion/egress: [cpp/src/io/](../cudf/cpp/src/io/) (Parquet/ORC/CSV/JSON/Avro + compression and metadata)
- Runtime/context: [cpp/src/runtime/](../cudf/cpp/src/runtime/) (global context, JIT cache init, nvCOMP loading)

Representative kernel-heavy folders/files:

- Joins: [cpp/src/join/mixed_join*.cu](../cudf/cpp/src/join/), [cpp/src/join/distinct_hash_join.cu](../cudf/cpp/src/join/distinct_hash_join.cu), [cpp/src/join/filtered_join.cu](../cudf/cpp/src/join/filtered_join.cu), [cpp/src/join/sort_merge_join.hpp](../cudf/cpp/src/join/sort_merge_join.hpp)
- GroupBy: [cpp/src/groupby/groupby.cu](../cudf/cpp/src/groupby/groupby.cu), [cpp/src/groupby/hash/*](../cudf/cpp/src/groupby/hash/), [cpp/src/groupby/sort/*](../cudf/cpp/src/groupby/sort/)
- Parquet IO filters: [cpp/src/io/parquet/predicate_pushdown.cpp](../cudf/cpp/src/io/parquet/predicate_pushdown.cpp), [cpp/src/io/parquet/bloom_filter_reader.cu](../cudf/cpp/src/io/parquet/bloom_filter_reader.cu)

---

## 3) Database-system aspects implemented

### 3.1 Relational/dataframe execution operators

Implemented strongly:

- Selection/filter (including AST predicate-based filtering)
- Projection/expressions (AST + row IR conversion)
- Join variants (hash join, sort-merge join, filtered join, semi/full/left/inner paths)
- GroupBy aggregates and grouped scans
- Sorting, searching, set-like operations, reshaping

Evidence:

- [cpp/include/cudf/join/join.hpp](../cudf/cpp/include/cudf/join/join.hpp)
- [cpp/src/join/*](../cudf/cpp/src/join/)
- [cpp/src/groupby/groupby.cu](../cudf/cpp/src/groupby/groupby.cu)
- [cpp/include/cudf/ast/expressions.hpp](../cudf/cpp/include/cudf/ast/expressions.hpp)

### 3.2 SQL frontend / transactions / storage-engine services

Not implemented as a full DBMS in `libcudf`:

- No SQL parser/planner/optimizer in core [cpp/libcudf](../cudf/cpp/)
- No transaction manager / WAL / lock manager / MVCC subsystem
- No persistent table/index storage manager in the OLTP sense

Interpretation: `libcudf` is an execution/compute substrate for higher-level systems (Python cuDF, Polars GPU engine, SQL engines integrating with RAPIDS).

---

## 4) File formats and I/O capabilities

### 4.1 Supported formats (core)

From [cpp/include/cudf/io/*](../cudf/cpp/include/cudf/io/), [cpp/src/io/*](../cudf/cpp/src/io/):

- Parquet
- ORC
- CSV
- JSON
- Avro (reader path)
- Text submodules under [cpp/src/io/text/](../cudf/cpp/src/io/text/)

Evidence:

- [cpp/include/cudf/io/parquet.hpp](../cudf/cpp/include/cudf/io/parquet.hpp)
- [cpp/include/cudf/io/orc.hpp](../cudf/cpp/include/cudf/io/orc.hpp)
- [cpp/include/cudf/io/csv.hpp](../cudf/cpp/include/cudf/io/csv.hpp)
- [cpp/include/cudf/io/json.hpp](../cudf/cpp/include/cudf/io/json.hpp)
- [cpp/include/cudf/io/avro.hpp](../cudf/cpp/include/cudf/io/avro.hpp)

### 4.2 Metadata/statistics-aware I/O

Implemented:

- Parquet metadata/schema APIs, row-group metadata
- ORC metadata/schema/statistics APIs
- Writer-side statistics controls (`statistics_freq`), dictionary policy, compression controls

Evidence:

- [cpp/include/cudf/io/parquet_metadata.hpp](../cudf/cpp/include/cudf/io/parquet_metadata.hpp)
- [cpp/include/cudf/io/orc_metadata.hpp](../cudf/cpp/include/cudf/io/orc_metadata.hpp)
- [cpp/include/cudf/io/types.hpp](../cudf/cpp/include/cudf/io/types.hpp)

---

## 5) Data structures used as “indexes” (DB interpretation)

`libcudf` does **not** implement classic persistent DB indexes (e.g., B+ trees on disk pages). Instead it uses execution-time index-like structures optimized for GPU analytics:

1. **Join/groupby hash structures (ephemeral, device-resident)**
   - Built with cuCollections (`cuco::static_set`, pair-based hash storage, probing schemes).
   - Used for hash joins, key remapping, and hash groupby paths.
   - Evidence: [cpp/src/join/mixed_join.cu](../cudf/cpp/src/join/mixed_join.cu), [cpp/src/join/key_remapping.cu](../cudf/cpp/src/join/key_remapping.cu), [cpp/src/join/distinct_hash_join.cu](../cudf/cpp/src/join/distinct_hash_join.cu), [cpp/src/groupby/hash/*](../cudf/cpp/src/groupby/hash/).

2. **Row index vectors / gather maps**
   - Join outputs are index vectors (`left_indices`, `right_indices`) with sentinel `JoinNoMatch`.
   - Used to materialize join outputs and post-filtering semantics.
   - Evidence: [cpp/include/cudf/join/join.hpp](../cudf/cpp/include/cudf/join/join.hpp), [cpp/src/join/filter_join_indices.cu](../cudf/cpp/src/join/filter_join_indices.cu).

3. **Offset indexes for variable-length/nested columns**
   - Strings/lists use offset child columns as positional index structures into chars/child buffers.
   - Evidence: [cpp/include/cudf/strings/strings_column_view.hpp](../cudf/cpp/include/cudf/strings/strings_column_view.hpp), [cpp/include/cudf/lists/lists_column_view.hpp](../cudf/cpp/include/cudf/lists/lists_column_view.hpp).

4. **Dictionary-encoded indices**
   - Dictionary columns store sorted unique keys + integer indices.
   - Evidence: [cpp/include/cudf/dictionary/dictionary_column_view.hpp](../cudf/cpp/include/cudf/dictionary/dictionary_column_view.hpp).

So: index-like structures exist, but primarily as **in-memory analytical execution structures**, not full secondary-index subsystems.

---

## 6) Buffer pool / memory management model

### 6.1 What is implemented

- Memory management delegates to **RMM memory resources** (`rmm::device_async_resource_ref`, current device MR wrappers).
- APIs consistently take stream + memory resource params for allocation control.
- Global context handles optional component init (JIT cache, nvCOMP loading).

Evidence:

- [cpp/include/cudf/utilities/memory_resource.hpp](../cudf/cpp/include/cudf/utilities/memory_resource.hpp)
- [cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md](../cudf/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md) (Memory Resources section)
- [cpp/include/cudf/context.hpp](../cudf/cpp/include/cudf/context.hpp)
- [cpp/src/runtime/context.cpp](../cudf/cpp/src/runtime/context.cpp)

### 6.2 What is not implemented

- No traditional DBMS buffer pool/page replacement manager for disk pages.
- No page cache with pin/unpin, LRU/clock replacement, dirty-page writeback semantics at `libcudf` layer.

Interpretation: device memory pooling/caching is handled by RMM allocator strategies, not a DB-style page buffer manager.

---

## 7) Physical vs logical plans

### 7.1 In core `libcudf`

- No full query logical/physical planner akin SQL engines.
- There is expression-level AST + internal row IR (`cudf::detail::row_ir`) for expression evaluation/codegen support.
- Operator dispatch decisions exist (e.g., choose hash vs sort groupby).

Evidence:

- [cpp/include/cudf/ast/expressions.hpp](../cudf/cpp/include/cudf/ast/expressions.hpp)
- [cpp/src/groupby/groupby.cu](../cudf/cpp/src/groupby/groupby.cu)

### 7.2 In `cudf_polars` (outside core `libcudf`, but in this repo)

- Explicit logical IR translation and execution model.
- Documentation explicitly distinguishes translated logical-plan IR and final “physical-plan” IR in streaming executor.
- Includes statistics-based planning options and join/reduction heuristics.

Evidence:

- [python/cudf_polars/docs/overview.md](../cudf/python/cudf_polars/docs/overview.md)

Conclusion: physical/logical planning is mostly a **higher-layer concern** (`cudf_polars`), while `libcudf` is the kernel/operator backend.

---

## 8) Query optimization techniques observed

### 8.1 Operator/algorithm selection

- GroupBy dynamically chooses hash vs sort path depending on key sortedness and supported aggregations.
- Sort-groupby helper reuse avoids redundant work after sorted state is established.

Evidence:

- [cpp/src/groupby/groupby.cu](../cudf/cpp/src/groupby/groupby.cu)

### 8.2 Predicate pushdown and pruning in Parquet scan

- AST-based stats filtering over row-group min/max/null-count metadata.
- Bloom filter-based row-group elimination for equality predicates.
- Reports surviving row groups in metadata fields.

Evidence:

- [cpp/src/io/parquet/predicate_pushdown.cpp](../cudf/cpp/src/io/parquet/predicate_pushdown.cpp)
- [cpp/src/io/parquet/stats_filter_helpers.hpp](../cudf/cpp/src/io/parquet/stats_filter_helpers.hpp)
- [cpp/src/io/parquet/bloom_filter_reader.cu](../cudf/cpp/src/io/parquet/bloom_filter_reader.cu)
- [cpp/include/cudf/io/types.hpp](../cudf/cpp/include/cudf/io/types.hpp) (`num_row_groups_after_stats_filter`, `num_row_groups_after_bloom_filter`)

### 8.3 I/O partitioning/chunking and compressed execution paths

- Parquet reader implementation includes chunking/preprocess kernels and codec integration.
- ORC supports stripe/row-index-aware reading controls.

Evidence:

- [cpp/src/io/parquet/reader_impl_chunking*.{cu,hpp}](../cudf/cpp/src/io/parquet/)
- [cpp/include/cudf/io/orc.hpp](../cudf/cpp/include/cudf/io/orc.hpp)

### 8.4 Expression/runtime tuning hooks

- Optional JIT paths (`use_jit_filter`, context flags for JIT cache/codegen dump).

Evidence:

- [cpp/include/cudf/io/parquet.hpp](../cudf/cpp/include/cudf/io/parquet.hpp)
- [cpp/include/cudf/context.hpp](../cudf/cpp/include/cudf/context.hpp)

---

## 9) Bottom-line DBMS characterization

`libcudf` implements a substantial subset of **analytical DB execution engine** functionality:

- Columnar execution primitives
- GPU-optimized join/groupby/sort/filter kernels
- Rich file-format readers/writers with metadata/statistics-aware pruning
- In-memory index-like execution structures (hash tables, offsets, dictionary indices)

But it is **not** a complete autonomous DBMS in core C++:

- No full SQL planning pipeline in `libcudf`
- No classic persistent index manager
- No transaction/log/recovery subsystem
- No DB-style buffer pool/page manager

A practical mental model:

- `libcudf` = GPU columnar execution + I/O kernel library
- `cudf_polars`/other frontends = where higher-level logical/physical planning appears

---

## 10) CUDA libraries used across `.cu` files (repo-wide scan)

Snapshot from a repository-wide scan of all `.cu` files:

- Total `.cu` files scanned: **504**

### Ranked by importance (prevalence + execution role)

| Rank | Library | Include hits in `.cu` | Distinct `.cu` files | How it is used |
|---|---|---:|---:|---|
| 1 | Thrust | 974 | 304 | Core GPU parallel algorithms (`transform`, `reduce`, `scan`, `sort`, iterator pipelines) across transforms, reductions, text, and IO kernels. |
| 2 | cuCollections (`cuco`) | 32 | 22 | GPU hash/set/map primitives for hash joins, groupby hash paths, dictionary encoding, search/contains, and Parquet bloom filtering. |
| 3 | CUB | 35 | 29 | Block/warp/device collectives (`BlockReduce`, `BlockScan`, `WarpReduce`, `DeviceReduce`) for low-level kernel performance, especially in join/bitmask/IO decode-encode paths. |
| 4 | CUDA Cooperative Groups | 33 | 21 | Fine-grained intra-block/warp coordination (`this_thread_block`, tiled partitions, group reductions) in text, groupby, and Parquet kernels. |
| 5 | CUDA C++ Standard Library (`libcu++`) | 321 | 193 | Device-side modern C++ utilities (`cuda::std` types such as `optional`, `span`, `tuple`, atomics, traits) used in generic and JIT-related kernels. |
| 6 | nvCOMP | 2 | 2 | Compression/decompression integration paths used by IO runtime and codec loading/execution. |
| 7 | CUDA Runtime/Driver headers | 5 | 5 | Direct runtime/driver integration where lower-level CUDA APIs are needed. |

### Short examples of usage in this codebase

- **Thrust**: [cpp/src/transform/mask_to_bools.cu](../cudf/cpp/src/transform/mask_to_bools.cu), [cpp/src/reductions/sum_with_overflow.cu](../cudf/cpp/src/reductions/sum_with_overflow.cu), [cpp/src/text/minhash.cu](../cudf/cpp/src/text/minhash.cu)
- **cuCollections (`cuco`)**: [cpp/src/join/key_remapping.cu](../cudf/cpp/src/join/key_remapping.cu), [cpp/src/groupby/hash/compute_groupby.cu](../cudf/cpp/src/groupby/hash/compute_groupby.cu), [cpp/src/io/parquet/bloom_filter_reader.cu](../cudf/cpp/src/io/parquet/bloom_filter_reader.cu)
- **CUB**: [cpp/src/io/parquet/page_enc.cu](../cudf/cpp/src/io/parquet/page_enc.cu), [cpp/src/bitmask/null_mask.cu](../cudf/cpp/src/bitmask/null_mask.cu), [cpp/src/join/sort_merge_join.cu](../cudf/cpp/src/join/sort_merge_join.cu)
- **Cooperative Groups**: [cpp/src/io/parquet/decode_preprocess.cu](../cudf/cpp/src/io/parquet/decode_preprocess.cu), [cpp/src/text/wordpiece_tokenize.cu](../cudf/cpp/src/text/wordpiece_tokenize.cu), [cpp/src/groupby/hash/compute_shared_memory_aggs.cu](../cudf/cpp/src/groupby/hash/compute_shared_memory_aggs.cu)
- **`libcu++`**: [cpp/src/quantiles/tdigest/tdigest_aggregation.cu](../cudf/cpp/src/quantiles/tdigest/tdigest_aggregation.cu), [cpp/src/stream_compaction/unique.cu](../cudf/cpp/src/stream_compaction/unique.cu), [cpp/src/transform/jit/kernel.cu](../cudf/cpp/src/transform/jit/kernel.cu)
- **nvCOMP**: [cpp/src/runtime/context.cpp](../cudf/cpp/src/runtime/context.cpp), [cpp/src/io/comp/](../cudf/cpp/src/io/comp/)

### Top 10 most-used methods/symbols per library (from `.cu`/`.cuh` scan)

Note: This ranking is based on namespace-qualified symbol frequency in `.cu`/`.cuh` files. For template-heavy libraries, entries include both callable APIs and heavily used type/iterator symbols.

#### Thrust

1. `thrust::make_counting_iterator` (513)
2. `thrust::counting_iterator` (317)
3. `thrust::transform` (231)
4. `thrust::seq` (174)
5. `thrust::make_transform_iterator` (146)
6. `thrust::make_zip_iterator` (95)
7. `thrust::for_each_n` (63)
8. `thrust::host_vector` (58)
9. `thrust::for_each` (53)
10. `thrust::make_discard_iterator` (51)

#### cuCollections (`cuco`)

1. `cuco::pair` (87)
2. `cuco::extent` (35)
3. `cuco::linear_probing` (35)
4. `cuco::empty_key` (31)
5. `cuco::static_set` (25)
6. `cuco::op` (19)
7. `cuco::detail` (15)
8. `cuco::static_set_ref` (15)
9. `cuco::thread_scope_device` (13)
10. `cuco::storage` (11)

#### CUB

1. `cub::BlockReduce` (42)
2. `cub::DeviceReduce` (24)
3. `cub::BlockScan` (23)
4. `cub::DeviceRadixSort` (22)
5. `cub::WarpReduce` (17)
6. `cub::DeviceScan` (16)
7. `cub::DeviceSegmentedReduce` (14)
8. `cub::DeviceMergeSort` (12)
9. `cub::DeviceSegmentedSort` (12)
10. `cub::DeviceTransform` (10)

#### CUDA Cooperative Groups

1. `cg::this_thread_block` (57)
2. `cg::tiled_partition` (29)
3. `cg::this_grid` (17)
4. `cg::thread_block` (14)
5. `cg::reduce` (13)
6. `cg::plus` (8)
7. `cg::less` (4)
8. `cg::thread_block_tile` (3)
9. `cg::greater` (2)
10. `cg::tile` (2)

#### CUDA C++ Standard Library (`libcu++`)

1. `cuda::std::pair` (166)
2. `cuda::std::distance` (150)
3. `cuda::std::chrono` (136)
4. `cuda::std::optional` (97)
5. `cuda::std::get` (94)
6. `cuda::std::numeric_limits` (94)
7. `cuda::std::is_same_v` (82)
8. `cuda::std::byte` (79)
9. `cuda::std::memory_order_relaxed` (69)
10. `cuda::std::plus` (62)

#### nvCOMP

Direct `nvcomp*` method calls are sparse in `.cu`/`.cuh` (integration is mostly through adapter logic and status types).

Top observed symbols:

1. `nvcomp` (6)
2. `nvcomp_stats` (4)
3. `nvcompStatus_t` (3)
4. `nvcomp_status` (2)
5. `nvcomp_adapter` (1)
6. `nvcompSuccess` (1)

#### CUDA Runtime/Driver APIs

1. `cudaMemcpyAsync` (36)
2. `cudaMemsetAsync` (14)
3. `cudaGetDevice` (6)
4. `cudaEventRecord` (5)
5. `cudaGetSymbolAddress` (4)
6. `cudaDeviceSynchronize` (4)
7. `cudaDeviceGetAttribute` (4)
8. `cudaMemcpyToSymbolAsync` (3)
9. `cudaEventCreate` (3)
10. `cudaStreamWaitEvent` (3)

### Additional note

`RMM` appears very frequently in `.cu` sources (726 include hits across 374 files) for stream and memory-resource management, but it is a RAPIDS memory library rather than a CUDA Toolkit library.

---

## 11) Deep dive: CUDA Programming Guide advanced features (4.2–4.20)

Method: static scan of `.cu` and `.cuh` sources in this repo plus manual spot-checking of matched files.

### Summary matrix

| Feature | Evidence in `.cu/.cuh` | Assessment |
|---|---|---|
| 4.2 CUDA Graphs | No `cudaGraph*` / stream-capture API hits in `.cu/.cuh` | **Not observed** |
| 4.3 Stream-Ordered Memory Allocator | Extensive `rmm::device_async_resource_ref` usage; no direct `cudaMallocAsync`/`cudaMemPool*` hits | **Observed (via RMM abstraction), no direct runtime API usage** |
| 4.4 Cooperative Groups | Widespread `cooperative_groups`, `cg::tiled_partition`, `cg::this_thread_block`, `cg::this_grid` | **Strongly observed** |
| 4.5 Programmatic Dependent Launch & Sync | `cudaEventRecord`, `cudaStreamWaitEvent` used across IO/interop | **Observed** |
| 4.6 Green Contexts | No `cudaGreenCtx`/`cuGreenCtx` hits | **Not observed** |
| 4.7 Lazy Loading | No concrete lazy-module loading API usage in `.cu/.cuh` | **Not observed** |
| 4.8 Error Log Management | `cudaPeekAtLastError`, `cudaGetLastError` present | **Observed (basic runtime error polling)** |
| 4.9 Asynchronous Barriers | No `cuda::barrier` / `<cuda/barrier>` / `mbarrier` hits | **Not observed** |
| 4.10 Pipelines | No `cuda::pipeline` / `<cuda/pipeline>` hits | **Not observed** |
| 4.11 Asynchronous Data Copies | Many `cudaMemcpyAsync` uses | **Strongly observed** |
| 4.12 Work Stealing with Cluster Launch Control | No cluster-launch API/attributes found | **Not observed** |
| 4.13 L2 Cache Control | No persisting-L2 access-policy APIs found | **Not observed** |
| 4.14 Memory Synchronization Domains | Heavy `cuda::atomic_ref<..., cuda::thread_scope_*>` and memory-order usage | **Strongly observed** |
| 4.15 Interprocess Communication | No `cudaIpc*` hits in `.cu/.cuh` | **Not observed** |
| 4.16 Virtual Memory Management | No `cuMemMap`/`cuMemCreate` style VMM API hits | **Not observed** |
| 4.17 Extended GPU Memory | No `cudaMallocManaged`/`cudaMemPrefetchAsync` hits in `.cu/.cuh` | **Not observed** |
| 4.18 CUDA Dynamic Parallelism | No clear device-side child-kernel launch pattern found | **Not observed** |
| 4.19 CUDA Interoperability with APIs | No concrete graphics/external-memory interop API use in `.cu/.cuh` | **Not observed** |
| 4.20 Driver Entry Point Access | No `cuGetProcAddress` / `cudaGetDriverEntryPoint` hits | **Not observed** |

### Evidence and snippets for observed features

#### 4.3 Stream-Ordered Memory Allocator (via RMM async resource refs)

This codebase primarily uses stream-ordered allocation through RMM (`device_async_resource_ref`) passed through APIs.

Source: [cpp/src/transform/transform.cu](../cudf/cpp/src/transform/transform.cu)

```cuda-cpp
rmm::cuda_stream_view stream,
rmm::device_async_resource_ref mr)
{
   auto output_typenames  = cudf::jit::output_type_names(output_columns);
```

#### 4.4 Cooperative Groups

Common usage includes tiled warp partitions and group-level ops for decode and text kernels.

Source: [cpp/src/io/parquet/decode_preprocess.cu](../cudf/cpp/src/io/parquet/decode_preprocess.cu)

```cuda-cpp
auto const warp = cg::tiled_partition<cudf::detail::warp_size>(block);
if (warp.meta_group_rank() == 0) {
   ...
   uint32_t const list_mask = warp.ballot(is_list);
```

#### 4.5 Programmatic Dependent Launch and Synchronization

Stream dependency is built using events (`record` + `wait`) across multi-stage pipelines.

Source: [cpp/src/io/text/multibyte_split.cu](../cudf/cpp/src/io/text/multibyte_split.cu)

```cuda-cpp
CUDF_CUDA_TRY(cudaStreamWaitEvent(scan_stream.value(), last_launch_event));
...
CUDF_CUDA_TRY(cudaEventRecord(last_launch_event, scan_stream.value()));
```

Also in Arrow device interop:

Source: [cpp/src/interop/from_arrow_device.cu](../cudf/cpp/src/interop/from_arrow_device.cu)

```cuda-cpp
if (input->sync_event != nullptr) {
   CUDF_CUDA_TRY(
      cudaStreamWaitEvent(stream.value(), *reinterpret_cast<cudaEvent_t*>(input->sync_event)));
}
```

#### 4.8 Error Log Management

Runtime error probing APIs are used (not a separate “error log subsystem”, but explicit error checks).

Source: [cpp/src/io/fst/dispatch_dfa.cuh](../cudf/cpp/src/io/fst/dispatch_dfa.cuh)

```cuda-cpp
cudaError_t error = cudaSuccess;
if (CubDebug(error = cudaPeekAtLastError())) return error;
```

Source: [cpp/tests/identify_stream_usage/test_default_stream_identification.cu](../cudf/cpp/tests/identify_stream_usage/test_default_stream_identification.cu)

```cuda-cpp
err = cudaGetLastError();
if (err != cudaSuccess) { throw std::runtime_error("Kernel failed on non-default stream!"); }
```

#### 4.11 Asynchronous Data Copies

`cudaMemcpyAsync` is used broadly for device/host staging in utilities, IO, and interop.

Source: [cpp/src/utilities/cuda_memcpy.cu](../cudf/cpp/src/utilities/cuda_memcpy.cu)

```cuda-cpp
CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
```

#### 4.14 Memory Synchronization Domains

The code uses scoped atomics (`thread_scope_device`, `thread_scope_block`) and explicit memory orders.

Source: [cpp/src/reductions/any.cu](../cudf/cpp/src/reductions/any.cu)

```cuda-cpp
cuda::atomic_ref<int32_t, cuda::thread_scope_device> ref{*d_result};
ref.fetch_or(1, cuda::std::memory_order_relaxed);
```

### Practical interpretation

- The advanced CUDA features that are most actively used in `libcudf` kernels are:
   - **Cooperative Groups**
   - **Async stream/event synchronization patterns**
   - **Async copies (`cudaMemcpyAsync`)**
   - **Scoped atomics and memory-order controls**
- A number of newer/specialized guide topics (CUDA Graphs, async barriers/pipelines, cluster launch control, VMM, IPC, dynamic parallelism) are **not evident** in `.cu/.cuh` at present.

---

## 12) End-to-end layered call stack: `read_csv` → `DataFrame.groupby(...).agg(...)`

### Example workflow

```python
import cudf

df = cudf.read_csv("input.csv")
out = df.groupby("key").agg({"value": "sum"})
```

### 12.1 Layered call stack (main functions only)

#### A. CSV read path

1. **Python API layer (`cudf`)**
   - `cudf.read_csv(...)`
   - [python/cudf/cudf/io/csv.py](../cudf/python/cudf/cudf/io/csv.py) → `read_csv(...)`
   - Main handoff call: `plc.io.csv.read_csv(options)`

2. **Python bindings layer (`pylibcudf`)**
   - [python/pylibcudf/pylibcudf/io/csv.pyx](../cudf/python/pylibcudf/pylibcudf/io/csv.pyx) → `read_csv(...)`
   - Main handoff call: `cpp_read_csv(options.c_obj, s.view(), mr.get_mr())`

3. **C++ API layer (`libcudf` front door)**
   - [cpp/src/io/functions.cpp](../cudf/cpp/src/io/functions.cpp) → `cudf::io::read_csv(...)`
   - Main handoff call: `cudf::io::detail::csv::read_csv(...)`

4. **C++/CUDA execution layer (`libcudf` CSV reader)**
   - [cpp/src/io/csv/reader_impl.cu](../cudf/cpp/src/io/csv/reader_impl.cu) → `read_csv(...)`
   - Main stages:
     - `select_data_and_row_offsets(...)`
     - `load_data_and_gather_row_offsets(...)`
     - `cudf::io::csv::gpu::gather_row_offsets(...)`
     - `cudf::io::csv::gpu::remove_blank_rows(...)`
     - `determine_column_types(...)` / `infer_column_types(...)`
     - `decode_data(...)`
     - `cudf::io::csv::gpu::decode_row_column_data(...)`

#### B. GroupBy aggregate path

1. **Python API layer (`cudf`)**
   - `df.groupby("key").agg({"value": "sum"})`
   - [python/cudf/cudf/core/groupby/groupby.py](../cudf/python/cudf/cudf/core/groupby/groupby.py) → `GroupBy.agg(...)`
   - Main stage call: `_aggregate(...)`
   - Main handoff calls:
     - `plc.groupby.GroupBy(...)` (key-group object construction)
     - `plc.groupby.GroupByRequest(...)` (aggregation requests)
     - `plc_groupby.aggregate(requests)`

2. **Python bindings layer (`pylibcudf`)**
   - [python/pylibcudf/pylibcudf/groupby.pyx](../cudf/python/pylibcudf/pylibcudf/groupby.pyx)
   - `GroupBy.__cinit__(...)` creates `new groupby(...)`
   - `GroupBy.aggregate(...)` calls `dereference(self.c_obj).aggregate(...)`

3. **C++ API/execution dispatch layer (`libcudf`)**
   - [cpp/src/groupby/groupby.cu](../cudf/cpp/src/groupby/groupby.cu)
   - `groupby::aggregate(...)` → `dispatch_aggregation(...)`
   - Dispatch decision:
     - `detail::hash::groupby(...)` if hash path is valid
     - `sort_aggregate(...)` otherwise

4. **Hash groupby CUDA execution path (common for `sum`)**
   - [cpp/src/groupby/hash/groupby.cu](../cudf/cpp/src/groupby/hash/groupby.cu)
     - `can_use_hash_groupby(...)`
     - `dispatch_groupby(...)`
   - [cpp/src/groupby/hash/compute_groupby.cu](../cudf/cpp/src/groupby/hash/compute_groupby.cu)
     - `compute_groupby(...)`
     - `compute_single_pass_aggs(...)`

### 12.2 What the GPU is doing at each major step

#### CSV ingest (`read_csv`)

1. **Load + partition input bytes**
   - GPU receives CSV byte buffer(s) and partitions work by byte ranges.
   - Kernels scan for row boundaries while respecting quoting/escape rules.

2. **Find row offsets**
   - `gpu::gather_row_offsets(...)` computes row start/end offsets in parallel.
   - `gpu::remove_blank_rows(...)` removes blank-line rows without host-side loops.

3. **Infer/assign column types**
   - GPU samples/parses fields and classifies candidate dtypes for inferred columns.
   - User dtypes/parse hints override inference where specified.

4. **Decode fields into typed device columns**
   - `gpu::decode_row_column_data(...)` parses tokens directly into typed output buffers.
   - Null masks are built on device; valid counts determine null counts.
   - For string columns, quote/doublequote normalization is done on GPU string columns.

5. **Materialize table**
   - Device buffers become `cudf::column` objects and return as a `table_with_metadata`.

#### GroupBy aggregate (`groupby(...).agg(sum)`)

1. **Build key-grouping object**
   - Keys are wrapped as device table views and prepared for grouped execution.

2. **Choose algorithm: hash vs sort**
   - `groupby::dispatch_aggregation(...)` selects hash path when key/value types and aggs support fast atomic/hash implementation; otherwise sort path.

3. **Hash path: build/insert keys in GPU hash table**
   - Device row hash/equality functors are generated for key rows.
   - A `cuco::static_set` is built; each input row inserts/probes its key bucket in parallel.

4. **Compute aggregates per group**
   - Single-pass aggregations (like `sum`) are updated in parallel using atomic-friendly kernels.
   - Compound aggs (if requested) are finalized from cached partial results.

5. **Gather unique keys + finalize output columns**
   - Group key gather-map is produced and applied.
   - Aggregation result columns are materialized and returned to Python.

### 12.3 Main libraries used in this end-to-end path

- **`libcudf` IO + groupby kernels**: core execution implementation
- **Thrust**: iterator pipelines, parallel transforms/loops in both CSV and groupby internals
- **cuCollections (`cuco`)**: hash-table/set backbone for hash groupby
- **CUB**: low-level collective/reduction primitives used by several aggregation and IO kernels
- **CUDA Runtime APIs**: stream/event and async memory-copy primitives where needed
- **RMM**: stream-ordered device memory allocation and buffer ownership across layers

### 12.4 Practical DBMS interpretation of this stack

- This is an **operator pipeline** (scan/parse → groupby aggregate), not a page-buffered DB engine plan.
- GPU work is dominated by:
  - parallel tokenization/decoding for ingest,
  - hash-table construction/probing,
  - atomic/group reductions,
  - gather/materialization into columnar outputs.
- The layered stack is: **Python API → Cython bindings → libcudf C++ dispatch → CUDA kernels + GPU primitives**.

### 12.5 Worksheets (how to trace kernels and data exchange)

Use these as working sheets while reading code. Mark each row as you confirm it in source, and add timings/profiler notes.

#### Worksheet A — `read_csv` trace sheet

| Stage | Primary function(s) | File(s) | Kernel launch boundary to look for | Data exchange checkpoint | Main library/library primitive |
|---|---|---|---|---|---|
| API entry | `read_csv(...)` | [python/cudf/cudf/io/csv.py](../cudf/python/cudf/cudf/io/csv.py) | N/A (Python layer) | Python args/options normalized into reader options | `cudf` Python API |
| Python→Cython handoff | `plc.io.csv.read_csv(options)` | [python/cudf/cudf/io/csv.py](../cudf/python/cudf/cudf/io/csv.py) | N/A | `DataFrame.from_pylibcudf(...)` materialization on return | `pylibcudf` bridge |
| Cython→C++ handoff | `cpp_read_csv(...)` | [python/pylibcudf/pylibcudf/io/csv.pyx](../cudf/python/pylibcudf/pylibcudf/io/csv.pyx) | N/A | stream + memory resource explicitly passed | RMM stream/memory plumbing |
| C++ front door | `cudf::io::read_csv(...)` | [cpp/src/io/functions.cpp](../cudf/cpp/src/io/functions.cpp) | N/A | datasource objects created from source info | libcudf I/O dispatch |
| Row-boundary discovery | `load_data_and_gather_row_offsets(...)`, `gpu::gather_row_offsets(...)` | [cpp/src/io/csv/reader_impl.cu](../cudf/cpp/src/io/csv/reader_impl.cu) | wrapped GPU launch (`gpu::...`) | input byte buffer staged to device; row offsets created on device | CUDA kernels + Thrust/CUB support |
| Row cleanup | `gpu::remove_blank_rows(...)` | [cpp/src/io/csv/reader_impl.cu](../cudf/cpp/src/io/csv/reader_impl.cu) | wrapped GPU launch | row-offset vector compacted on device | CUDA kernels |
| Type resolution | `determine_column_types(...)`, `infer_column_types(...)` | [cpp/src/io/csv/reader_impl.cu](../cudf/cpp/src/io/csv/reader_impl.cu) | type inference kernels/helpers | inferred dtypes and flags flow host↔device | libcudf + device spans |
| Decode to columns | `decode_data(...)`, `gpu::decode_row_column_data(...)` | [cpp/src/io/csv/reader_impl.cu](../cudf/cpp/src/io/csv/reader_impl.cu) | wrapped GPU launch (`gpu::decode...`) | token bytes → typed column buffers + null masks | CUDA kernels |
| String quote normalization | `cudf::strings::detail::replace(...)`, `copy_if_else(...)` | [cpp/src/io/csv/reader_impl.cu](../cudf/cpp/src/io/csv/reader_impl.cu) | Thrust/strings kernel pipeline | selective replacement for quoted fields | Thrust + libcudf strings |
| Output assembly | `table_with_metadata` return | [cpp/src/io/csv/reader_impl.cu](../cudf/cpp/src/io/csv/reader_impl.cu) | N/A | device columns wrapped as libcudf table, then Python DataFrame | libcudf column/table model |

#### Worksheet B — `groupby(key).agg({value: "sum"})` trace sheet

| Stage | Primary function(s) | File(s) | Kernel launch boundary to look for | Data exchange checkpoint | Main library/library primitive |
|---|---|---|---|---|---|
| API entry | `GroupBy.agg(...)` | [python/cudf/cudf/core/groupby/groupby.py](../cudf/python/cudf/cudf/core/groupby/groupby.py) | N/A (Python layer) | aggregation spec normalized into internal requests | `cudf` GroupBy API |
| Request build | `_aggregate(...)`, `plc.groupby.GroupByRequest(...)` | [python/cudf/cudf/core/groupby/groupby.py](../cudf/python/cudf/cudf/core/groupby/groupby.py) | N/A | value columns + agg kinds packaged for C++ | `pylibcudf` request objects |
| Cython groupby object | `GroupBy.__cinit__(...)` | [python/pylibcudf/pylibcudf/groupby.pyx](../cudf/python/pylibcudf/pylibcudf/groupby.pyx) | N/A | key table views captured; null policy set | libcudf groupby constructor |
| Cython aggregate handoff | `GroupBy.aggregate(...)` | [python/pylibcudf/pylibcudf/groupby.pyx](../cudf/python/pylibcudf/pylibcudf/groupby.pyx) | N/A | calls `cudf::groupby::groupby::aggregate(...)` | Cython→C++ bridge |
| Algorithm dispatch | `groupby::dispatch_aggregation(...)` | [cpp/src/groupby/groupby.cu](../cudf/cpp/src/groupby/groupby.cu) | branch: hash vs sort path | request set inspected for hash compatibility | libcudf dispatch |
| Hash compatibility check | `can_use_hash_groupby(...)` | [cpp/src/groupby/hash/groupby.cu](../cudf/cpp/src/groupby/hash/groupby.cu) | N/A (decision logic) | validates agg/type support and atomic viability | libcudf type/agg dispatch |
| Hash table build/probe | `compute_groupby(...)`, `cuco::static_set` | [cpp/src/groupby/hash/compute_groupby.cu](../cudf/cpp/src/groupby/hash/compute_groupby.cu) | Thrust kernels (`tabulate`, `for_each_n`) + cuco ops | key rows hashed and inserted/probed on device | cuco + Thrust |
| Aggregate update | `compute_single_pass_aggs(...)` | [cpp/src/groupby/hash/compute_groupby.cu](../cudf/cpp/src/groupby/hash/compute_groupby.cu) | aggregation kernels and atomics | per-group partial/final sums updated in device memory | CUDA atomics + CUB/Thrust helpers |
| Final gather/materialize | gather map + output extraction | [cpp/src/groupby/hash/compute_groupby.cu](../cudf/cpp/src/groupby/hash/compute_groupby.cu), [cpp/src/groupby/hash/groupby.cu](../cudf/cpp/src/groupby/hash/groupby.cu) | gather kernel path | unique keys + aggregate columns returned to Python | libcudf gather + column/table outputs |

#### Quick grep checklist for kernel-launch boundaries

- `thrust::transform|thrust::for_each|thrust::tabulate`
- `cub::Device|cub::Block|cub::Warp`
- `gpu::[a-zA-Z0-9_]+\(` (libcudf launch wrappers)
- `cudaMemcpyAsync|cudaStreamWaitEvent|cudaEventRecord`
- `cuco::static_set|cuco::static_map`

Use these patterns to find the true execution boundaries when direct `<<< >>>` launches are not visible.

---

## 13) End-to-end layered call stack: reading Arrow data (`from_arrow`)

### Example workflow

```python
import pyarrow as pa
import cudf

tbl = pa.table({"a": [1, 2, 3], "b": [10, 20, 30]})
df = cudf.DataFrame.from_arrow(tbl)
```

### 13.1 Layered call stack (main functions only)

#### A. Python API (`pyarrow.Table` → `cudf.DataFrame`)

1. **Frame entrypoint**
   - [python/cudf/cudf/core/frame.py](../cudf/python/cudf/cudf/core/frame.py) → `Frame.from_arrow(...)`
   - Iterates Arrow columns and calls `ColumnBase.from_arrow(...)`

2. **Column entrypoint**
   - [python/cudf/cudf/core/column/column.py](../cudf/python/cudf/cudf/core/column/column.py) → `ColumnBase.from_arrow(...)`
   - Main handoff call: `plc.Column.from_arrow(array)`

3. **Python bindings (`pylibcudf`)**
   - [python/pylibcudf/pylibcudf/column.pyx](../cudf/python/pylibcudf/pylibcudf/column.pyx) → `Column.from_arrow(...)`
   - Dispatches by Arrow C interface support:
     - `__arrow_c_device_array__` (device path)
     - `__arrow_c_array__` (host path)
     - `__arrow_c_stream__` (streamed host batches)

#### B. C++ interop core (`libcudf`)

1. **Owning wrapper and conversion entry**
   - [cpp/src/interop/arrow_data_structures.cpp](../cudf/cpp/src/interop/arrow_data_structures.cpp)
   - `arrow_column(...)` / `arrow_table(...)` constructors bridge Arrow containers to libcudf views.

2. **Host Arrow ingestion path**
   - [cpp/src/interop/from_arrow_host.cu](../cudf/cpp/src/interop/from_arrow_host.cu)
   - Main entrypoints:
     - `from_arrow(...)`
     - `from_arrow_column(...)`
     - `from_arrow_host(...)`
     - `from_arrow_host_column(...)`

3. **Device Arrow ingestion path**
   - [cpp/src/interop/from_arrow_device.cu](../cudf/cpp/src/interop/from_arrow_device.cu)
   - Main entrypoints:
     - `from_arrow_device(...)`
     - `from_arrow_device_column(...)`
   - Type dispatch via `get_column(...)` / `dispatch_from_arrow_device`

4. **Arrow stream ingestion path**
   - [cpp/src/interop/from_arrow_stream.cu](../cudf/cpp/src/interop/from_arrow_stream.cu)
   - `from_arrow_stream(...)` / `from_arrow_stream_column(...)`
   - Per-chunk conversion using `from_arrow(...)` / `from_arrow_column(...)`, then concatenate.

### 13.2 What the GPU is doing at each major step

#### Host Arrow (`__arrow_c_array__`) path

1. **Allocate device columns/buffers**
   - libcudf allocates target device columns with RMM.

2. **Copy host Arrow buffers to device**
   - `cudaMemcpyAsync` copies fixed-width values, offsets, chars, and masks.
   - This is a host→device data migration path.

3. **Normalize bitmasks / booleans**
   - `copy_shifted_bitmask<<<...>>>` adjusts offset-shifted Arrow masks.
   - bool columns are converted from packed bit representation to cudf bool column (`mask_to_bools`).

4. **Materialize nested/dictionary/string columns**
   - Nested children are recursively converted.
   - Dictionary keys/indices and string offsets/chars are reconstructed as cudf-native columns.

#### Device Arrow (`__arrow_c_device_array__`) path

1. **Stream synchronization with producer**
   - If Arrow provides a sync event, `cudaStreamWaitEvent(...)` establishes ordering.

2. **Zero-copy view construction when layout-compatible**
   - For compatible types, libcudf builds `column_view` directly over Arrow device buffers.
   - No host transfer is required.

3. **Selective normalization where needed**
   - Some types still require conversion/ownership columns (e.g., bool unpacking or special handling paths).

4. **Return libcudf table/column views with owned-memory sidecar**
   - Custom deleters keep any newly-owned columns alive where zero-copy is not sufficient.

#### Arrow stream path

1. **Read chunk stream (ArrowArrayStream)**
   - Iteratively pulls chunks from stream interface.

2. **Convert each chunk (host/device path per chunk source)**
   - Uses the same conversion functions as non-stream paths.

3. **Concatenate chunk outputs**
   - Final table/column assembled by concatenating per-chunk cudf outputs.

### 13.3 Main libraries used in Arrow read path

- **`libcudf` interop layer** (`from_arrow_host.cu`, `from_arrow_device.cu`, `from_arrow_stream.cu`)
- **Nanoarrow / Arrow C Data Interface** (`ArrowSchema`, `ArrowArray`, `ArrowDeviceArray`, `ArrowArrayStream`)
- **CUDA Runtime APIs** (`cudaMemcpyAsync`, `cudaStreamWaitEvent`) for transfers/sync
- **RMM** for device allocation and stream-scoped memory ownership
- **Thrust/CUDA utilities** indirectly in conversion helpers and transforms

### 13.4 Worksheets (kernel/data-exchange trace)

#### Worksheet A — Host Arrow (`pyarrow.Table`) trace sheet

| Stage | Primary function(s) | File(s) | Kernel launch boundary to look for | Data exchange checkpoint | Main library/library primitive |
|---|---|---|---|---|---|
| API entry | `Frame.from_arrow(...)` | [python/cudf/cudf/core/frame.py](../cudf/python/cudf/cudf/core/frame.py) | N/A | Arrow table iterated column-by-column | `cudf` Python API |
| Column handoff | `ColumnBase.from_arrow(...)` | [python/cudf/cudf/core/column/column.py](../cudf/python/cudf/cudf/core/column/column.py) | N/A | `plc.Column.from_arrow(array)` invoked | `pylibcudf` bridge |
| Cython dispatch | `Column.from_arrow(...)` (`__arrow_c_array__`) | [python/pylibcudf/pylibcudf/column.pyx](../cudf/python/pylibcudf/pylibcudf/column.pyx) | N/A | Arrow capsules (schema/array) passed to C++ | Arrow C interface |
| Host conversion entry | `from_arrow(...)` / `from_arrow_host_column(...)` | [cpp/src/interop/from_arrow_host.cu](../cudf/cpp/src/interop/from_arrow_host.cu) | wrapper call boundary | Arrow CPU buffers become conversion input | libcudf interop |
| Buffer copy | `dispatch_copy_from_arrow_host` | [cpp/src/interop/from_arrow_host.cu](../cudf/cpp/src/interop/from_arrow_host.cu) | `cudaMemcpyAsync` | host→device copy for data/mask/offsets | CUDA runtime |
| Mask normalization | `copy_shifted_bitmask<<<...>>>` | [cpp/src/interop/from_arrow_host.cu](../cudf/cpp/src/interop/from_arrow_host.cu) | explicit kernel launch | shifted validity mask corrected for Arrow offsets | CUDA kernel |
| Bool/string/nested handling | `mask_to_bools`, `string_column_from_arrow_host`, recursive child conversion | [cpp/src/interop/from_arrow_host.cu](../cudf/cpp/src/interop/from_arrow_host.cu), [cpp/src/interop/from_arrow_host_strings.cu](../cudf/cpp/src/interop/from_arrow_host_strings.cu) | helper kernels + async copies | packed host Arrow encodings converted to cudf-native columns | libcudf interop + CUDA |
| Materialization | `arrow_column` / `arrow_table` wrapper | [cpp/src/interop/arrow_data_structures.cpp](../cudf/cpp/src/interop/arrow_data_structures.cpp) | N/A | cudf views/ownership objects returned to Python | libcudf interop wrappers |

#### Worksheet B — Device Arrow (`__arrow_c_device_array__`) trace sheet

| Stage | Primary function(s) | File(s) | Kernel launch boundary to look for | Data exchange checkpoint | Main library/library primitive |
|---|---|---|---|---|---|
| Cython dispatch | `Column.from_arrow(...)` (`__arrow_c_device_array__`) | [python/pylibcudf/pylibcudf/column.pyx](../cudf/python/pylibcudf/pylibcudf/column.pyx) | N/A | device Arrow capsules passed through | Arrow C device interface |
| Device conversion entry | `from_arrow_device(...)` / `from_arrow_device_column(...)` | [cpp/src/interop/from_arrow_device.cu](../cudf/cpp/src/interop/from_arrow_device.cu) | wrapper call boundary | ArrowDeviceArray validated for CUDA accessibility | libcudf interop |
| Producer/consumer sync | `cudaStreamWaitEvent(...)` | [cpp/src/interop/from_arrow_device.cu](../cudf/cpp/src/interop/from_arrow_device.cu) | runtime sync API | consume buffer only after producer event | CUDA runtime sync |
| Type dispatch | `get_column(...)` + `dispatch_from_arrow_device` | [cpp/src/interop/from_arrow_device.cu](../cudf/cpp/src/interop/from_arrow_device.cu) | conversion helper boundary | layout-compatible columns map to direct `column_view` | libcudf type dispatch |
| Special-case conversions | bool unpacking, list child slicing, dictionary child handling | [cpp/src/interop/from_arrow_device.cu](../cudf/cpp/src/interop/from_arrow_device.cu) | utility kernels/helpers | selective allocations for non-trivial representations | libcudf + CUDA utilities |
| Ownership + return | `unique_table_view_t` / `unique_column_view_t` with custom deleter | [cpp/include/cudf/interop.hpp](../cudf/cpp/include/cudf/interop.hpp) | N/A | preserves ownership where zero-copy is incomplete | libcudf interop ownership model |

#### Quick grep checklist for Arrow-read execution boundaries

- `from_arrow_host|from_arrow_device|from_arrow_stream`
- `cudaMemcpyAsync|cudaStreamWaitEvent`
- `copy_shifted_bitmask|mask_to_bools`
- `ArrowSchema|ArrowArray|ArrowDeviceArray|ArrowArrayStream`
- `concatenate\(` (stream chunk assembly)

Use these to distinguish pure interface plumbing from real transfer/compute points.
