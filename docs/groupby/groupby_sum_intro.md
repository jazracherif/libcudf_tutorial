# Inside libcudf: GPU acceleration of Relational Primitives - A Deep Technical Dive

Traditional database execution engines were designed for the CPU: optimized for a handful of powerful cores, deep cache hierarchies, and sequential or lightly-vectorized processing. However, as data volumes grow and CPU frequency scaling plateaus, Database researchers have increasingly turned to hardware acceleration. GPUs, with their massive memory bandwidth and thousands of parallel execution units, offer a highly compelling paradigm shift for analytical query performance.

But mapping relational algebra onto GPUs introduces a massive semantic gap. Operators like **joins**, **aggregations**, and **sorts** must be entirely reimagined for a SIMT (Single Instruction, Multiple Thread) architecture. Conventional algorithms natively optimized for CPUs often hit brutal bottlenecks on GPUs due to **thread divergence**, **uncoalesced memory access**, and severe penalties for **global synchronisation**. 

To bridge this runtime gap, NVIDIA developed **libcudf**: a C++ library implementing foundational DataFrame operations and relational primitives natively on the GPU. It has emerged as the de facto execution framework for a massive portion of the accelerated data ecosystem, underpinning projects like Spark RAPIDS, Dask-cuDF, and numerous independent database research efforts.

The central questions driving this deep technical dive are:
- **How good is libcudf?** 
- **How does it translate fundamental relational operators into massively parallel GPU kernels?**
- **What does the developer tooling look like, and how does one reason about its hardware utilization?** 
- **Can libcudf, as a technological foundation, sustain the complex demands of next-generation distributed databases?** 
- **What are its structural strengths, and where does the GPU memory/compute model impose hard limits?**

To answer these questions, we begin by mapping the landscape — identifying at a high level which algorithms underlie key primitives. We introduce just enough GPU architecture concepts to make the implementation legible, then ground the analysis in an actual profiling capture to evaluate real-world execution.

This report was compiled with assistance from AI agents.

## Mapping the internals of libcudf's GroupBy operation

`GROUP BY` is one of the foundational physical operators in relational database systems. Its job is to partition an input relation into disjoint subsets (groups) that share the same value for a designated key expression, and then reduce each group to a single output row by applying one or more aggregate functions — `SUM`, `COUNT`, `MIN`, `MAX`, `AVG`, etc.

In a query execution engine the `GROUP BY` physical operator must solve two subproblems simultaneously:

1. **Key partitioning** — determine, for every input row, which output group it belongs to. This is effectively a dictionary-encoding problem: map an arbitrarily-typed key (integer, string, composite) to a dense integer group-id in [0, K) where K is the number of distinct keys.

2. **Aggregation** — reduce all rows assigned to the same group-id to a single scalar per aggregate column (e.g., sum all values from column C for group-id 3).

If the data cannot be stored fully in the main memory, it must be broken up into chunks and the algorithm adapted to handles operating instead of blocks od data incrementally, as they move in and out of memory.

On CPUs this is typically implemented via a **hash table** (hash aggregate) or a **sort** followed by a sequential scan (sort aggregate). The choice between them depends on whether the aggregate function requires ordering and on the expected cardinality of the key column. 

On GPUs the same two strategies exist, but the implementation constraints are very different:

| Concern | CPU hash agg | GPU hash agg |
|---------|-------------|--------------|
| Parallelism unit | Core (few, fat) | CUDA thread (thousands, thin) |
| Atomic contention | Low — few threads touch the same bucket | High — thousands of threads may hash to the same group |
| Memory hierarchy | L1/L2/L3 cache | Shared memory (fast, 48–96 KB/SM) + global memory (slow, high bandwidth) |
| Key equality | Arbitrary comparator | Must be expressed as a device-callable functor |

The central challenge on GPU is **reducing global atomic contention** for the aggregation step while still achieving near-linear throughput over hundreds of millions of rows.

---

## Scope of This Analysis

This document analyses the **`groupby` + `sum` physical operator in NVIDIA's [libcudf](https://github.com/rapidsai/cudf)** — the GPU dataframe library that underpins the RAPIDS ecosystem and is used by cuDF (Python), Spark-RAPIDS, and Dask-cuDF, among others.

Specifically we trace the exact execution path triggered by the following query:

```sql
SELECT   o_orderstatus,
         SUM(o_totalprice) AS total_price
FROM     orders
GROUP BY o_orderstatus;
```

In C++ with libcudf this is expressed as:

```cpp
cudf::table_view tv = cudf_table->view() ... // read from arrow parquet file

// Create GroupBy operator by specifying the `key` column to group on
cudf::groupby::groupby gb(cudf::table_view{{tv.column(src.key_col)}});

// create aggregation for each column, here only 1 SUM agg
cudf::groupby::aggregation_request req;
req.values = tv.column(src.value_col);
req.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

// Aggregate on default stream
auto [result_keys, agg_results] = gb.aggregate({req});
```

The goals are:

- Understand **how libcudf selects and executes the groupby sum path** using a string key and float64 column.
- Map every GPU kernel launch to its source location in cuDF, [cuCollections](https://github.com/NVIDIA/cuCollections), [Thrust](https://github.com/NVIDIA/cccl), and [CUB](https://github.com/NVIDIA/cccl).
- **CApture a real run of the algorithm with real Nsight Systems on GB10** — confirm kernel names, ordering, and timing on a 100M-row workload.
- Identify the dominant performance costs and explain the two-level shared-memory aggregation strategy that libcudf uses to reduce global atomic contention.

## Running the experiment and generating the data

This analysis is based on a 100M-row Parquet file generated by the TPC-H script, meant to reproduce the Orders table. Follow the setup described in the README before running these commands.

**1. Generate 100M rows of TPC-H Orders data:**

```bash
python scripts/make_tpch_orders.py --rows 100000000 --output orders_100M.parquet
```

**2. Run the groupby binary:**

```bash
./build/libcudf_tpch_orders_groupby --input data/orders_100M.parquet
```

**3. Profile with Nsight Systems (timeline + CUDA API + NVTX):**

```bash
mkdir -p reports
nsys profile \
    --trace cuda,osrt,nvtx \
    --output reports/libcudf_groupby_100M \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby --input data/orders_100M.parquet
```

**4. Profile with Nsight Compute (per-kernel hardware counters):**

```bash
ncu --set full \
    -o reports/libcudf_groupby_100M \
    ./build/libcudf_tpch_orders_groupby --input data/orders_100M.parquet
```

**5. RMM allocation trace (CPU call stack per GPU allocation):**

```bash
./build/libcudf_tpch_orders_groupby --input data/orders_100M.parquet --rmm-trace
```

Or via environment variable (useful when the binary is launched under `nsys`/`ncu`):

```bash
RMM_INSTRUMENT=1 nsys profile \
    --trace cuda,osrt,nvtx \
    --output reports/libcudf_groupby_100M \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby --input data/orders_100M.parquet
```

### Libraries under analysis

| Library | Role in this operator |
|---------|----------------------|
| **libcudf** (`rapidsai/cudf`) | Top-level groupby orchestration, kernel launchers, row hashing/equality, result finalization |
| **cuCollections** (`NVIDIA/cccl`) | `cuco::static_set` — the GPU open-addressing hash set at the heart of key deduplication |
| **Thrust** (`NVIDIA/cccl`) | Parallel fill, tabulate, for_each, gather — all groupby support kernels |
| **CUB** (`NVIDIA/cccl`) | Device-wide scan and select (stream compaction inside `retrieve_all`); underlying backend for Thrust |
| **RMM** (`rapidsai/rmm`) | Stream-ordered GPU memory allocation for all intermediate buffers |
| **libcudacxx** (`NVIDIA/cccl`) | `cuda::atomic_ref`, `cuda::std::atomic_flag`, cooperative groups — device-side synchronisation primitives |


---

## Benchmark Dataset

All measurements and call-graph annotations are grounded in a single concrete workload:

| # | Column | Type | Role |
|---|--------|------|------|
| 0 | `id` | `int32` | — |
| 1 | `score` | `float64` | — |
| 2 | `label` | `utf8` (string) | **groupby key** |
| 3 | `active` | `bool` | — |
| 4 | `amount` | `int64` | **SUM target** |
| 5 | `ratio` | `float32` | — |
| 6 | `timestamp` | `timestamp[ms, UTC]` | — |
| 7 | `category` | `dictionary<int8, utf8>` | — |

**100 million rows**, Arrow/Parquet format, already loaded into device memory as a `cudf::table_view`.

Notable properties that affect the implementation path:
- `label` is a **variable-length UTF-8 string** column → row hashing uses MurmurHash3 over raw character bytes; key gather after aggregation requires a CUB prefix-scan over character offsets.
- `amount` is `int64` → no type widening needed; output type stays `int64`; native `atomicAdd` is available.
- Key cardinality is **low relative to row count** → the shared-memory aggregation sub-path is taken (≤ 128 distinct `label` values per CUDA block).

---

## Document Series

This analysis is split into three focused documents:

| Document | Focus |
|----------|-------|
| [Part I — Algorithm Overview](groupby_sum_1_algorithm_overview.md) | High-level description of the hash groupby algorithm: path selection, `cuco::static_set` layout, two-kernel shared-memory strategy, output key gather, complexity summary, and the life of `global_mapping_indices` |
| [Part II — Nsight Analysis](groupby_sum_2_nsight_analysis.md) | Ground-truth GPU kernel trace from an actual Nsight Systems capture: kernel table with exact durations, raw kernel sequence, and performance breakdown by phase |
| [Part III — Code Analysis](groupby_sum_3_code_analysis.md) | Function-by-function call stack from `groupby::aggregate()` down to the atomic `SUM` update, with source file links, RMM allocation table, and library layer summary |

---

## Key Insights

**The hash path dominates for `SUM` on numeric types.** libcudf never considers the sort path when the aggregation has native atomic support (`SUM`, `MIN`, `MAX`, `COUNT`, …). The decision is made at the C++ dispatch layer before any GPU work begins.

**Two kernels do the heavy lifting — and only two.** Despite a trace of thirteen distinct kernel launches, 97% of GPU time is consumed by just two: `mapping_indices_kernel` (7.0 ms, key deduplication + index mapping) and `single_pass_shmem_aggs_kernel` (5.0 ms, two-phase SUM accumulation). Everything else is bookkeeping.

**The `cuco::static_set` initialisation is the second largest cost.** Filling 200M sentinel slots to prepare the hash table takes 4.0 ms — more than the aggregation kernel itself. For workloads where the hash table can be reused across multiple aggregation calls this cost would amortise; in the single-call case it cannot.

**Shared memory is the key to scalability.** Rather than having 100M threads compete on K global atomics, libcudf stages each block's contribution through a private 128-slot shared-memory accumulator. The number of global atomic writes is bounded by `num_blocks × 128`, not by the input row count — a reduction of roughly three orders of magnitude for this dataset.

**`DeviceSelectSweepKernel` (unique-key extraction) is consistently under-estimated.** At 3.5 ms it is the third most expensive step, yet it fires entirely inside `cuco::static_set::retrieve_all()` — a single lib call with no visible cuDF source entry point. It is invisible to call-graph analysis and only shows up in a profiler.

**String keys add a fixed post-aggregation cost, not a per-row cost.** The four CUB/cuDF kernels that gather and copy the output `label` strings (I–L) operate on K unique keys, not 100M rows. For low-cardinality string keys this phase costs ~20 µs regardless of input size.

**libcudf is a four-library system.** A correct mental model of one operator requires understanding cuCollections (hash table), Thrust and CUB (parallel primitives), RMM (memory), and libcudacxx (device-side atomics and synchronisation) — not just the cuDF source itself.
