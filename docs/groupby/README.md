# Inside libcudf: GPU acceleration of Relational Primitives - A Deep Technical Dive

Traditional database execution engines were designed for the CPU: optimized for a handful of powerful cores, deep cache hierarchies, and sequential or lightly-vectorized processing. However, as data volumes grow and CPU frequency scaling plateaus, Database researchers have increasingly turned to hardware acceleration. GPUs, with their massive memory bandwidth and thousands of parallel execution units, offer a highly compelling paradigm shift for analytical query performance.

But mapping relational algebra onto GPUs introduces a massive semantic gap. Operators like **joins**, **aggregations**, and **sorts** must be entirely reimagined for the GPU SIMT (Single Instruction, Multiple Thread) architecture. Conventional algorithms natively optimized for CPUs often hit brutal bottlenecks on GPUs due to **thread divergence**, **uncoalesced memory access**, and severe penalties for **global synchronisation**. 

To bridge this runtime gap, NVIDIA developed **libcudf**: a C++ library implementing foundational DataFrame operations and relational primitives natively on the GPU. It has emerged as the de facto execution framework for a massive portion of the accelerated data ecosystem, underpinning projects like `Spark RAPIDS`, `Dask-cuDF`, `Velox CuDF` and numerous independent database research efforts.

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

In a query execution engine the `GROUP BY` physical operator must solve two logical subproblems:

1. **Key partitioning** — determine, for every input row, which output group it belongs to. This is effectively a dictionary-encoding problem: map an arbitrarily-typed key (integer, string, composite) to a dense integer group-id in [0, K) where K is the number of distinct keys.

2. **Aggregation** — reduce all rows assigned to the same group-id to a single scalar per aggregate column (e.g., sum all values from column C for group-id 3).

These two subproblems are algorithm-agnostic: the same logical goals can be achieved via two fundamentally different physical strategies. The **sort-aggregate** approach sorts all rows by key first, after which identical keys are contiguous and can be reduced in a single scan; comparison sort costs $O(n \log n)$, while radix sort can be linear for fixed-width keys. The **hash-aggregate** approach builds a hash table mapping each distinct key to its running accumulator, updating it in expected $O(n)$ time — no sort required, but concurrent writes to shared buckets introduce contention. If the data exceeds available memory, both strategies must be adapted to process it in chunks.

Both strategies exist on GPU, but porting either from CPU to GPU introduces hardware constraints that shape the implementation. The table below lists the most important ones, ordered by impact:

| # | Concern | CPU implementation | GPU implementation |
|---|---------|--------------------|--------------------|
| 1 | Parallelism unit | A few powerful cores, each with branch prediction, out-of-order execution, and large private caches | Many simple CUDA threads; throughput comes from keeping many warps resident and ready to run, not from making each thread fast |
| 2 | Memory hierarchy & performance model | L1/L2/L3 caches optimise for latency and are mostly managed by hardware | Fast on-chip shared memory must be managed explicitly; global memory is high-bandwidth but high-latency, so performance depends on coalesced access and latency hiding across warps |
| 3 | Sort cost (sort-aggregate) | Comparison sort is mature and supports arbitrary key types, but remains $O(n \log n)$ | Radix sort is very fast for fixed-width keys; variable-length strings require indirect comparison/gather work, so sort-aggregate becomes less attractive for this workload |
| 4 | Atomic contention | A small number of cores contend for shared hash buckets | Thousands of threads may update the same group; atomic read-modify-write operations then serialize, becoming the main hash-aggregate bottleneck |
| 5 | Warp divergence | Each core follows its own instruction stream | Threads execute in groups of 32 (*warps*). If threads probe different numbers of hash slots, the warp waits for the slowest lane, reducing effective parallelism |
| 6 | Capacity planning | Dynamic data structures can grow incrementally | Device memory must be provisioned before kernels run. Unknown output sizes require conservative over-allocation or an extra counting pass and host-side allocation before launching the next kernel; this is often more painful for joins than for groupby |
| 7 | Synchronization model | Lock-based and lock-free data structures are both practical | GPU algorithms rely on hardware atomics and lock-free patterns; for variable-length keys there is no native atomic update, so the algorithm must separate key comparison from fixed-width index/accumulator updates |
| 8 | Key equality | Comparators can call arbitrary host code | Comparators must be device-callable (`__device__`) and cannot use virtual dispatch or call back to the CPU; strings and composite keys need custom on-device equality logic |
| 9 | Out-of-memory handling | Database engines can spill partitions or runs to disk when RAM is insufficient | libcudf does not implement operator-level spilling: the hash table, sort buffers, and outputs must fit in GPU memory or the operation fails |

The central challenges on GPU are **managing atomic contention and warp divergence** during aggregation, **explicitly controlling the memory hierarchy** to maximise bandwidth, and making all key-comparison and synchronisation logic expressible entirely on-device — while still achieving near-linear throughput over hundreds of millions of rows.

In this blog, I break down how libcudf solves these challenges.

---

## Scope of This Analysis

This document analyses the **`groupby` + `sum` physical operator in NVIDIA's [libcudf](https://github.com/rapidsai/cudf)** — the GPU dataframe library that underpins the RAPIDS ecosystem and is used by cuDF (Python), Spark-RAPIDS, and Dask-cuDF, among others. The analysis is grounded in a **100 million row workload** (1.8 GB Parquet file) run on the DGX Spark workstation, featuring a the GB10 blackwhell with 128GB LPDDRX unified memory with host Arm Cpus.

Specifically we trace the exact execution path triggered by the following query:

```sql
SELECT   o_orderstatus,
         SUM(o_totalprice) AS total_price
FROM     orders
GROUP BY o_orderstatus;
```

The below libcudf C++ code is invoked on an ingested table stored in the Apach Arrow format:

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
- **Capture a real run of the algorithm with real Nsight Systems on GB10** — confirm kernel names, ordering, and timing on a 100M-row workload.
- Identify the dominant performance costs and explain the two-level shared-memory aggregation strategy that libcudf uses to reduce global atomic contention.

## Running the experiment and generating the data

This analysis is based on a 100M-row Parquet file generated by the TPC-H script, meant to reproduce the Orders table. Follow the setup described in the README before running these commands.

**0. Activate the conda environment:**

```bash
conda activate libcudf-tutorial
```

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

**5. View the RMM allocation trace (CPU call stack per GPU allocation):**

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

**6. Extract CUDA events from the Nsight report to CSV:**

```bash
python scripts/extract_nsys_events.py \
    reports/libcudf_groupby_100M.nsys-rep \
    docs/groupby/logs/libcudf_groupby_orders_100M_libcudf_aggregate.csv \
    --nvtx-label "libcudf:aggregate"
```

**7. Generate the Nsight timeline HTML report:**

```bash
python scripts/csv_to_nsight_html.py \
    docs/groupby/logs/libcudf_groupby_orders_100M_libcudf_aggregate.csv \
    docs/groupby/logs/nsight_100m_aggregate_timeline.html
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
| 0 | `o_orderkey` | `int64` | unique order identifier |
| 1 | `o_custkey` | `int64` | FK → Customer table |
| 2 | `o_orderstatus` | `utf8` (string) | **groupby key** — `'F'` / `'O'` / `'P'` |
| 3 | `o_totalprice` | `float64` | **SUM target** — total monetary value |
| 4 | `o_orderdate` | `date32` | date the order was placed |
| 5 | `o_orderpriority` | `utf8` | `'1-URGENT'` … `'5-LOW'` |
| 6 | `o_clerk` | `utf8` | clerk who processed the order |
| 7 | `o_shippriority` | `int32` | shipping priority (0 = normal) |
| 8 | `o_comment` | `utf8` | free-form comment (≤79 chars) |

**100 million rows**, Arrow/Parquet format, already loaded into device memory as a `cudf::table_view`.

Notable properties that affect the implementation path:
- `o_orderstatus` is a **variable-length UTF-8 string** column → row hashing uses MurmurHash3 over raw character bytes; key gather after aggregation requires a CUB prefix-scan over character offsets.
- `o_totalprice` is `float64` → output type stays `float64`; native `atomicAdd` is available.
- Key cardinality is **low relative to row count** (only 3 distinct values: `'F'`, `'O'`, `'P'`) → the shared-memory aggregation sub-path is taken (≤ 128 distinct `o_orderstatus` values per CUDA block).

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

**Four kernels account for all meaningful GPU work.** Despite a trace of thirteen distinct kernel launches, four kernels together consume ~17.5 ms — essentially all GPU kernel time: `single_pass_shmem_aggs_kernel` (5.119 ms, two-phase SUM accumulation), `mapping_indices_kernel` (4.784 ms, key deduplication + index mapping), `cuco::static_set` initialisation (4.105 ms, sentinel-fill of 200M slots), and `DeviceSelectSweepKernel` (3.439 ms, unique-key stream compaction). Everything else is bookkeeping at the µs scale.

**The `cuco::static_set` initialisation is the third most expensive kernel.** Filling 200M sentinel slots to prepare the hash table takes 4.105 ms — behind only the aggregation and insert kernels. For workloads where the hash table can be reused across multiple aggregation calls this cost would amortise; in the single-call case it cannot.

**Shared memory is the key to scalability.** Rather than having 100M threads compete on K global atomics, libcudf stages each block's contribution through a private 128-slot shared-memory accumulator. The number of global atomic writes is bounded by `num_blocks × 128`, not by the input row count — a reduction of roughly three orders of magnitude for this dataset.

**`DeviceSelectSweepKernel` (unique-key extraction) is consistently under-estimated.** At 3.439 ms it is the fourth most expensive kernel, yet it fires entirely inside `cuco::static_set::retrieve_all()` — a single lib call with no visible cuDF source entry point. It is invisible to call-graph analysis and only shows up in a profiler.

**String keys add a fixed post-aggregation cost, not a per-row cost.** The four CUB/cuDF kernels that gather and copy the output `label` strings (I–L) operate on K unique keys, not 100M rows. For low-cardinality string keys this phase costs ~20 µs regardless of input size.

**libcudf is a four-library system.** A correct mental model of one operator requires understanding cuCollections (hash table), Thrust and CUB (parallel primitives), RMM (memory), and libcudacxx (device-side atomics and synchronisation) — not just the cuDF source itself.
