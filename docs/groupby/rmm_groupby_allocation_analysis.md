# RMM Allocation Analysis — `cudf::groupby::groupby::aggregate`

Analysis of GPU memory allocations captured via `rmm_backtrace_resource_adaptor`
for a groupby on **100 million rows** of TPC-H Orders data
(`GROUP BY o_orderstatus, SUM(o_totalprice)`).

Trace file: `data/rmm_trace_100M.txt`

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Allocations under `groupby::aggregate` | 23 |
| Deallocations under `groupby::aggregate` | 19 |
| Unfreed at return (output columns) | 4 |
| Total bytes allocated | **2,000,503,640 B (~2.0 GB)** |
| Largest single allocation | 800,000,000 B (800 MB) |
| Smallest allocation | 3 B |

---

## 2. Size Distribution

| Size range | Count | Bytes | % of total |
|------------|------:|------:|-----------:|
| < 1 KB (metadata/bookkeeping) | 15 | ~2 KB | < 0.01% |
| 1 KB – 1 MB (working buffers) | 4 | ~503 KB | < 0.01% |
| 100 MB – 1 GB (bulk data) | **4** | **2,000,000,000 B** | **~100%** |

99.97% of all bytes come from just **4 large allocations**.

---

## 3. Allocation Breakdown by Call Path

### Phase 1 — Row equality preprocessing (1 allocation, 143 B)

```
preprocessed_table::create
  └─ table_device_view_base (contiguous_copy_column_device_views)
       └─ rmm::device_buffer  [143 B]
```

Copies the `column_device_view` descriptors for the groupby key column into GPU
memory so the `cuco` hash table comparator can access them from device code.
Cost is proportional to the number of key columns × metadata size — essentially
free.

---

### Phase 2 — Hash table construction (1 allocation, **800 MB**)

```
compute_groupby
  └─ cuco::static_set  [800,000,000 B]
```

The `cuco::static_set<int>` GPU hash table holding all unique group keys.

**Why 800 MB?**
- 100 M rows × 4 bytes/int × 2× over-allocation (default load factor ≈ 0.5) = **800 MB**
- The factor of 2 is mandatory: a hash table at 50% occupancy needs 2× the
  number of slots as entries to keep probe chains short and GPU occupancy high.
- This is the **single largest allocation** — 40% of the total 2.0 GB budget.

---

### Phase 3 — Single-pass aggregation (5 allocations, ~400 MB dominant)

```
compute_single_pass_aggs
  ├─ rmm::device_buffer  [400,000,000 B]   ← partial aggregation buffer
  ├─ rmm::device_buffer  [221,184 B]        ← staging/work buffer
  ├─ rmm::device_buffer  [1,728 B]          ← small work buffer
  ├─ rmm::device_buffer  [4 B]              ← flag/counter
  └─ rmm::device_buffer  [4 B]              ← flag/counter
```

The 400 MB buffer holds one `float64` partial sum per input row
(100 M × 4 bytes = 400 MB) — the running aggregation output slot for each row
before unique group compaction.

The 221 KB and 1.7 KB buffers are staging/temporary buffers used during the
aggregation kernel dispatch.

---

### Phase 4 — Key extraction from the hash table (3 allocations, ~400 MB dominant)

```
extract_populated_keys
  ├─ rmm::device_buffer  [400,000,000 B]   ← output key index vector
  ├─ cuco::retrieve_all temp  [278,783 B]   ← atomic output counter
  └─ cuco::retrieve_all temp  [8 B]         ← device-side size scalar
```

`extract_populated_keys` calls `cuco::retrieve_all` to dump all occupied slots
from the hash table into a flat `device_uvector<int>`.

- **400 MB** = 100 M × 4 bytes: worst-case upper bound (one slot per input row)
  before deduplication. The actual number of unique groups is tiny (3 values of
  `o_orderstatus`), so only the first 3 elements of this vector are meaningful —
  but the full 100 M × 4-byte allocation is made unconditionally because the
  number of unique groups is not known ahead of time.
- The 278 KB buffer is a Thrust temporary used by `exclusive_scan` inside
  `cuco::retrieve_all` to compute output positions.

---

### Phase 5 — Key transform map (1 allocation, **400 MB**)

```
compute_key_transform_map
  └─ rmm::device_buffer  [400,000,000 B]
```

Builds a per-row mapping from each of the 100 M input rows to its assigned
group slot in the result table. Also 100 M × 4 bytes = 400 MB.

---

### Phase 6 — Results table allocation (1 allocation, 24 B)

```
create_results_table → make_numeric_column → make_fixed_width_column
  └─ rmm::device_buffer  [24 B]
```

Allocates the output column for the `SUM(o_totalprice)` aggregation result.
24 bytes = 3 groups × 8 bytes per `float64`. Tiny.

---

### Phase 7 — Output key column gather (11 allocations, small)

```
compute_groupby → cudf::detail::gather (key column materialisation)
  ├─ strings::detail::gather  (o_orderstatus is a string column)
  │    ├─ column_device_view::create   [79 B]
  │    ├─ make_numeric_column (offsets) [16 B]
  │    ├─ device_scalar<long>           [8 B]
  │    ├─ thrust::exclusive_scan temp   [1,279 B]
  │    └─ strings char data             [3 B]   ← 3 chars total ("O", "F", "P")
  │
  └─ gather_bitmask
       ├─ create_null_mask               [64 B]
       ├─ make_device_uvector (ptrs)     [8 B]
       ├─ contiguous_copy_column_views   [143 B]
       └─ make_zeroed_device_uvector     [4 B]
```

Gathers the unique key values (3 rows) from the original 100 M-row string
column using the key indices extracted in Phase 4.

Because `o_orderstatus` is a **string column**, the gather requires:
1. Computing new string offsets via `exclusive_scan` (thrust temp: 1,279 B)
2. A `device_scalar<long>` to hold total output char count on device (8 B)
3. The actual string character buffer — **3 bytes** for the 3 distinct status
   codes (`"O"`, `"F"`, `"P"`)

This entire phase allocates < 2 KB total.

---

## 4. Memory Lifecycle — What Is Freed vs. Retained

Of 23 allocations, **19 are freed** before `groupby::aggregate` returns. The 4
retained allocations are the output result columns:

| Ptr | Bytes | Column |
|-----|------:|--------|
| `0xf7a97ca36c00` | 4 | Aggregation result data (3 × float32?) |
| `0xf7a97ca37000` | 24 | Aggregation result data (3 × float64) |
| `0xf7a97ca00600` | 3 | String char data (`"O" "F" "P"`) |
| `0xf7a97ca00c00` | 4 | Null mask / bitmask for result |

These are the output `cudf::column` buffers returned in the `groupby_result`
table — intentionally not freed.

---

## 5. Key Insights

### 5.1 Peak memory is dominated by 3 equal-sized int32 arrays

Three separate 400 MB allocations all have the same shape: **100 M × 4 bytes**.

| Allocation | Purpose |
|------------|---------|
| `compute_single_pass_aggs` 400 MB | Per-row aggregation accumulator |
| `extract_populated_keys` 400 MB | Per-row key index output from hash table |
| `compute_key_transform_map` 400 MB | Per-row group mapping |

If these three arrays were alive simultaneously the groupby would require
800 MB (hash table) + 3 × 400 MB = **2.0 GB** concurrently.
The RMM pool reuses the backing memory across the 19 deallocations, so the
actual high-water mark is lower than the sum of all allocations.

### 5.2 The hash table load factor drives the largest cost

The `cuco::static_set` is sized at **2× the number of input rows** regardless
of how many unique groups there are. For this workload (100 M rows, 3 unique
groups) the table is almost entirely empty — 99.999997% of slots are unused.
This overhead is inherent to hash-based groupby when the cardinality is
not known ahead of time.

### 5.3 The 100 M × 4-byte key index array is also worst-case

`extract_populated_keys` pre-allocates 400 MB for up to 100 M unique groups.
The actual fill is 3 entries (12 bytes). For low-cardinality groupby this is
a significant source of over-allocation. A two-pass approach (count unique
groups first, then extract) would reduce this to near-zero but would require
an extra pass over the hash table.

### 5.4 String gather is cheap at low output cardinality

The string column gather (Phase 7) allocates < 2 KB total for 3 output rows,
including only 3 bytes of character data. The `exclusive_scan` temp buffer
(1.3 KB) is larger than the actual output. This cost would grow with output
cardinality, not input size.

### 5.5 19 of 23 allocations are freed within the call — good reuse

All large working buffers (hash table, aggregation buffer, key indices) are
freed before returning. The RMM pool recycles these backing pages for
subsequent operations, so the high-water mark measured across the full
workload is not double-counted.

---

## 6. Memory Scaling Estimates

For `N` input rows:

| Buffer | Size |
|--------|------|
| `cuco::static_set` hash table | `2 × N × sizeof(int)` = `8N` bytes |
| Per-row aggregation buffer | `N × sizeof(value_type)` = `4N`–`8N` bytes |
| Key index output vector | `N × sizeof(int)` = `4N` bytes |
| Key transform map | `N × sizeof(int)` = `4N` bytes |
| **Total (approximate)** | **~20–24 × N bytes** |

For 100 M rows: 20 × 100 M = **2.0 GB** — consistent with the observed 2.0 GB.

This means groupby GPU memory usage scales **linearly with input size**,
independent of output cardinality. For 1 B rows you would need ~20 GB of GPU
memory for the groupby working set alone.
