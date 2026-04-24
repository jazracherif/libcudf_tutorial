# libcudf-tutorial Tutorial – groupby sum example

Demonstrates creating an Arrow-backed table in C++ with libcudf and running a `groupby().sum()` reduction.

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
  - [1.0 Clone the repository](#10-clone-the-repository)
  - [1.1 Create the conda environment (first time only)](#11-create-the-conda-environment-first-time-only)
- [2. Build](#2-build)
- [3. Generate sample data](#3-generate-sample-data)
  - [3.1 Basic usage](#31-basic-usage-10-rows--dataordersparquet)
  - [3.2 Options](#32-options)
- [4. Run](#4-run)
  - [4.1 Inline Arrow mode (no arguments)](#41-inline-arrow-mode-no-arguments)
  - [4.2 Parquet mode (--input flag)](#42-parquet-mode---input-flag)
  - [4.3 RMM allocation tracing (--rmm-trace)](#43-rmm-allocation-tracing---rmm-trace)
- [5. Profile with NVIDIA Nsight Compute](#5-profile-with-nvidia-nsight-compute)
  - [5.1 Basic command-line profile](#51-basic-command-line-profile)
  - [5.2 Target specific kernels](#52-target-specific-kernels)
  - [5.3 Useful metric sets](#53-useful-metric-sets)
  - [5.4 Notes](#54-notes)
- [6. Profile with NVIDIA Nsight Systems](#6-profile-with-nvidia-nsight-systems)
  - [6.1 Basic profile (save a report)](#61-basic-profile-save-a-report)
  - [6.2 Trace CUDA API + kernels + memory copies](#62-trace-cuda-api--kernels--memory-copies)
  - [6.3 Print a quick summary to stdout (no GUI needed)](#63-print-a-quick-summary-to-stdout-no-gui-needed)
  - [6.4 Source file and line info in backtraces](#64-source-file-and-line-info-in-backtraces)
  - [6.5 Notes](#65-notes)
  - [6.6 Extract events from a report to CSV](#66-extract-events-from-a-report-to-csv)
  - [6.7 Generate a Host↔Device message flow diagram](#67-generate-a-hostdevice-message-flow-diagram)
  - [6.8 Generate an interactive HTML sequence diagram](#68-generate-an-interactive-html-sequence-diagram)
- [7. Clean](#7-clean)
- [8. Files](#8-files)
- [9. Internals and Documentation](#9-internals-and-documentation)

## 1. Prerequisites

- NVIDIA GPU with CUDA drivers installed
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- A `libcudf-tutorial` conda environment with the Rapids stack

### 1.0 Clone the repository

Use the following command for faster cloning that includes the submodules and explore the code analysis.

`--recurse-submodules` initializes and clones `cccl/`, `cuCollections/`, and `cudf/` in one step.
`--shallow-submodules` fetches only the single commit each submodule is pinned to (`--depth 1`),
skipping their full git history and reducing the download from several GB to a few hundred MB.

```bash
git clone --recurse-submodules --shallow-submodules git@github.com:jazracherif/libcudf_tutorial.git
```

### 1.1 Create the conda environment (first time only)

```bash
conda env create -f environment.yml
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

> Adjust `cudf` and `cuda-version` in `environment.yml` to match your installed CUDA toolkit.

## 2. Build

```bash
conda activate libcudf-tutorial
make
```

This runs CMake with `CMAKE_PREFIX_PATH=$CONDA_PREFIX`, which lets CMake find
cudf, Arrow and RMM automatically, then compiles the binary into `build/`.

## 3. Generate sample data

`scripts/make_tpch_orders.py` creates a typed Parquet (or Arrow IPC) file under `data/`
that can be used as input to `libcudf_tpch_orders_groupby`.  It requires `pyarrow`,
which is included in the `environment.yml` conda environment.

### 3.1 Basic usage (10 rows → `data/orders.parquet`)

```bash
python scripts/make_tpch_orders.py
```

### 3.2 Options

| Flag | Default | Description |
|------|---------|-------------|
| `--rows N` | `10` | Number of rows to generate |
| `--output FILENAME` | `orders.parquet` | Output filename (written inside `./data/`) |
| `--format parquet\|ipc` | `parquet` | Output format: Parquet or Arrow IPC (Feather v2) |

Example — generate 1000 rows into a custom file:

```bash
python scripts/make_tpch_orders.py --rows 1000 --output my_orders.parquet
```

The generated table matches the [TPC-H Orders](https://www.tpc.org/tpch/) schema.
The script prints the schema and a preview of the first five rows:

```
Schema:
o_orderkey: int64
o_custkey: int64
o_orderstatus: string
o_totalprice: double
o_orderdate: date32[day]
o_orderpriority: string
o_clerk: string
o_shippriority: int32
o_comment: string

First 5 rows:
 o_orderkey  o_custkey o_orderstatus  o_totalprice  o_orderdate o_orderpriority             o_clerk  o_shippriority           o_comment
          1          1             F          1.99   2024-01-01        1-URGENT  Clerk#000000001               0   order 1 placed
          2          2             O          2.99   2024-01-01          2-HIGH  Clerk#000000002               0   order 2 placed
 ...

Wrote 10 rows × 9 cols → data/orders.parquet
```

Columns produced (TPC-H Orders schema):

| # | Name | Type | Description | Role |
|---|------|------|-------------|------|
| 0 | `o_orderkey` | `int64` | Unique order identifier | |
| 1 | `o_custkey` | `int64` | FK → Customer table | |
| 2 | `o_orderstatus` | `utf8` | `'F'` fulfilled \| `'O'` open \| `'P'` pending | groupby key |
| 3 | `o_totalprice` | `float64` | Total order value in USD | sum value |
| 4 | `o_orderdate` | `date32` | Date the order was placed | |
| 5 | `o_orderpriority` | `utf8` | `'1-URGENT'` through `'5-LOW'` | |
| 6 | `o_clerk` | `utf8` | Clerk ID, e.g. `Clerk#000000001` | |
| 7 | `o_shippriority` | `int32` | Shipping priority (always 0 in TPC-H) | |
| 8 | `o_comment` | `utf8` | Free-form comment (≤79 chars) | |

## 4. Run

### 4.1 Inline Arrow mode (no arguments)

Builds a small in-process Arrow `RecordBatch` with the two TPC-H Orders
columns used by the groupby — `o_orderstatus` (utf8) as key and
`o_totalprice` (float64) as value — then runs
`groupby(o_orderstatus).sum(o_totalprice)`:

```bash
./build/libcudf_tpch_orders_groupby
```

Expected output:

```
=== Input Arrow RecordBatch ===
o_orderstatus: ["F", "O", "F", "P", "O"]
o_totalprice:  [173665.47, 46929.18, 193846.25, 32151.78, 121200.00]

=== groupby(o_orderstatus).sum(o_totalprice) ===
o_orderstatus:
F
O
P
sum(o_totalprice):
367511.72
168129.18
32151.78
```

### 4.2 Parquet mode (--input flag)

Pass a Parquet file path with `--input` (or `-i`) to load data from disk.
The file is expected to follow the TPC-H Orders schema produced by
`make_tpch_orders.py`.  By default the groupby runs on `o_orderstatus` (key)
and sums `o_totalprice` (value).

```bash
./build/libcudf_tpch_orders_groupby --input data/orders.parquet
```

Override the default column names via the `from_parquet()` arguments in
`libcudf_tpch_orders_groupby.cu` if you want to group by a different column.

Expected output:

```
=== Loaded Parquet: data/orders.parquet (9 cols, N rows) ===
    key_col="o_orderstatus" [2]
    value_col="o_totalprice" [3]
=== groupby(o_orderstatus).sum(o_totalprice) ===
o_orderstatus:
F
O
P
sum(o_totalprice):
<aggregated totals per status>
```

### 4.3 RMM allocation tracing (--rmm-trace)

On the **DGX Spark (GB10, ARM/SBSA)** platform, `nsys --cudabacktrace` and
`--sample cpu` are unavailable because the Linux kernel on this machine has
CPU sampling disabled.  This means Nsight Systems cannot capture the host-side
call stack that triggered each CUDA kernel launch or memory allocation.

As a workaround, this binary includes a custom RMM memory resource adaptor
(`rmm_backtrace_resource_adaptor.hpp`) that intercepts every GPU allocation at
the C++ level and captures the call stack directly using `backtrace()` /
`backtrace_symbols()`.  This works independently of the kernel profiling
subsystem and requires no special permissions.

Pass `--rmm-trace` to install two RMM memory resource adaptors that
instrument every GPU allocation and deallocation:

- **stdout** — a demangled CPU call stack showing *who* triggered the allocation
- **`rmm_alloc_log.csv`** — a CSV log with timestamp, pointer, size, and stream for every event

```bash
./build/libcudf_tpch_orders_groupby --input data/orders.parquet --rmm-trace
```

Or enable via environment variable (useful when running under `nsys` or `ncu`
without modifying the profiler command):

```bash
RMM_INSTRUMENT=1 ./build/libcudf_tpch_orders_groupby --input data/orders.parquet
```

Sample output per allocation:

```
[RMM] allocate    ptr=0xe62767a00000  bytes=400000000
  #0  cudf::detail::make_device_uvector_async<...>(...)  (+0x...)
  #1  cudf::io::parquet::detail::decompress_page_data(...)
  #2  ...
```

> **Why some frames show as `libcudf.so(+0xADDRESS)`**
>
> `backtrace_symbols()` resolves names solely from the **dynamic symbol table**
> (`.dynsym`) of each loaded shared library — it calls `dladdr()` internally and
> does not read the static `.symtab`.  The conda release build of `libcudf.so`
> strips private symbols from `.dynsym`, so only exported public API functions
> are resolvable.
>
> | Symbol type | Resolved? | Reason |
> |-------------|-----------|--------|
> | Exported public API (`cudf::groupby::aggregate`, etc.) | ✅ Yes | Present in `.dynsym` — required for dynamic linking |
> | Internal implementation functions | ❌ No — shows `libcudf.so(+0xOFFSET)` | Hidden visibility or not exported; stripped from `.dynsym` |
> | Your own binary's functions | ✅ Yes (if linked with `-rdynamic`) | `-rdynamic` adds all symbols to `.dynsym` |
> | Functions in a debug build | ✅ Yes | Full `.symtab` present; `addr2line` can further resolve to `file:line` |
>
> The frames that *are* resolved — `cudf::groupby::aggregate →
> dispatch_aggregation → detail::hash::groupby` — are exactly the ones that
> matter: they are part of libcudf's public C++ API.  The unresolved frames
> before them are internal RMM pool management functions that were intentionally
> not exported.
>
> To resolve all frames:
> - Build libcudf from source with `-DCMAKE_BUILD_TYPE=Debug` (`.symtab` intact) and then pipe the addresses through `addr2line`
> - Or pass `-rdynamic` to your own binary's link step (only helps for symbols in your binary, not in `libcudf.so`)

## 5. Profile with NVIDIA Nsight Compute

[Nsight Compute](https://developer.nvidia.com/nsight-compute) (`ncu`) profiles individual CUDA kernels with hardware performance counters.

### 5.1 Basic command-line profile

Collect all default metrics and save a report file:

```bash
ncu --set full \
    -o reports/libcudf_groupby \
    ./build/libcudf_tpch_orders_groupby --input ./data/orders.parquet
```

> `--set full` collects the complete default metric set.  Substitute `--set basic` for a faster, lighter collection.

### 5.2 Target specific kernels

libcudf launches many internal kernels.  Narrow the profile to kernels whose names match a pattern:

```bash
# Profile only groupby-related kernels
ncu --kernel-name-base function \
    --kernel-name "groupby\|reduce\|hash" \
    --set full \
    -o reports/libcudf_groupby \
    ./build/libcudf_tpch_orders_groupby --input ./data/orders.parquet
```

### 5.3 Useful metric sets

| Flag | What it collects |
|------|-----------------|
| `--set basic` | Achieved occupancy, memory throughput, SM efficiency |
| `--set full` | All default counters across compute, memory, scheduler |
| `--section MemoryWorkloadAnalysis` | L1/L2/HBM bandwidth and cache hit rates |
| `--section ComputeWorkloadAnalysis` | Pipe utilization, warp stalls |
| `--section SpeedOfLight` | Achieved vs peak compute and memory throughput |

Example — collect only memory and roofline sections:

```bash
ncu --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    -o reports/libcudf_mem \
    ./build/libcudf_tpch_orders_groupby --input ./data/orders.parquet
```

### 5.4 Notes

- Profiling requires root or `CAP_SYS_ADMIN` (or `/proc/sys/kernel/perf_event_paranoid ≤ 2`).  If you see a permissions error:
  ```bash
  sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'
  ```
- Create the output directory first: `mkdir -p reports`
- The binary must be built in `Release` mode (the default) for meaningful results; debug builds add overhead.

## 6. Profile with NVIDIA Nsight Systems

[Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`) gives a system-wide timeline — CPU threads, CUDA API calls, kernel launches, memory copies, and NVTX annotations — making it the right first step before diving into per-kernel metrics with Nsight Compute.

### 6.1 Basic profile (save a report)

```bash
mkdir -p reports
nsys profile \
    --output reports/libcudf_groupby \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby --input ./data/orders.parquet
```

This produces `reports/libcudf_groupby.nsys-rep`.  Open it in the Nsight Systems GUI:

```bash
nsys-ui reports/libcudf_groupby.nsys-rep
```

Or on a remote machine, copy the `.nsys-rep` file locally and open it with the Nsight Systems desktop app.

### 6.2 Trace CUDA API + kernels + memory copies

```bash
nsys profile \
    --trace cuda,osrt,nvtx \
    --output reports/libcudf_groupby \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby --input ./data/orders.parquet
```

| `--trace` value | What is recorded |
|-----------------|-----------------|
| `cuda` | CUDA API calls, kernel launches, memcpy/memset |
| `osrt` | OS runtime: pthreads, semaphores, signals |
| `nvtx` | User-defined NVTX ranges (if any) |
| `cudnn` | cuDNN API calls |

Print a summary of all NVTX ranges by name and total duration (no GUI needed):

```bash
nsys stats --report nvtx_pushpop_sum reports/libcudf_groupby.nsys-rep
```

libcudf uses the push/pop NVTX API, so `nvtx_pushpop_*` to show api calls in nsigth, see some choices here:

| Report | Output style | What it shows |
|--------|-------------|---------------|
| `nvtx_pushpop_sum` | Aggregated | One row per unique range name: total time, call count, min/avg/max duration — best for identifying slow phases |
| `nvtx_pushpop_trace` | Per-instance | One row per individual range invocation with start time, end time, duration and thread ID — useful for spotting outliers or variance across calls |
| `nvtx_startend_sum` | Aggregated | Same as above but for ranges created with the start/end API (used by PyTorch autograd, user code spanning threads) |
| `nvtx_sum` | Aggregated | Combined view merging both push/pop and start/end ranges into one table |

### 6.3 Print a quick summary to stdout (no GUI needed)

```bash
nsys profile \
    --stats true \
    --output reports/libcudf_groupby \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby --input ./data/orders.parquet
```

`--stats true` prints a text table of top kernels, API calls, and memory operations after the run.

### 6.4 Source file and line info in backtraces

To resolve kernel launches and CPU calls back to file names and line numbers,
first rebuild with debug symbols (already enabled in `CMakeLists.txt` via
`--generate-line-info` for CUDA and `-g` for C++):

```bash
make clean && make
```

Then profile with CPU sampling and DWARF unwinding:

```bash
nsys profile \
    --trace cuda,nvtx,osrt \
    --sample cpu \
    --cpuctxsw system-wide \
    --backtrace dwarf \
    --cuda-memory-usage true \
    --output reports/libcudf_groupby \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby --input ./data/orders.parquet
```

| Flag | Effect |
|------|--------|
| `--sample cpu` | Periodic CPU call-stack samples — shows which host-side code drives kernel launches |
| `--backtrace dwarf` | Resolves stack frames to file name and line number using DWARF debug info |
| `--cpuctxsw system-wide` | Captures thread context switches system-wide for scheduling visibility |

In `nsys-ui`:
- Click any kernel in the timeline → **Source** tab in the bottom panel shows the CUDA call stack with `file:line` resolved
- The **CPU Sampling** row shows a flame graph that maps to your source lines in `libcudf_tpch_orders_groupby.cu`

> Line info for libcudf's own internal kernels resolves only if the Rapids conda package was built with `-lineinfo` (the `rapidsai` channel release builds typically are).

### 6.5 Notes

- `nsys` does **not** require elevated permissions for basic CUDA tracing.
- Use Nsight Systems first to find which kernels take the most time, then drill into them with `ncu`.
- The `.nsys-rep` file can be opened on any machine with the Nsight Systems GUI installed (no GPU required for viewing).

### 6.6 Extract events from a report to CSV

`scripts/extract_nsys_events.py` converts a `.nsys-rep` report into a
time-ordered CSV scoped to a named NVTX region.  It exports the report to
SQLite via `nsys export` and then queries the CUPTI activity tables directly.

#### Basic usage

```bash
python scripts/extract_nsys_events.py reports/libcudf_groupby.nsys-rep
```

Scope to a specific NVTX label (default: `aggregate`):

```bash
python scripts/extract_nsys_events.py \
    reports/libcudf_groupby.nsys-rep \
    docs/groupby/logs/nsight_aggregate.csv \
    --nvtx-label "libcudf:aggregate"
```

Scope to multiple NVTX labels at once (all events land in a single CSV; the
`NVTXLabel` column identifies which region each row came from):

```bash
python scripts/extract_nsys_events.py \
    reports/libcudf_groupby.nsys-rep \
    docs/groupby/logs/nsight_multi.csv \
    --nvtx-label "libcudf:aggregate" "libcudf:merge"
```

Extract all events in the report (ignore NVTX scoping):

```bash
python scripts/extract_nsys_events.py reports/libcudf_groupby.nsys-rep --all
```

Include memory allocation/deallocation rows in the output (suppressed by default;
their `Bytes` values are always propagated to the matching CUDA API rows via `corrId`):

```bash
python scripts/extract_nsys_events.py reports/libcudf_groupby.nsys-rep \
    --include-memory
```

Include 1-byte `__device__` global variable registrations (filtered by default):

```bash
python scripts/extract_nsys_events.py reports/libcudf_groupby.nsys-rep \
    --include-device-static
```

#### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--nvtx-label LABEL [LABEL …]` | `libcudf:aggregate` | One or more NVTX push/pop region labels to scope the output. Each label's events appear in the same CSV with a `NVTXLabel` column. Tries the full label first, then the suffix after the last `:`. |
| `--all` | off | Ignore NVTX scoping; extract every event in the report. |
| `--include-memory` | off | Emit memory allocation/deallocation rows in the CSV. |
| `--include-device-static` | off | Include `CUDA_MEMOPR_MEMORY_KIND_DEVICE_STATIC` events (1-byte `__device__` globals from libcudacxx CPOs). |
| `--keep-sqlite` | off | Keep the intermediate `.sqlite` file (re-export is slow; the file is reused automatically on subsequent runs). |

#### Output columns

| Column | Description |
|--------|-------------|
| `Start_ns` | Event start time in nanoseconds (sort key) |
| `NVTXLabel` | The NVTX region label this event was captured under (empty when `--all` is used) |
| `Type` | `kernel`, `cuda api`, `memcpy`, `memset`, or `memory` |
| `Namespace` | For kernel rows: the C++ namespace prefix (e.g. `cub::detail::scan`) |
| `Name` | Kernel name or CUDA API name (version suffix stripped) |
| `APIVersion` | Version suffix extracted from the CUDA API name (e.g. `v3020`, `v7000`) |
| `ShortName` | Short / unqualified name as reported by Nsight |
| `Start` | Human-readable start time (e.g. `1.720762256s`) |
| `Duration` | Human-readable duration (e.g. `4.105 ms`) |
| `Duration_ns` | Duration in nanoseconds |
| `Bytes` | Bytes transferred or allocated; for `cuda api` rows this is propagated from the matching memory event via `corrId` |
| `Grid` | Kernel grid dimensions (e.g. `390625,1,1`) |
| `Block` | Kernel block dimensions (e.g. `256,1,1`) |
| `Stream` | CUDA stream ID |
| `Device` | GPU index (e.g. `GPU 0`) |
| `TID` | Host thread ID (CUDA API rows only) |
| `ExtraDetail` | Shared-memory sizes, registers per thread, `corrId`, etc. |
| `FullName` | Full demangled kernel signature (kernel rows only) |

### 6.7 Generate a Host↔Device message flow diagram

`scripts/csv_to_mermaid_flow.py` reads a CSV produced by `extract_nsys_events.py`
and writes a Markdown file containing a Mermaid `sequenceDiagram` that shows
every event as an arrow between **Host** and **Device**.

```bash
python scripts/csv_to_mermaid_flow.py \
    docs/groupby/logs/nsight_100m_aggregate.csv \
    docs/groupby/logs/nsight_100m_aggregate_flow.md
```

Limit to the first 50 events and only show kernels + memcpy:

```bash
python scripts/csv_to_mermaid_flow.py nsight.csv flow.md \
    --max-rows 50 --types kernel memcpy
```

#### Arrow conventions

| Arrow | Style | Meaning |
|-------|-------|---------|
| `H->>D` | solid | Blocking host→device call (`cudaMalloc`, `cudaFree`, actual `Memcpy`) |
| `H-->>D` | dashed | Async host→device call (`cudaLaunchKernel`, `cudaMemcpyAsync`, kernel execution) |
| `D->>H` | solid | Device signals host after blocking wait (`cudaStreamSynchronize`) |
| `D->>D` | solid | Device-to-device memcpy |
| `Note over H,D` | note | Memory allocation/deallocation event |
| `H->>H` | solid self | Host-only driver call (only with `--include-host-only`) |

#### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--title TEXT` | filename stem | Markdown heading and diagram title |
| `--max-rows N` | unlimited | Truncate to first N events (warns if > 200) |
| `--types TYPE …` | all | Filter to `kernel`, `cuda api`, `memcpy`, `memset`, `memory` |
| `--include-host-only` | off | Show host-only driver calls (`cuLibraryLoadData`, `cuKernelGetName`, etc.) as self-arrows |
| `--no-duration` | off | Omit duration from edge labels |
| `--no-bytes` | off | Omit byte counts from edge labels |

### 6.8 Generate an interactive HTML sequence diagram

`scripts/csv_to_nsight_html.py` reads the same CSV produced by
`extract_nsys_events.py` and writes a **fully self-contained HTML file** —
no server, no external assets — containing an inline SVG sequence diagram with
an interactive detail panel.

#### Basic usage

```bash
python scripts/csv_to_nsight_html.py \
    docs/groupby/logs/nsight_aggregate.csv \
    docs/groupby/logs/nsight_aggregate_flow.html
```

Add a custom title:

```bash
python scripts/csv_to_nsight_html.py \
    docs/groupby/logs/nsight_aggregate.csv \
    docs/groupby/logs/nsight_aggregate_flow.html \
    --title "libcudf groupby — 100M rows — Nsight Trace"
```

Also emit host-only driver calls (`cuLibraryLoadData`, `cuGetProcAddress`, …)
as note boxes on the Host lifeline:

```bash
python scripts/csv_to_nsight_html.py nsight.csv out.html --include-host-only
```

#### Layout

The page is divided into two columns:

| Column | Content |
|--------|---------|
| **Diagram** (left, scrollable) | Sticky participant header row (`Host (CPU)` / `Device (GPU)`) + one SVG row per event |
| **Detail panel** (right, 420 px fixed) | Legend + a two-column key/value table populated when any row is clicked |

Each SVG row shows:

- **Timeline** column — cumulative offset from the first event and the delta from the previous row.
- **Arrow** (CUDA API calls, Memcpy, Memset) — solid for blocking calls, dashed for async.
- **Amber box** (sync points such as `cudaStreamSynchronize`) — centred on the Host lifeline.
- **Blue box** (kernel executions) — centred on the Device lifeline.
- **Grey spanning box** (memory allocation/deallocation events) — spans both lifelines.

Click any row to populate the detail panel with all CSV fields for that event.

#### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `OUTPUT` | `<input stem>_flow.html` | Path for the generated HTML file |
| `--title TEXT` | filename stem | `<h2>` heading at the top of the diagram column |
| `--include-host-only` | off | Include host-only driver calls as note boxes on the Host lifeline |

## 7. Clean

```bash
make clean
```

## 8. Files

| File | Description |
|------|-------------|
| `src/libcudf_tpch_orders_groupby.cu` | Main C++/CUDA source |
| `include/rmm_backtrace_resource_adaptor.hpp` | Custom RMM MR adaptor that prints a demangled call stack on every alloc/dealloc |
| `scripts/make_tpch_orders.py` | Python script to generate TPC-H Orders Parquet/IPC data |
| `scripts/extract_nsys_events.py` | Extract kernels, CUDA API calls, and memory events from a `.nsys-rep` report into a time-ordered CSV, scoped to an NVTX region |
| `scripts/csv_to_mermaid_flow.py` | Convert the events CSV into a Markdown file with a Mermaid `sequenceDiagram` showing the Host↔Device message flow |
| `scripts/csv_to_nsight_html.py` | Convert the events CSV into a fully self-contained interactive HTML sequence diagram with a sticky header, scrollable event rows, and a click-to-detail right panel |
| `CMakeLists.txt`      | CMake build definition |
| `Makefile`            | Thin wrapper around CMake |
| `environment.yml`     | Conda environment definition |
| `docs/` | Technical deep-dives on adaptor internals (see below) |

## 9. Internals and Documentation

### Background — why this adaptor exists

On the **DGX Spark (GB10, ARM/SBSA)** platform the Linux kernel has CPU
sampling disabled, which means `nsys --sample cpu` and `nsys --cudabacktrace`
are both unavailable.  Without CPU sampling, Nsight Systems cannot show the
host-side call stack that triggered each GPU kernel launch or memory
allocation.

`rmm_backtrace_resource_adaptor` is a workaround: it wraps any RMM memory
resource and, on every `allocate` / `deallocate`, calls `backtrace()` directly
in-process to capture the current CPU call stack — no kernel profiling
subsystem, no special permissions required.  It then resolves each raw frame
address to a demangled symbol name and prints the result to stdout.

> **Performance caveat** — `backtrace()` walks the call stack on every
> allocation and deallocation, which adds overhead proportional to stack depth.
> The first time a unique frame address is seen, resolution may fork an
> `addr2line` subprocess; subsequent hits are served from an in-memory cache at
> negligible cost.  Net effect: the adaptor noticeably slows down
> allocation-heavy workloads and should be used **only during profiling/debugging**,
> not in production code paths.

### Symbol resolution in `rmm_backtrace_resource_adaptor`

[`docs/backtrace_symbol_resolution.md`](docs/backtrace_symbol_resolution.md)
explains in detail how the RMM backtrace adaptor resolves raw instruction
addresses into human-readable function names:

- ELF binary structure (`.dynsym` vs `.symtab`, DWARF)
- ASLR and load-bias arithmetic (`virtual_addr - dli_fbase`)
- APIs used: `backtrace()`, `backtrace_symbols()`, `dladdr()`, `addr2line`
- The full 4-step resolution pipeline implemented in `addr2line_resolve()`:
  1. `dladdr()` `dli_sname` — fast in-process `.dynsym` lookup (no subprocess)
  2. `addr2line` subprocess — reads `.symtab` for private/internal symbols
  3. `backtrace_symbols_resolve()` fallback — for stripped `.so` files
  4. Raw offset string — last resort
- Per-address result cache (`unordered_map` + `mutex`)
- Why some frames still show as `lib.so(+0xOFFSET)` and how to fix them