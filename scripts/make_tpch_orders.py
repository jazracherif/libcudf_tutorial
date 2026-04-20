"""
make_tpch_orders.py

Creates an Arrow table matching the TPC-H Orders schema and writes it to a
Parquet file (default: orders.parquet). The resulting file can be read back
in C++ with libcudf via cudf::io::read_parquet().

Supports generating very large files (e.g. 1B rows) by writing in batches
so that memory usage stays bounded regardless of total row count.

Example business meaning:
    Each row represents one order from the TPC-H benchmark Orders table.
    See: https://www.tpc.org/tpch/

Columns produced (matches TPC-H Orders schema):
    o_orderkey       : int64    — unique order identifier
    o_custkey        : int64    — foreign key to Customer table
    o_orderstatus    : utf8     — single char: 'F' (fulfilled), 'O' (open), 'P' (pending)
    o_totalprice     : float64  — total monetary value of the order
    o_orderdate      : date32   — date the order was placed
    o_orderpriority  : utf8     — priority class: '1-URGENT' … '5-LOW'
    o_clerk          : utf8     — clerk who processed the order, e.g. 'Clerk#000000001'
    o_shippriority   : int32    — shipping priority (0 = normal in TPC-H)
    o_comment        : utf8     — free-form comment field (up to 79 chars)
"""

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Fixed pools used for low-cardinality columns in the TPC-H Orders example.
_ORDER_STATUSES   = np.array(["F", "O", "P"], dtype=object)
_ORDER_PRIORITIES = np.array(["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"], dtype=object)

# Epoch offset in milliseconds (2024-01-01 UTC)
_BASE_TS_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

SCHEMA = pa.schema([
    pa.field("o_orderkey",      pa.int64()),    # unique order identifier
    pa.field("o_custkey",       pa.int64()),    # FK → Customer.c_custkey
    pa.field("o_orderstatus",   pa.utf8()),     # 'F' fulfilled | 'O' open | 'P' pending
    pa.field("o_totalprice",    pa.float64()),  # total price of the order in USD
    pa.field("o_orderdate",     pa.date32()),   # date the order was placed
    pa.field("o_orderpriority", pa.utf8()),     # '1-URGENT' through '5-LOW'
    pa.field("o_clerk",         pa.utf8()),     # clerk ID, e.g. 'Clerk#000000001'
    pa.field("o_shippriority",  pa.int32()),    # shipping priority (0 in standard TPC-H)
    pa.field("o_comment",       pa.utf8()),     # free-form comment (≤79 chars)
])


def make_batch(start: int, size: int) -> pa.RecordBatch:
    """Build one Arrow RecordBatch covering rows [start, start + size).

    All column values are deterministically derived from the row index so that
    batches can be generated independently (useful for parallel or resumable
    writes) and still produce consistent data across runs.

    Args:
        start: 0-based index of the first row in this batch.
        size:  Number of rows to generate.

    Returns:
        An Arrow RecordBatch with shape (size, 9) conforming to SCHEMA.
    """
    idx = np.arange(start, start + size, dtype=np.int64)

    orderkeys     = pa.array((idx + 1), type=pa.int64())   # sequential, 1-based
    custkeys      = pa.array((idx % 150_000 + 1), type=pa.int64())   # synthetic FK, cycles through 150k customers (TPC-H SF=1)
    statuses      = pa.array(_ORDER_STATUSES[idx % 3], type=pa.utf8())    # 'F', 'O', or 'P' in round-robin
    total_prices  = pa.array(np.round((idx % 499_999 + 1).astype(np.float64) + 0.99, 2), type=pa.float64())  # synthetic value in [1.00, 500_000.00]
    order_dates   = pa.array(((_BASE_TS_MS + idx * 3_600_000) // 86_400_000).astype(np.int32), type=pa.date32())  # synthetic value in [1.00, 500_000.00]
    priorities    = pa.array(_ORDER_PRIORITIES[idx % 5], type=pa.utf8())    # cycles through 5 priority levels
    clerk_ids     = pa.array([f"Clerk#{(i % 1_000) + 1:09d}" for i in idx.tolist()], type=pa.utf8())  # synthetic value in [1.00, 500_000.00]
    ship_priority = pa.array(np.zeros(size, dtype=np.int32), type=pa.int32())   # always 0 in standard TPC-H
    comments      = pa.array([f"order {k} placed" for k in (idx + 1).tolist()], type=pa.utf8()) # short synthetic comment referencing order key

    return pa.record_batch(
        [orderkeys, custkeys, statuses, total_prices, order_dates,
         priorities, clerk_ids, ship_priority, comments],
        schema=SCHEMA,
    )


def write_parquet(path: str, n_rows: int, batch_size: int) -> None:
    """Write n_rows of TPC-H Orders data to a Parquet file.

    Rows are generated and flushed in chunks of `batch_size` so that peak
    memory usage is O(batch_size) rather than O(n_rows).  Progress is printed
    to stdout for runs larger than 10 million rows.

    Args:
        path:       Absolute or relative path of the output .parquet file.
        n_rows:     Total number of rows to write.
        batch_size: Rows per in-memory batch (controls peak RAM usage).
    """
    with pq.ParquetWriter(path, SCHEMA) as writer:
        written = 0
        while written < n_rows:
            chunk = min(batch_size, n_rows - written)
            writer.write_batch(make_batch(written, chunk))
            written += chunk
            if n_rows >= 10_000_000:
                pct = written / n_rows * 100
                print(f"\r  {written:,} / {n_rows:,} rows ({pct:.1f}%)", end="", flush=True)
    if n_rows >= 10_000_000:
        print()
    print(f"Wrote {n_rows:,} rows × {len(SCHEMA)} cols → {path}")


def write_ipc(path: str, n_rows: int, batch_size: int) -> None:
    """Write n_rows of TPC-H Orders data to an Arrow IPC (Feather v2) file.

    Arrow IPC / Feather v2 is a zero-copy columnar format that can be
    memory-mapped directly, making it faster to read than Parquet for
    repeated local access.  Use Parquet for portability and compression.

    Args:
        path:       Absolute or relative path of the output .ipc file.
        n_rows:     Total number of rows to write.
        batch_size: Rows per in-memory batch (controls peak RAM usage).
    """
    import pyarrow.ipc as ipc
    with ipc.new_file(path, SCHEMA) as writer:
        written = 0
        while written < n_rows:
            chunk = min(batch_size, n_rows - written)
            writer.write_batch(make_batch(written, chunk))
            written += chunk
            if n_rows >= 10_000_000:
                pct = written / n_rows * 100
                print(f"\r  {written:,} / {n_rows:,} rows ({pct:.1f}%)", end="", flush=True)
    if n_rows >= 10_000_000:
        print()
    print(f"Wrote {n_rows:,} rows × {len(SCHEMA)} cols → {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a typed Arrow table and write to file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows",       type=int, default=10,
                        help="Total number of rows to generate")
    parser.add_argument("--batch-size", type=int, default=1_000_000,
                        help="Rows per write batch (controls peak memory usage)")
    parser.add_argument("--output",     type=str, default="orders.parquet",
                        help="Output filename (written inside ./data/)")
    parser.add_argument("--format",     choices=["parquet", "ipc"], default="parquet",
                        help="Output format: parquet or Arrow IPC (Feather v2)")
    parser.add_argument("--preview",    type=int, default=5,
                        help="Number of rows to preview (0 = skip preview)")
    args = parser.parse_args()

    # --output must be a plain filename, not a path — output dir is always data/
    if os.path.basename(args.output) != args.output or args.output in (".", ".."):
        parser.error(f"--output must be a filename, not a path (got: {args.output!r}). "
                     f"Files are always written to {DATA_DIR}/")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Ensure the output filename carries the correct extension for the chosen format.
    expected_ext = ".parquet" if args.format == "parquet" else ".ipc"
    output = args.output if args.output.endswith(expected_ext) else args.output + expected_ext
    out_path = os.path.join(DATA_DIR, output)

    print("Schema:")
    print(SCHEMA)

    if args.preview > 0:
        preview_rows = min(args.preview, args.rows)
        df = make_batch(0, preview_rows).to_pandas()
        print(f"\nFirst {preview_rows} rows:")
        print(df.to_string(index=False))
    print()

    if args.format == "parquet":
        write_parquet(out_path, args.rows, args.batch_size)
    else:
        write_ipc(out_path, args.rows, args.batch_size)
