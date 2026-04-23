#!/usr/bin/env python3
"""Extract CUDA kernels, API calls, and memory operations from an Nsight Systems
.nsys-rep report, scoped to one or more named NVTX regions (default: 'aggregate').

Outputs a single time-ordered CSV.  Each row has a NVTXLabel column identifying
which NVTX region it came from, making it easy to compare events across regions.

Requirements:
    - nsys (Nsight Systems CLI) in PATH
    - Python 3.8+

Usage:
    python scripts/extract_nsys_events.py report.nsys-rep [output.csv]
    python scripts/extract_nsys_events.py report.nsys-rep out.csv --nvtx-label "groupby"
    python scripts/extract_nsys_events.py report.nsys-rep out.csv \\
        --nvtx-label "libcudf:aggregate" "libcudf:merge"
    python scripts/extract_nsys_events.py report.nsys-rep out.csv --all
"""

import argparse
import csv
import os
import sqlite3
import subprocess
import sys

# ---------------------------------------------------------------------------
# nsys export
# ---------------------------------------------------------------------------

def export_to_sqlite(nsys_rep: str, sqlite_out: str) -> None:
    cmd = [
        "nsys", "export",
        "--type", "sqlite",
        "--force-overwrite", "true",
        "--output", sqlite_out,
        nsys_rep,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[ERROR] nsys export failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

_string_cache: dict[int, str] = {}


def lookup_string(conn: sqlite3.Connection, name_id: int) -> str:
    if name_id in _string_cache:
        return _string_cache[name_id]
    cur = conn.execute("SELECT value FROM StringIds WHERE id = ?", (name_id,))
    row = cur.fetchone()
    val = row[0] if row else f"<id:{name_id}>"
    _string_cache[name_id] = val
    return val


def ns_to_sec_str(ns: int) -> str:
    return f"{ns / 1e9:.9f}s"


def ns_to_human(ns: int) -> str:
    if ns < 1_000:
        return f"{ns} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.3f} \u03bcs"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    return f"{ns / 1_000_000_000:.3f} s"


def load_enum(conn: sqlite3.Connection, table: str) -> dict[int, str]:
    """Load an ENUM_* table into an {id: label} dict."""
    try:
        cur = conn.execute(f"SELECT id, label FROM {table}")
        return {row[0]: row[1] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        return {}


# ---------------------------------------------------------------------------
# NVTX range detection
# ---------------------------------------------------------------------------

def find_nvtx_range(conn: sqlite3.Connection, label: str) -> tuple[int, int]:
    """Return (start_ns, end_ns) for the innermost NVTX range matching label.

    Searches for the label in both the direct `text` column and via the
    `textId` → StringIds join.  Also tries just the suffix after the last `:`
    so that labels like "libcudf:aggregate" match NVTX text stored as
    "aggregate" (the domain prefix is recorded separately by nsys).
    """
    query_direct = """
        SELECT start, end FROM NVTX_EVENTS
        WHERE lower(text) LIKE ? AND end IS NOT NULL
        ORDER BY (end - start) ASC
    """
    query_via_id = """
        SELECT n.start, n.end FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE lower(s.value) LIKE ? AND n.end IS NOT NULL
        ORDER BY (n.end - n.start) ASC
    """
    # Build candidate patterns: full label, then suffix after last ':'
    candidates = [label.lower()]
    if ":" in label:
        candidates.append(label.lower().rsplit(":", 1)[-1])

    rows: list = []
    for candidate in candidates:
        pattern = f"%{candidate}%"
        rows = conn.execute(query_direct, (pattern,)).fetchall()
        if not rows:
            rows = conn.execute(query_via_id, (pattern,)).fetchall()
        if rows:
            break

    if not rows:
        print(f"[WARN] No NVTX '{label}' region found — extracting all events",
              file=sys.stderr)
        return 0, 2**63 - 1

    start_ns, end_ns = int(rows[0][0]), int(rows[0][1])
    print(
        f"[INFO] NVTX '{label}' range: "
        f"{ns_to_sec_str(start_ns)} – {ns_to_sec_str(end_ns)} "
        f"(duration {ns_to_human(end_ns - start_ns)})",
        file=sys.stderr,
    )
    return start_ns, end_ns


# ---------------------------------------------------------------------------
# Kernel name helpers
# ---------------------------------------------------------------------------

def shorten_kernel_name(full_name: str) -> str:
    """Return the qualified function name, stripping the return type,
    template parameters, and argument list.

    Examples:
      'void cub::detail::select::DeviceSelectSweepKernel<A,B>(C,D)'
        → 'cub::detail::select::DeviceSelectSweepKernel'
      'cudf::groupby::detail::hash::<unnamed>::single_pass_shmem_aggs_kernel(...)'
        → 'cudf::groupby::detail::hash::<unnamed>::single_pass_shmem_aggs_kernel'
    """
    import re
    name = full_name.strip()
    # Cut at the first '<' or '(' to drop templates and argument list
    m = re.match(r'^([\w\s:*&<>()]+?)(?:<|\(|$)', name)
    if m:
        name = m.group(1).strip()
    # Drop leading return-type tokens (words before the first '::')
    # e.g. "void cub::..." → "cub::..."
    parts = name.split('::')
    if len(parts) > 1 and ' ' in parts[0]:
        parts[0] = parts[0].rsplit(' ', 1)[-1]
    return '::'.join(parts)


# ---------------------------------------------------------------------------
# Kernel extraction  (CUPTI_ACTIVITY_KIND_KERNEL)
# ---------------------------------------------------------------------------

def extract_kernels(conn: sqlite3.Connection, t0: int, t1: int) -> list[dict]:
    cur = conn.execute("""
        SELECT start, end,
               demangledName, shortName, mangledName,
               gridX, gridY, gridZ, blockX, blockY, blockZ,
               streamId, deviceId,
               dynamicSharedMemory, staticSharedMemory,
               registersPerThread, correlationId
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= ? AND start <= ?
        ORDER BY start
    """, (t0, t1))

    rows = []
    for r in cur.fetchall():
        (t_start, t_end,
         demangled, short, mangled,
         gx, gy, gz, bx, by, bz,
         stream, device,
         dyn_shmem, stat_shmem, regs, corr) = r

        dur = (int(t_end) - int(t_start)) if t_end is not None else 0
        # Name columns are StringIds foreign keys (integers), not raw strings
        demangled_str = lookup_string(conn, demangled) if isinstance(demangled, int) else (demangled or "")
        short_str = lookup_string(conn, short) if isinstance(short, int) else (short or "")
        mangled_str = lookup_string(conn, mangled) if isinstance(mangled, int) else (mangled or "")
        full_name = demangled_str or mangled_str or short_str or "<kernel>"
        name = shorten_kernel_name(full_name) if full_name != "<kernel>" else "<kernel>"
        detail = []
        if dyn_shmem:
            detail.append(f"dynShmem={dyn_shmem}B")
        if stat_shmem:
            detail.append(f"statShmem={stat_shmem}B")
        if regs:
            detail.append(f"regsPerThread={regs}")
        if corr:
            detail.append(f"corrId={corr}")

        ns, short_name = split_kernel_namespace(name)
        rows.append({
            "Type": "kernel",
            "Namespace": ns,
            "Name": short_name,
            "ShortName": short_str or "",
            "FullName": full_name,
            "APIVersion": "",
            "Start_ns": int(t_start),
            "Start": ns_to_sec_str(int(t_start)),
            "Duration": ns_to_human(dur),
            "Duration_ns": dur,
            "Bytes": "",
            "Grid": f"{gx},{gy},{gz}" if gx is not None else "",
            "Block": f"{bx},{by},{bz}" if bx is not None else "",
            "Stream": str(stream) if stream is not None else "",
            "Device": f"GPU {device}" if device is not None else "",
            "TID": "",
            "ExtraDetail": "; ".join(detail),
        })
    return rows


# ---------------------------------------------------------------------------
# CUDA Runtime API extraction  (CUPTI_ACTIVITY_KIND_RUNTIME)
# ---------------------------------------------------------------------------

def split_kernel_namespace(name: str) -> tuple[str, str]:
    """Split 'cub::detail::for_each::static_kernel' into
    ('cub::detail::for_each', 'static_kernel').
    Returns ('', name) if there is no '::' separator.
    """
    parts = [p for p in name.split("::") if p]  # drop empty segments
    if len(parts) >= 2:
        return "::".join(parts[:-1]), parts[-1]
    return "", name


def strip_api_version(name: str) -> tuple[str, str]:
    """Split 'cudaMalloc_v3020' into ('cudaMalloc', 'v3020').
    Returns (name, '') if no version suffix is present.
    """
    import re
    m = re.match(r'^(.+?)(_v\d+)$', name)
    if m:
        return m.group(1), m.group(2)[1:]  # strip leading '_'
    return name, ""


def extract_cuda_api(conn: sqlite3.Connection, t0: int, t1: int,
                     corr_bytes: dict | None = None) -> list[dict]:
    cur = conn.execute("""
        SELECT nameId, start, end, globalTid, correlationId, returnValue
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE start >= ? AND start <= ?
        ORDER BY start
    """, (t0, t1))

    rows = []
    for r in cur.fetchall():
        name_id, t_start, t_end, tid, corr, ret = r
        raw_name = lookup_string(conn, name_id)
        name, api_ver = strip_api_version(raw_name)
        dur = (int(t_end) - int(t_start)) if t_end is not None else 0
        detail = []
        if corr:
            detail.append(f"corrId={corr}")
        if ret and ret != 0:
            detail.append(f"retVal={ret}")

        rows.append({
            "Type": "cuda api",
            "Namespace": "",
            "Name": name,
            "ShortName": name,
            "FullName": "",
            "APIVersion": api_ver,
            "Start_ns": int(t_start),
            "Start": ns_to_sec_str(int(t_start)),
            "Duration": ns_to_human(dur),
            "Duration_ns": dur,
            "Bytes": str(corr_bytes[corr]) if corr_bytes and corr and corr in corr_bytes else "",
            "Grid": "",
            "Block": "",
            "Stream": "",
            "Device": "",
            "TID": str(tid) if tid is not None else "",
            "ExtraDetail": "; ".join(detail),
        })
    return rows


# ---------------------------------------------------------------------------
# Memcpy extraction  (CUPTI_ACTIVITY_KIND_MEMCPY)
# ---------------------------------------------------------------------------

def extract_memcpy(conn: sqlite3.Connection, t0: int, t1: int) -> list[dict]:
    copy_kind_map = load_enum(conn, "ENUM_CUDA_MEMCPY_OPER")
    # Fallback hardcoded labels for common kinds
    copy_kind_map.setdefault(1, "Host-to-Device")
    copy_kind_map.setdefault(2, "Device-to-Host")
    copy_kind_map.setdefault(8, "Device-to-Device")
    copy_kind_map.setdefault(10, "Peer-to-Peer")

    cur = conn.execute("""
        SELECT start, end, bytes, copyKind, srcKind, dstKind,
               streamId, deviceId, correlationId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE start >= ? AND start <= ?
        ORDER BY start
    """, (t0, t1))

    rows = []
    for r in cur.fetchall():
        t_start, t_end, nbytes, copy_kind, src_kind, dst_kind, stream, device, corr = r
        dur = (int(t_end) - int(t_start)) if t_end is not None else 0
        kind_label = copy_kind_map.get(copy_kind, f"kind={copy_kind}")
        name = f"Memcpy ({kind_label})"
        detail = []
        if corr:
            detail.append(f"corrId={corr}")

        rows.append({
            "Type": "memcpy",
            "Namespace": "",
            "Name": name,
            "ShortName": name,
            "FullName": "",
            "APIVersion": "",
            "Start_ns": int(t_start),
            "Start": ns_to_sec_str(int(t_start)),
            "Duration": ns_to_human(dur),
            "Duration_ns": dur,
            "Bytes": str(nbytes) if nbytes is not None else "",
            "Grid": "",
            "Block": "",
            "Stream": str(stream) if stream is not None else "",
            "Device": f"GPU {device}" if device is not None else "",
            "TID": "",
            "ExtraDetail": "; ".join(detail),
        })
    return rows


# ---------------------------------------------------------------------------
# Memset extraction  (CUPTI_ACTIVITY_KIND_MEMSET)
# ---------------------------------------------------------------------------

def extract_memset(conn: sqlite3.Connection, t0: int, t1: int) -> list[dict]:
    cur = conn.execute("""
        SELECT start, end, bytes, streamId, deviceId, correlationId
        FROM CUPTI_ACTIVITY_KIND_MEMSET
        WHERE start >= ? AND start <= ?
        ORDER BY start
    """, (t0, t1))

    rows = []
    for r in cur.fetchall():
        t_start, t_end, nbytes, stream, device, corr = r
        dur = (int(t_end) - int(t_start)) if t_end is not None else 0
        detail = [f"corrId={corr}"] if corr else []

        rows.append({
            "Type": "memset",
            "Namespace": "",
            "Name": "Memset",
            "ShortName": "Memset",
            "FullName": "",
            "APIVersion": "",
            "Start_ns": int(t_start),
            "Start": ns_to_sec_str(int(t_start)),
            "Duration": ns_to_human(dur),
            "Duration_ns": dur,
            "Bytes": str(nbytes) if nbytes is not None else "",
            "Grid": "",
            "Block": "",
            "Stream": str(stream) if stream is not None else "",
            "Device": f"GPU {device}" if device is not None else "",
            "TID": "",
            "ExtraDetail": "; ".join(detail),
        })
    return rows


# ---------------------------------------------------------------------------
# GPU memory allocation/deallocation  (CUDA_GPU_MEMORY_USAGE_EVENTS)
# ---------------------------------------------------------------------------

def extract_memory_allocs(
    conn: sqlite3.Connection, t0: int, t1: int,
    include_device_static: bool = False,
) -> tuple[list[dict], dict]:
    """Return (rows, corr_bytes_map) where corr_bytes_map maps correlationId→bytes
    for all memory events in the window (used to annotate CUDA API rows).
    """
    oper_map = load_enum(conn, "ENUM_CUDA_DEV_MEM_EVENT_OPER")
    mem_kind_map = load_enum(conn, "ENUM_CUDA_MEM_KIND")

    # CUDA_MEMOPR_MEMORY_KIND_DEVICE_STATIC = 5
    # These are __device__ global variables registered at module-load time
    # (libcudacxx CPO tag objects, constexpr sentinels, etc.) — 1 byte each,
    # zero runtime cost.  Filtered out by default.
    kind_filter = "" if include_device_static else "AND memKind != 5"

    cur = conn.execute(f"""
        SELECT start, bytes, memoryOperationType, memKind,
               name, streamId, deviceId, correlationId
        FROM CUDA_GPU_MEMORY_USAGE_EVENTS
        WHERE start >= ? AND start <= ?
        {kind_filter}
        ORDER BY start
    """, (t0, t1))

    rows = []
    corr_bytes: dict = {}
    for r in cur.fetchall():
        t_start, nbytes, oper_type, mem_kind, name_str, stream, device, corr = r
        oper_label = oper_map.get(oper_type, f"oper={oper_type}")
        kind_label = mem_kind_map.get(mem_kind, "")
        name = f"Memory {oper_label}" + (f" ({kind_label})" if kind_label else "")
        detail = []
        if name_str:
            detail.append(f"name={name_str}")
        if corr:
            detail.append(f"corrId={corr}")
            if nbytes is not None:
                corr_bytes[corr] = nbytes

        rows.append({
            "Type": "memory",
            "Namespace": "",
            "Name": name,
            "ShortName": name,
            "FullName": "",
            "APIVersion": "",
            "Start_ns": int(t_start),
            "Start": ns_to_sec_str(int(t_start)),
            "Duration": "",
            "Duration_ns": 0,
            "Bytes": str(nbytes) if nbytes is not None else "",
            "Grid": "",
            "Block": "",
            "Stream": str(stream) if stream is not None else "",
            "Device": f"GPU {device}" if device is not None else "",
            "TID": "",
            "ExtraDetail": "; ".join(detail),
        })
    return rows, corr_bytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "Start_ns",
    "NVTXLabel",
    "Type", "Namespace", "Name", "APIVersion", "ShortName",
    "Start", "Duration", "Duration_ns",
    "Bytes", "Grid", "Block", "Stream", "Device", "TID",
    "ExtraDetail", "FullName",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("nsys_rep", help="Path to .nsys-rep report file")
    parser.add_argument(
        "output_csv", nargs="?",
        help="Output CSV path (default: <report_stem>_<nvtx_label>.csv)",
    )
    parser.add_argument(
        "--nvtx-label", nargs="+", default=["libcudf:aggregate"],
        metavar="LABEL",
        help=(
            "One or more NVTX region labels to scope extraction "
            "(default: 'libcudf:aggregate'). "
            "Each label produces its own set of rows tagged with a NVTXLabel column. "
            "Example: --nvtx-label libcudf:aggregate libcudf:merge"
        ),
    )
    parser.add_argument(
        "--all", action="store_true", dest="extract_all",
        help="Extract all events (ignore NVTX scoping)",
    )
    parser.add_argument(
        "--keep-sqlite", action="store_true",
        help="Keep the intermediate .sqlite file after export",
    )
    parser.add_argument(
        "--include-device-static", action="store_true", dest="include_device_static",
        help=(
            "Include 'Memory Allocation (Device Static)' events — i.e. __device__ "
            "global variables registered at CUDA module load time (libcudacxx CPOs, "
            "constexpr sentinels, etc.). These are 1-byte zero-cost symbols and are "
            "filtered out by default."
        ),
    )
    parser.add_argument(
        "--include-memory", action="store_true", dest="include_memory",
        help=(
            "Include memory allocation/deallocation rows in the output. "
            "By default these are suppressed (but their Bytes values are still "
            "propagated to the corresponding CUDA API rows via corrId)."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.nsys_rep):
        sys.exit(f"[ERROR] File not found: {args.nsys_rep}")

    stem = os.path.splitext(os.path.abspath(args.nsys_rep))[0]
    sqlite_path = stem + ".sqlite"
    if args.extract_all:
        label_slug = "all"
    else:
        label_slug = "_".join(
            lbl.replace(" ", "_").replace(":", "_") for lbl in args.nvtx_label
        )
    output_csv = args.output_csv or (stem + f"_{label_slug}.csv")

    if not os.path.isfile(sqlite_path):
        print(f"[INFO] Exporting to SQLite: {sqlite_path}", file=sys.stderr)
        export_to_sqlite(args.nsys_rep, sqlite_path)
    else:
        print(f"[INFO] Reusing existing SQLite: {sqlite_path}", file=sys.stderr)

    conn = sqlite3.connect(sqlite_path)

    rows: list[dict] = []

    if args.extract_all:
        t0, t1 = 0, 2**63 - 1
        print("[INFO] Extracting all events (--all)", file=sys.stderr)
        label_rows: list[dict] = []
        mem_rows, corr_bytes = extract_memory_allocs(
            conn, t0, t1, include_device_static=args.include_device_static
        )
        label_rows += extract_kernels(conn, t0, t1)
        label_rows += extract_cuda_api(conn, t0, t1, corr_bytes=corr_bytes)
        label_rows += extract_memcpy(conn, t0, t1)
        label_rows += extract_memset(conn, t0, t1)
        if args.include_memory:
            label_rows += mem_rows
        for r in label_rows:
            r["NVTXLabel"] = ""
        rows += label_rows
    else:
        for label in args.nvtx_label:
            t0, t1 = find_nvtx_range(conn, label)
            label_rows = []
            mem_rows, corr_bytes = extract_memory_allocs(
                conn, t0, t1, include_device_static=args.include_device_static
            )
            label_rows += extract_kernels(conn, t0, t1)
            label_rows += extract_cuda_api(conn, t0, t1, corr_bytes=corr_bytes)
            label_rows += extract_memcpy(conn, t0, t1)
            label_rows += extract_memset(conn, t0, t1)
            if args.include_memory:
                label_rows += mem_rows
            for r in label_rows:
                r["NVTXLabel"] = label
            rows += label_rows

    rows.sort(key=lambda r: r["Start_ns"])

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    conn.close()

    if not args.keep_sqlite:
        # Only remove if we created it; leave pre-existing SQLite alone
        pass  # keep it — re-export is slow; user can delete manually

    print(
        f"[INFO] {len(rows)} events written to: {output_csv}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
