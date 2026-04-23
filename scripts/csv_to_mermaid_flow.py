#!/usr/bin/env python3
"""Convert a CUDA events CSV (produced by extract_nsys_events.py) into a
Markdown file with a Mermaid sequenceDiagram showing the Host↔Device message flow.

Arrow conventions:
  ->>   solid  : synchronous / blocking  (cudaMalloc, cudaFree, actual memcpy)
  -->>  dashed : asynchronous            (cudaLaunchKernel, cudaMemcpyAsync, kernels)
  D->>H         : device signals host    (cudaStreamSynchronize — host was blocking)

Host-only driver calls (cuLibraryLoadData, cuKernelGetName, etc.) are hidden
by default; use --include-host-only to show them as Host self-arrows.

Usage:
    python scripts/csv_to_mermaid_flow.py events.csv [output.md]
    python scripts/csv_to_mermaid_flow.py events.csv out.md --title "Groupby flow"
    python scripts/csv_to_mermaid_flow.py events.csv --max-rows 60
    python scripts/csv_to_mermaid_flow.py events.csv --types kernel memcpy cuda api
    python scripts/csv_to_mermaid_flow.py events.csv --include-host-only
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Classification tables
# ---------------------------------------------------------------------------

# All event types emitted by extract_nsys_events.py; used as the default
# --types filter so every row is included unless the user narrows the set.
ALL_TYPES = {"kernel", "cuda api", "memcpy", "memset", "memory"}

# APIs that block the host until the GPU work completes  →  solid H->>D
# The CPU stalls for the full duration (e.g. cudaMalloc waits for the driver
# to allocate and zero device memory before returning).
_BLOCKING = frozenset({
    "cudaMalloc", "cudaMallocHost", "cudaMallocManaged", "cudaMallocPitch",
    "cudaFree", "cudaFreeHost",
    "cudaDeviceSynchronize", "cudaThreadSynchronize",
    "cudaEventSynchronize",
})

# APIs whose semantic is "wait for device to finish"  →  Note over H
# The host blocks until the GPU stream drains; rendered as a note over H
# rather than an arrow to avoid cluttering the diagram.
_SYNC_BACK = frozenset({
    "cudaStreamSynchronize", "cuStreamSynchronize",
    "cudaDeviceSynchronize", "cudaThreadSynchronize",
    "cudaEventSynchronize",
})

# Host-side-only driver calls: these touch only CPU-side driver state.
# No GPU memory is allocated and no kernel or DMA is enqueued as a result.
# Hidden by default to reduce noise; use --include-host-only to show them
# as Host self-arrows (H->>H).
_HOST_ONLY = frozenset({
    "cuLibraryLoadData", "cuLibraryUnload",
    "cuLibraryGetKernel", "cuKernelGetName",
    "cuGetProcAddress", "cuInit",
    "cudaOccupancyAvailableDynamicSMemPerBlock",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "cudaFuncGetAttributes", "cudaGetDevice", "cudaGetDeviceProperties",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(raw: str) -> str:
    """Format a raw byte-count string into a human-readable size."""
    try:
        b = int(raw)
    except (ValueError, TypeError):
        return f"{raw} B"
    if b >= 1_000_000_000:
        return f"{b / 1_000_000_000:.2f} GB"
    if b >= 1_000_000:
        return f"{b / 1_000_000:.1f} MB"
    if b >= 1_000:
        return f"{b // 1_000} KB"
    return f"{b} B"


def _sanitize(text: str) -> str:
    """Remove / replace characters that confuse Mermaid's sequenceDiagram parser.

    Mermaid treats '"' as a string delimiter inside labels, newlines break
    parsing, and ';' is used as a statement separator in some Mermaid contexts.
    """
    return (
        text.replace('"', "'")
            .replace("\n", " ")
            .replace(";", ",")
    )


def _build_label(name: str, dur: str = "", bytes_: str = "", extra: str = "") -> str:
    """Assemble 'Name (dur, size, extra)' label."""
    parts = []
    if dur:
        parts.append(dur)
    if bytes_:
        parts.append(_fmt_bytes(bytes_))
    if extra:
        parts.append(extra)
    suffix = f" ({', '.join(parts)})" if parts else ""
    return _sanitize(name + suffix)


# ---------------------------------------------------------------------------
# Row classifier
# ---------------------------------------------------------------------------

def classify(
    row: dict,
    include_host_only: bool,
    no_duration: bool,
    no_bytes: bool,
) -> tuple[str, str, str, str, str | None] | None:
    """Return (src, dst, arrow, label, color) or None to suppress this row.

    src / dst : "H" | "D" | "NOTE"
    arrow     : "->>" (solid/blocking) | "-->>" (dashed/async)
    color     : rect background colour, or None
    """
    rtype  = row.get("Type", "").strip()
    name   = row.get("Name", "").strip()
    dur    = row.get("Duration", "").strip() if not no_duration else ""
    bytes_ = row.get("Bytes", "").strip()    if not no_bytes else ""
    grid   = row.get("Grid", "").strip()
    ns     = row.get("Namespace", "").strip()
    short  = row.get("ShortName", "").strip() or name

    # ── CUDA API ───────────────────────────────────────────────────────────
    if rtype == "cuda api":
        # Pure driver bookkeeping with no observable GPU side-effect — skip
        # unless the user explicitly asked for the noise.
        if name in _HOST_ONLY:
            if not include_host_only:
                return None
            return ("H", "H", "->>", _build_label(name, dur), None)

        # Synchronisation points: the host stalls until the stream drains.
        # Shown as a note over H rather than an arrow to reduce visual clutter.
        if name in _SYNC_BACK:
            return ("NOTE", "H", "", _build_label(name, dur), None)

        # Blocking calls stall the host thread (solid arrow); everything else
        # is fire-and-forget async (dashed arrow).
        arrow = "->>" if name in _BLOCKING else "-->>"
        return ("H", "D", arrow, _build_label(name, dur, bytes_), None)

    # ── Kernel ────────────────────────────────────────────────────────────
    # Kernels run entirely on the device so there is no H↔D arrow to draw.
    # Rendered as a note box over D with up to three lines:
    #   Line 1: namespace::name (duration)
    #   Line 2: Grid:<<<x,y,z>>>, Block:<<<x,y,z>>>   — launch configuration
    #   Line 3: statShmem, regsPerThread, …            — ExtraDetail fields
    if rtype == "kernel":
        block  = row.get("Block", "").strip()
        extra  = row.get("ExtraDetail", "").strip()
        base   = f"{ns}::{short}" if ns else short
        line1  = _sanitize(base + (f" ({dur})" if dur else ""))
        line2_parts = []
        if grid:
            line2_parts.append(f"Grid:<<<{grid}>>>")
        if block:
            line2_parts.append(f"Block:<<<{block}>>>")
        dim_str = ", ".join(line2_parts)
        # Strip corrId from ExtraDetail (internal profiler bookkeeping)
        extra_clean = re.sub(r";\s*corrId=\d+|corrId=\d+;\s*|corrId=\d+", "", extra).strip().strip(";").strip()
        label = line1
        if dim_str:
            label += "<br/>" + dim_str
        if extra_clean:
            label += "<br/>" + _sanitize(extra_clean)
        return ("NOTE", "D", "", label, None)

    # ── Memcpy ────────────────────────────────────────────────────────────
    # The CSV has one memcpy row per actual DMA transfer (device-side timing).
    # The paired cuda api row (cudaMemcpyAsync) captures host-side enqueue
    # overhead; both rows share the same corrId.
    if rtype == "memcpy":
        n = name.lower()
        parts = []
        if dur:
            parts.append(dur)
        if bytes_:
            parts.append(_fmt_bytes(bytes_))
        suffix = f" ({', '.join(parts)})" if parts else ""

        if "host-to-device" in n or "h2d" in n:
            return ("H", "D", "->>",  f"Memcpy H\u2192D{suffix}", None)
        if "device-to-host" in n or "d2h" in n:
            return ("D", "H", "->>",  f"Memcpy D\u2192H{suffix}", None)
        if "device-to-device" in n or "d2d" in n:
            return ("D", "D", "->>",  f"Memcpy D\u2192D{suffix}", None)
        return ("H", "D", "->>", f"Memcpy{suffix}", None)

    # ── Memset ────────────────────────────────────────────────────────────
    if rtype == "memset":
        parts = []
        if dur:
            parts.append(dur)
        if bytes_:
            parts.append(_fmt_bytes(bytes_))
        suffix = f" ({', '.join(parts)})" if parts else ""
        return ("H", "D", "-->>", f"Memset{suffix}", None)

    # ── Memory alloc/dealloc  (shown as notes) ────────────────────────────
    # Memory rows represent RMM/CUDA allocator events and are not tied to a
    # single participant, so they span both H and D as a wide note box.
    if rtype == "memory":
        parts = []
        if bytes_:
            parts.append(_fmt_bytes(bytes_))
        suffix = f" ({', '.join(parts)})" if parts else ""
        return ("NOTE", "", "", _sanitize(name + suffix), None)

    return None


# ---------------------------------------------------------------------------
# Mermaid line formatter
# ---------------------------------------------------------------------------

def to_mermaid_line(src: str, dst: str, arrow: str, label: str) -> str:
    """Render a classified event as one Mermaid sequenceDiagram line.

    src == "NOTE" produces a note box instead of an arrow.  dst controls
    which participant the box is anchored to:
      "H"  → Note over H   (sync/wait calls)
      "D"  → Note over D   (kernel executions)
      ""   → Note over H,D (memory events spanning both sides)
    """
    if src == "NOTE":
        over = dst if dst else "H,D"
        return f"    Note over {over}: {label}"
    return f"    {src}{arrow}{dst}: {label}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_csv", help="Input CSV from extract_nsys_events.py")
    parser.add_argument(
        "output_md", nargs="?",
        help="Output Markdown path (default: <input_stem>_flow.md next to the CSV)",
    )
    parser.add_argument(
        "--title", default="",
        help="Diagram title (default: input filename stem)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0, metavar="N",
        help="Truncate to first N events (default: unlimited; warns if > 200)",
    )
    parser.add_argument(
        "--types", nargs="+", default=sorted(ALL_TYPES),
        metavar="TYPE",
        help=(
            "Event types to include (default: all). "
            f"Choices: {sorted(ALL_TYPES)}"
        ),
    )
    parser.add_argument(
        "--include-host-only", action="store_true", dest="include_host_only",
        help=(
            "Show host-only driver calls (cuLibraryLoadData, cuKernelGetName, etc.) "
            "as Host self-arrows. Hidden by default."
        ),
    )
    parser.add_argument(
        "--no-duration", action="store_true",
        help="Omit duration from edge labels",
    )
    parser.add_argument(
        "--no-bytes", action="store_true",
        help="Omit byte counts from edge labels",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        sys.exit(f"[ERROR] File not found: {args.input_csv}")

    stem      = Path(args.input_csv).stem
    output_md = args.output_md or str(Path(args.input_csv).parent / f"{stem}_flow.md")
    title     = args.title or stem.replace("_", " ")
    type_filter = set(args.types)

    # Read and filter by type
    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    raw_total = len(reader)
    all_rows = [r for r in reader if r.get("Type", "").strip() in type_filter]
    filtered_total = len(all_rows)
    if filtered_total < raw_total:
        print(
            f"[INFO] {filtered_total} of {raw_total} rows match type filter {sorted(type_filter)}",
            file=sys.stderr,
        )

    if args.max_rows and filtered_total > args.max_rows:
        print(
            f"[INFO] Truncating to {args.max_rows} of {filtered_total} filtered rows",
            file=sys.stderr,
        )
        all_rows = all_rows[: args.max_rows]
    elif filtered_total > 200:
        print(
            f"[WARN] {filtered_total} events — Mermaid may render slowly in some viewers. "
            "Use --max-rows N to truncate.",
            file=sys.stderr,
        )
    total_rows = filtered_total

    # Classify each CSV row into a Mermaid line.
    # Rows where classify() returns None (host-only calls, unknown types) are
    # counted as suppressed and omitted from the output diagram.
    mermaid_lines: list[str] = []
    suppressed = 0
    for row in all_rows:
        result = classify(row, args.include_host_only, args.no_duration, args.no_bytes)
        if result is None:
            suppressed += 1
            continue
        src, dst, arrow, label, color = result
        line = to_mermaid_line(src, dst, arrow, label)
        if color:
            mermaid_lines.append(f"    rect {color}")
            mermaid_lines.append(line)
            mermaid_lines.append("    end")
        else:
            mermaid_lines.append(line)

    print(
        f"[INFO] {len(mermaid_lines)} diagram lines, {suppressed} rows suppressed "
        f"→ {output_md}",
        file=sys.stderr,
    )
    if args.max_rows and total_rows > args.max_rows:
        remaining = total_rows - args.max_rows
        print(
            f"[WARN] Showing {args.max_rows} of {total_rows} rows — "
            f"{remaining} more rows not shown. Increase --max-rows to see them.",
            file=sys.stderr,
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"_Generated from `{args.input_csv}`_\n\n")
        f.write("```mermaid\n")
        f.write("%%{init: {'sequence': {'actorMargin': 300, 'width': 200}}}%%\n")
        f.write("sequenceDiagram\n")
        f.write("    participant H as Host\n")
        f.write("    participant D as Device\n")
        for line in mermaid_lines:
            f.write(line + "\n")
        f.write("```\n")


if __name__ == "__main__":
    main()
