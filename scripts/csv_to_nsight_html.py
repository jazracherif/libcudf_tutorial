#!/usr/bin/env python3
"""Generate a self-contained HTML sequence diagram from a CUDA events CSV.

Layout — three columns:
  Timeline   — absolute cumulative time from first event + delta from previous row
  Host (CPU) — lifeline; sync/blocking calls rendered as note boxes or arrows
  Device (GPU) — lifeline; kernel executions rendered as note boxes

Arrow style (same logic as csv_to_mermaid_flow.py):
  Solid  → blocking host calls  (cudaMalloc, cudaFree, actual Memcpy)
  Dashed → async / fire-and-forget (cudaLaunchKernel, cudaMemcpyAsync, Memset)

Click any row to see all CSV fields in the detail panel below.

Usage:
    python scripts/csv_to_nsight_html.py events.csv [output.html]
    python scripts/csv_to_nsight_html.py events.csv out.html --title "My Trace"
    python scripts/csv_to_nsight_html.py events.csv --include-host-only
"""

import argparse
import csv
import html as _html_mod
import json
import os
import re
import sys
from pathlib import Path

# ── Classification sets (identical to csv_to_mermaid_flow.py) ───────────────

ALL_TYPES = {"kernel", "cuda api", "memcpy", "memset", "memory"}

_BLOCKING = frozenset({
    "cudaMalloc", "cudaMallocHost", "cudaMallocManaged", "cudaMallocPitch",
    "cudaFree", "cudaFreeHost",
    "cudaDeviceSynchronize", "cudaThreadSynchronize",
    "cudaEventSynchronize",
})

_SYNC_BACK = frozenset({
    "cudaStreamSynchronize", "cuStreamSynchronize",
    "cudaDeviceSynchronize", "cudaThreadSynchronize",
    "cudaEventSynchronize",
})

_HOST_ONLY = frozenset({
    "cuLibraryLoadData", "cuLibraryUnload",
    "cuLibraryGetKernel", "cuKernelGetName",
    "cuGetProcAddress", "cuInit",
    "cudaOccupancyAvailableDynamicSMemPerBlock",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "cudaFuncGetAttributes", "cudaGetDevice", "cudaGetDeviceProperties",
})

# cuda api calls that are memory operations (coloured red like memcpy/memset)
_MEM_API = frozenset({
    "cudaMalloc", "cudaMallocHost", "cudaMallocManaged", "cudaMallocPitch",
    "cudaFree", "cudaFreeHost",
    "cudaMemcpy", "cudaMemcpyAsync",
    "cudaMemset", "cudaMemsetAsync",
})

# ── SVG layout constants ─────────────────────────────────────────────────────

TL_X    = 110   # right edge of timeline column (text right-aligned here)
H_X     = 300   # centre x of Host (CPU) lifeline
D_X     = 800   # centre x of Device (GPU) lifeline
SVG_W   = 1060  # total SVG width
BOX_W   = 350   # width of note / activation boxes

ARROW_H = 40    # row height for arrow events
NOTE_H  = 82    # row height for note-box events (kernel, sync)
HDR_H   = 52    # height of the participant header strip

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_bytes(raw):
    try:
        b = int(raw)
    except (ValueError, TypeError):
        return f"{raw} B" if raw else ""
    if b >= 1_000_000_000: return f"{b/1e9:.2f} GB"
    if b >= 1_000_000:     return f"{b/1e6:.1f} MB"
    if b >= 1_000:         return f"{b//1000} KB"
    return f"{b} B"


def _fmt_ns(ns):
    """Format a nanosecond value as a compact human-readable string."""
    if ns < 1_000:         return f"{ns} ns"
    if ns < 1_000_000:     return f"{ns/1000:.1f} \u03bcs"
    if ns < 1_000_000_000: return f"{ns/1e6:.3f} ms"
    return f"{ns/1e9:.4f} s"


def _html_escape(s):
    return _html_mod.escape(str(s))


def _arrow_stroke_width(bytes_raw):
    """Return 3.0 px for transfers > 1 MB, 1.5 px otherwise."""
    return 3.0 if (bytes_raw and bytes_raw > 1_000_000) else 1.0


def _clean_extra(extra):
    s = re.sub(r";\s*corrId=\d+|corrId=\d+;\s*|corrId=\d+", "", extra)
    return s.strip().strip(";").strip()


def _trunc(s, n=999):
    return s if len(s) <= n else s[:n - 1] + "\u2026"


# ── Row classifier ────────────────────────────────────────────────────────────

def classify(row, include_host_only=False):
    """Return a rendering dict for the row, or None to suppress it."""
    rtype    = row.get("Type", "").strip()
    name     = row.get("Name", "").strip()
    dur      = row.get("Duration", "").strip()
    bytes_   = row.get("Bytes", "").strip()
    grid     = row.get("Grid", "").strip()
    blk      = row.get("Block", "").strip()
    ns       = row.get("Namespace", "").strip()
    short    = row.get("ShortName", "").strip() or name
    start_s  = row.get("Start", "").strip()
    start_ns = int(row.get("Start_ns", "0") or "0")

    base = {"start_s": start_s, "start_ns": start_ns, "csv": row}

    # ── cuda api ─────────────────────────────────────────────────────────
    if rtype == "cuda api":
        if name in _HOST_ONLY:
            if not include_host_only:
                return None
            return {**base, "kind": "note_h",
                    "label": name, "label2": dur, "row_h": NOTE_H}

        if name in _SYNC_BACK:
            return {**base, "kind": "note_h",
                    "label": name, "label2": dur, "row_h": NOTE_H}

        solid = name in _BLOCKING
        bytes_int = int(bytes_) if bytes_ else 0
        bytes_sfx = f" ({_fmt_bytes(bytes_)})" if bytes_ else ""
        dur_sfx   = f" [{dur}]" if dur else ""
        label = f"{name}{bytes_sfx}{dur_sfx}"
        return {**base, "kind": "arrow",
                "src": "H", "dst": "D", "solid": solid,
                "label": label, "row_h": ARROW_H,
                "bytes_raw": bytes_int, "is_mem": name in _MEM_API}

    # ── kernel ────────────────────────────────────────────────────────────
    if rtype == "kernel":
        label = f"{short} ({dur})" if dur else short
        # Format launch configuration as Grid<<<x,y,z>>>, Block<<<x,y,z>>>
        launch_parts = []
        if grid: launch_parts.append(f"Grid<<<{grid}>>>")
        if blk:  launch_parts.append(f"Block<<<{blk}>>>")
        launch_dims = ", ".join(launch_parts)
        extra = _clean_extra(row.get("ExtraDetail", "").strip())
        return {**base, "kind": "note_d",
                "label":  label,
                "label2": launch_dims,
                "label3": extra,
                "row_h": NOTE_H}

    # ── memcpy ────────────────────────────────────────────────────────────
    if rtype == "memcpy":
        n = name.lower()
        bytes_int = int(bytes_) if bytes_ else 0
        bytes_sfx = f" ({_fmt_bytes(bytes_)})" if bytes_ else ""
        dur_sfx   = f" [{dur}]" if dur else ""
        if "device-to-host" in n or "d2h" in n:
            return {**base, "kind": "arrow",
                    "src": "D", "dst": "H", "solid": True,
                    "label": f"Memcpy D\u2192H{bytes_sfx}{dur_sfx}", "row_h": ARROW_H,
                    "bytes_raw": bytes_int, "is_mem": True}
        return {**base, "kind": "arrow",
                "src": "H", "dst": "D", "solid": True,
                "label": f"Memcpy H\u2192D{bytes_sfx}{dur_sfx}", "row_h": ARROW_H,
                "bytes_raw": bytes_int, "is_mem": True}

    # ── memset ────────────────────────────────────────────────────────────
    if rtype == "memset":
        bytes_sfx = f" ({_fmt_bytes(bytes_)})" if bytes_ else ""
        dur_sfx   = f" [{dur}]" if dur else ""
        return {**base, "kind": "arrow",
                "src": "H", "dst": "D", "solid": False,
                "label": f"Memset{bytes_sfx}{dur_sfx}", "row_h": ARROW_H,
                "bytes_raw": int(bytes_) if bytes_ else 0, "is_mem": True}

    # ── memory ────────────────────────────────────────────────────────────
    if rtype == "memory":
        label = name + (f" ({_fmt_bytes(bytes_)})" if bytes_ else "")
        return {**base, "kind": "note_both",
                "label": label, "label2": dur, "row_h": NOTE_H}

    return None


# ── SVG element builders ──────────────────────────────────────────────────────

def _tl_lines(row_top_y, row_height, cumulative_time, delta_time):
    """Return SVG <text> elements for the timeline column of one row.

    Renders two right-aligned strings at TL_X, vertically centred in the row:
      - cumulative_time (darker): total elapsed time since the first event.
      - delta_time (lighter grey): time since the previous row; omitted when
        it is the first event or when the value is empty.
    """
    row_mid_y = row_top_y + row_height // 2
    lines = [
        f'    <text x="{TL_X}" y="{row_mid_y - 5}" text-anchor="end"'
        f' class="time-cumulative">{_html_escape(cumulative_time)}</text>'
    ]
    if delta_time:
        lines.append(
            f'    <text x="{TL_X}" y="{row_mid_y + 7}" text-anchor="end"'
            f' class="time-delta">{_html_escape(delta_time)}</text>'
        )
    return lines


def _arrow_lines(ev, row_top_y):
    """Return SVG elements for a horizontal arrow between the two lifelines.

    Draws a labelled <line> from the source lifeline to the destination
    lifeline, with a 7 px inset at each end so the shaft doesn't overlap the
    participant boxes.  Arrow style depends on whether the call is blocking:
      - Solid stroke + dark colour  → blocking call (cudaMalloc, actual Memcpy, …)
      - Dashed stroke + light grey  → async / fire-and-forget (cudaLaunchKernel, …)
    The label is placed 4 px above the midpoint of the shaft.
    """
    arrow_y    = row_top_y + ev["row_h"] // 2
    src_x      = H_X if ev["src"] == "H" else D_X
    dst_x      = D_X if ev["dst"] == "D" else H_X
    lifeline_gap = 7
    # Inset both ends so the shaft starts/ends inside the lifeline circles
    shaft_x1, shaft_x2 = (
        (src_x + lifeline_gap, dst_x - lifeline_gap) if src_x < dst_x
        else (src_x - lifeline_gap, dst_x + lifeline_gap)
    )
    label_x      = (shaft_x1 + shaft_x2) // 2
    has_bytes    = ev.get("bytes_raw", 0) > 0
    is_mem       = ev.get("is_mem", False)
    marker_id    = ("arrowhead-solid-red" if ev["solid"] else "arrowhead-dashed-red") if is_mem \
                   else ("arrowhead-solid" if ev["solid"] else "arrowhead-dashed")
    dash_attr    = "" if ev["solid"] else ' stroke-dasharray="5 3"'
    if is_mem:
        stroke_color = "#cc2222" if ev["solid"] else "#cc6666"
    else:
        stroke_color = "#333" if ev["solid"] else "#666"
    stroke_w         = _arrow_stroke_width(ev.get("bytes_raw", 0))
    label_color_attr = ' style="fill:#cc2222"' if is_mem else ""
    return [
        f'    <text x="{label_x}" y="{arrow_y - 4}" text-anchor="middle"'
        f' class="arrow-label"{label_color_attr}>{_html_escape(ev["label"])}</text>',
        f'    <line x1="{shaft_x1}" y1="{arrow_y}" x2="{shaft_x2}" y2="{arrow_y}"'
        f' stroke="{stroke_color}" stroke-width="{stroke_w}"{dash_attr}'
        f' marker-end="url(#{marker_id})"/>',
    ]


def _note_lines(ev, row_top_y):
    """Return SVG elements for a single-lifeline note box (kernel or sync event).

    Draws a rounded <rect> centred on the Host or Device lifeline, then
    overlays up to three text lines vertically centred inside it:
      - box-label    (primary):   kernel short name + duration, or API name.
      - box-sublabel (secondary): CUDA launch dims  <<<grid, block>>>.
      - box-sublabel (tertiary):  ExtraDetail (regsPerThread, statShmem, …).
    Lines are spaced 13 px apart and the group is centred on the box mid-line.
    Colour scheme:
      - Blue  (#dce8f7) with blue border  → Device kernel  (note_d)
      - Amber (#fdf3dc) with amber border → Host sync point (note_h)
    """
    is_device    = ev["kind"] == "note_d"
    center_x     = D_X if is_device else H_X
    box_left_x   = center_x - BOX_W // 2
    box_top_y    = row_top_y + 4
    box_height   = ev["row_h"] - 8
    fill_color   = "#dce8f7" if is_device else "#fdf3dc"
    stroke_color = "#5580aa" if is_device else "#b08840"
    box_mid_y    = box_top_y + box_height // 2
    has_label2   = bool(ev.get("label2"))
    has_label3   = bool(ev.get("label3"))
    num_lines    = 1 + has_label2 + has_label3
    # Centre the block of lines on box_mid_y; lines are 13 px apart.
    # SVG y is baseline, so add ~4 px to visually centre single-line text.
    if num_lines == 3:
        label_y, label2_y, label3_y = box_mid_y - 13, box_mid_y + 1, box_mid_y + 14
    elif num_lines == 2:
        label_y, label2_y, label3_y = box_mid_y - 6, box_mid_y + 8, None
    else:
        label_y, label2_y, label3_y = box_mid_y + 4, None, None
    lines = [
        f'    <rect x="{box_left_x}" y="{box_top_y}" width="{BOX_W}" height="{box_height}"'
        f' rx="2" fill="{fill_color}" stroke="{stroke_color}" stroke-width="1"/>',
        f'    <text x="{center_x}" y="{label_y}" text-anchor="middle"'
        f' class="box-label">{_html_escape(ev["label"])}</text>',
    ]
    if has_label2:
        lines.append(
            f'    <text x="{center_x}" y="{label2_y}" text-anchor="middle"'
            f' class="box-sublabel">{_html_escape(ev["label2"])}</text>'
        )
    if has_label3:
        lines.append(
            f'    <text x="{center_x}" y="{label3_y}" text-anchor="middle"'
            f' class="box-sublabel">{_html_escape(ev["label3"])}</text>'
        )
    return lines


def _note_both_lines(ev, row_top_y):
    """Return SVG elements for a spanning note box that covers both lifelines.

    Used for memory allocation/deallocation events, which are not attributed to
    either participant exclusively.  The box stretches from the left edge of the
    Host column to the right edge of the Device column (width = D_X − H_X + BOX_W).
    Text layout follows the same up-to-three-line vertical-centering logic as _note_lines.
    """
    box_left_x   = H_X - BOX_W // 2
    box_width    = D_X - H_X + BOX_W
    box_top_y    = row_top_y + 4
    box_height   = ev["row_h"] - 8
    box_center_x = (H_X + D_X) // 2
    box_mid_y    = box_top_y + box_height // 2
    has_label2   = bool(ev.get("label2"))
    has_label3   = bool(ev.get("label3"))
    num_lines    = 1 + has_label2 + has_label3
    if num_lines == 3:
        label_y, label2_y, label3_y = box_mid_y - 13, box_mid_y + 1, box_mid_y + 14
    elif num_lines == 2:
        label_y, label2_y, label3_y = box_mid_y - 6, box_mid_y + 8, None
    else:
        label_y, label2_y, label3_y = box_mid_y + 4, None, None
    lines = [
        f'    <rect x="{box_left_x}" y="{box_top_y}" width="{box_width}" height="{box_height}"'
        f' rx="2" fill="#f5f5f5" stroke="#aaa" stroke-width="1"/>',
        f'    <text x="{box_center_x}" y="{label_y}" text-anchor="middle"'
        f' class="box-label">{_html_escape(ev["label"])}</text>',
    ]
    if has_label2:
        lines.append(
            f'    <text x="{box_center_x}" y="{label2_y}" text-anchor="middle"'
            f' class="box-sublabel">{_html_escape(ev["label2"])}</text>'
        )
    if has_label3:
        lines.append(
            f'    <text x="{box_center_x}" y="{label3_y}" text-anchor="middle"'
            f' class="box-sublabel">{_html_escape(ev["label3"])}</text>'
        )
    return lines


# ── SVG builder ───────────────────────────────────────────────────────────────

def build_svg_header():
    """Sticky header SVG: participant boxes + timeline label + bottom separator."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{HDR_H}">']
    for cx, lbl in [(H_X, "Host (CPU)"), (D_X, "Device (GPU)")]:
        bx = cx - 62
        out.append(f'  <rect x="{bx}" y="6" width="124" height="28" rx="3" fill="#eee" stroke="#aaa" stroke-width="1"/>')
        out.append(f'  <text x="{cx}" y="25" text-anchor="middle" class="participant-name">{lbl}</text>')
    out.append(f'  <text x="{TL_X}" y="25" text-anchor="end" class="timeline-header">Timeline</text>')
    # Bottom separator line so the header visually separates from the scrolling rows
    out.append(f'  <line x1="0" y1="{HDR_H - 1}" x2="{SVG_W}" y2="{HDR_H - 1}" stroke="#ccc" stroke-width="1"/>')
    out.append('</svg>')
    return "\n".join(out)


def build_svg_body(events):
    """Scrollable body SVG: lifelines + all event rows, y-origin = 0."""
    body_h = sum(e["row_h"] for e in events) + 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{body_h}">']

    # Arrowhead markers (defined here, where arrows are used)
    out.append(
        '  <defs>\n'
        '    <marker id="arrowhead-solid" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">\n'
        '      <path d="M0,0 L0,6 L8,3 z" fill="#333"/>\n'
        '    </marker>\n'
        '    <marker id="arrowhead-dashed" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">\n'
        '      <path d="M0,0 L0,6 L8,3 z" fill="#666"/>\n'
        '    </marker>\n'
        '    <marker id="arrowhead-solid-red" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">\n'
        '      <path d="M0,0 L0,6 L8,3 z" fill="#cc2222"/>\n'
        '    </marker>\n'
        '    <marker id="arrowhead-dashed-red" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">\n'
        '      <path d="M0,0 L0,6 L8,3 z" fill="#cc6666"/>\n'
        '    </marker>\n'
        '  </defs>'
    )

    # Lifelines span the full body height
    out.append(f'  <line x1="{H_X}" y1="0" x2="{H_X}" y2="{body_h}"'
               f' stroke="#ccc" stroke-width="1.5" stroke-dasharray="4 3"/>')
    out.append(f'  <line x1="{D_X}" y1="0" x2="{D_X}" y2="{body_h}"'
               f' stroke="#ccc" stroke-width="1.5" stroke-dasharray="4 3"/>')

    # Events — y starts at 0 (header is in a separate sticky SVG)
    y        = 0
    first_ns = events[0]["start_ns"] if events else 0
    prev_ns  = first_ns

    for idx, ev in enumerate(events):
        cur_ns   = ev["start_ns"]
        cumul_ns = cur_ns - first_ns
        delta_ns = cur_ns - prev_ns
        prev_ns  = cur_ns

        ts_str = _fmt_ns(cumul_ns) if cumul_ns > 0 else "t = 0"
        dt_str = f"+{_fmt_ns(delta_ns)}" if idx > 0 else ""

        rh = ev["row_h"]
        out.append(f'  <g class="event-row" data-idx="{idx}" onclick="selectEvent({idx})">')
        out.append(f'    <rect class="row-hitbox" x="0" y="{y}" width="{SVG_W}" height="{rh}"/>')
        out.extend(_tl_lines(y, rh, ts_str, dt_str))

        kind = ev["kind"]
        if kind == "arrow":
            out.extend(_arrow_lines(ev, y))
        elif kind in ("note_h", "note_d"):
            out.extend(_note_lines(ev, y))
        elif kind == "note_both":
            out.extend(_note_both_lines(ev, y))

        out.append("  </g>")
        y += rh

    out.append("</svg>")
    return "\n".join(out)


# ── HTML page wrapper ─────────────────────────────────────────────────────────
#
# Page layout (flex row, 100vh):
#
#   ┌────────────────────────────────────────────────────┬──────────────────┐
#   │  h2 — title bar (full width, above the flex row)   │                  │
#   ├────────────────────────────────────────────────────┤                  │
#   │  #diag-box  (flex: 1, scrollable)                  │  #detail         │
#   │                                                    │  (flex: 0 300px) │
#   │   SVG sequence diagram:                            │                  │
#   │   ┌ Timeline ─┬─ Host (CPU) ──┬─ Device (GPU) ─┐  │  Key/value table │
#   │   │ cumul     │   lifeline    │   lifeline      │  │  populated on    │
#   │   │ +delta    │               │                 │  │  row click.      │
#   │   │           │  [sync box]   │  [kernel box]   │  │                  │
#   │   │           │ ──────────►   │                 │  │  Shows all CSV   │
#   │   │           │ ◄── - - - -   │                 │  │  fields for the  │
#   │   └───────────┴───────────────┴─────────────────┘  │  selected event. │
#   └────────────────────────────────────────────────────┴──────────────────┘
#
# SVG internals:
#   - Dashed vertical lines drawn first as lifelines; event boxes overlay them.
#   - Each event is a <g class="event-row" data-idx="N" onclick="selectEvent(N)">.
#   - A transparent <rect class="row-hitbox"> covers the full row width for click/hover.
#   - Arrow events: <line> with solid stroke (blocking) or stroke-dasharray (async)
#     plus a <marker> arrowhead.  Label rendered above the midpoint of the line.
#   - Note boxes (note_h / note_d): <rect> + up to two <text> lines.
#     note_h → amber (#fdf3dc) centred on H_X; note_d → blue (#dce8f7) on D_X.
#   - note_both: wide rect spanning H_X … D_X (memory-range events).
#
# JS:
#   - allEvents[] — JSON array of CSV-field dicts, one entry per visible event.
#   - selectEvent(i) highlights the clicked <g> (.selected class → blue tint) and
#     renders a two-column <table> of non-empty CSV fields into #info-panel.

def _html_page(title, svg_header, svg_body, events_json):
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}               /* reset default browser spacing */
body {{ font-family: monospace; font-size: 14px; background: #fff; color: #222; }}
h2 {{ padding: 6px 12px; font-size: 16px; border-bottom: 1px solid #ddd; background: #f5f5f5; }} /* title bar above the diagram */
#desc {{ padding: 6px 0 10px; font-size: 12px; color: #555; line-height: 1.6;
         border-bottom: 1px solid #ddd; margin-bottom: 6px; }}                          /* legend above the detail table inside the right panel */
#page-layout {{ display: flex; flex-direction: row; height: 100vh; }}        /* side-by-side layout; fills viewport height */
#diagram-column {{ display: flex; flex-direction: column; flex: 0 0 auto; }}     /* inner column: title + description + diagram stacked vertically */
#diagram-scroll {{ flex: 1 1 0; overflow: auto; position: relative; }}    /* single scroll container — x+y scroll tracks header and body together */
#diagram-sticky-header {{ position: sticky; top: 0; z-index: 10; background: #fff; line-height: 0; }} /* participant boxes; sticks to top when rows scroll beneath */
#diagram-rows {{ display: block; }}                                        /* event rows body; flows directly below the sticky header */
#info-panel {{ flex: 0 0 520px; overflow-y: auto; padding: 6px 12px;     /* fixed 520px right panel; scrolls independently */
           border-left: 2px solid #ccc; background: #fafafa;
           font-size: 13px; color: #333; }}
#info-panel table {{ border-collapse: collapse; width: 100%; margin-top: 14px; }} /* no double-borders; gap between legend text and table */
#info-panel td {{ padding: 3px 8px; border: 1px solid #e8e8e8;
              vertical-align: top; word-break: break-all; }}           /* long hex addresses / paths wrap instead of overflowing */
#info-panel td:first-child {{ font-weight: bold; white-space: nowrap; color: #666;
                          width: 150px; background: #f0f0f0; word-break: normal; }} /* key column: fixed width, never broken mid-word */
#info-panel .placeholder {{ color: #aaa; padding: 4px; }}                           /* placeholder text shown before any row is clicked */
/* SVG styles */
svg text  {{ font-family: monospace; }}
.participant-name {{ font-size: 14px; font-weight: bold; fill: #333; }}       /* participant header labels: "Host (CPU)", "Device (GPU)" */
.timeline-header  {{ font-size: 13px; font-weight: bold; fill: #555; }}       /* "Timeline" column header */
.time-cumulative  {{ font-size: 12px; fill: #555; }}                           /* cumulative time from first event (darker) */
.time-delta       {{ font-size: 12px; fill: #999; }}                           /* delta from previous row (lighter grey) */
.arrow-label      {{ font-size: 13px; fill: #222; }}                           /* arrow label rendered above the midpoint of a <line> */
.box-label        {{ font-size: 13px; fill: #222; }}                           /* primary text line inside a note box (kernel name / API call) */
.box-sublabel     {{ font-size: 12px; fill: #666; }}                           /* secondary text line (grid/block dims or duration) */
.event-row    {{ cursor: pointer; }}                                        /* hand cursor signals all rows are clickable */
.event-row .row-hitbox       {{ fill: transparent; }}                              /* invisible rect covering full row width — enables hover/click on empty areas */
.event-row:hover .row-hitbox {{ fill: rgba(0, 0, 0, 0.04); }}                     /* faint dark tint on hover */
.event-row.selected .row-hitbox   {{ fill: rgba(60, 120, 200, 0.10); }}                /* blue tint applied by selectEvent() to the selected row */
</style>
</head>
<body>
<div id="page-layout">
  <div id="diagram-column">
    <h2>{title}</h2>
    <div id="diagram-scroll">
      <div id="diagram-sticky-header">{svg_header}</div>
      <div id="diagram-rows">{svg_body}</div>
    </div>
  </div>
  <div id="info-panel">
    <p id="legend">
      Sequence diagram of a single <code>groupby + SUM</code> call on 100&nbsp;M rows (Nsight Systems capture).
      Each row is one CUDA event — kernel launch, API call, or memory operation — in chronological order.
      <b>Timeline</b>: cumulative offset from first event + delta from previous row.
      <b>Host (CPU)</b>: solid arrows = blocking calls; dashed = async.
      Amber boxes = sync points. Blue boxes = GPU kernels.
      Click any row to populate the table below.
    </p>
    <span class="placeholder" id="info-placeholder">No event selected.</span>
  </div>
</div>
<script>
const allEvents = {events_json};
let selectedIdx = -1;
function selectEvent(i) {{
  if (selectedIdx >= 0) {{
    const prev = document.querySelector('.event-row[data-idx="' + selectedIdx + '"]');
    if (prev) prev.classList.remove('selected');
  }}
  selectedIdx = i;
  const row = document.querySelector('.event-row[data-idx="' + i + '"]');
  if (row) row.classList.add('selected');
  const e = allEvents[i];
  let rows = '';
  for (const [k, v] of Object.entries(e)) {{
    if (v === '' || v === null || v === undefined) continue;
    rows += '<tr><td>' + k + '</td><td>' + v + '</td></tr>';
  }}
  const placeholder = document.getElementById('info-placeholder');
  if (placeholder) placeholder.remove();
  let tbl = document.getElementById('info-table');
  if (!tbl) {{
    tbl = document.createElement('table');
    tbl.id = 'info-table';
    document.getElementById('info-panel').appendChild(tbl);
  }}
  tbl.innerHTML = rows;
}}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_csv", help="CSV from extract_nsys_events.py")
    ap.add_argument("output_html", nargs="?",
                    help="Output HTML path (default: <stem>_flow.html next to CSV)")
    ap.add_argument("--title", default="", help="Page title (default: CSV filename stem)")
    ap.add_argument("--include-host-only", action="store_true", dest="include_host_only",
                    help="Show suppressed host-only driver calls")
    args = ap.parse_args()

    if not os.path.isfile(args.input_csv):
        sys.exit(f"[ERROR] Not found: {args.input_csv}")

    stem  = Path(args.input_csv).stem
    out   = args.output_html or str(Path(args.input_csv).parent / f"{stem}_flow.html")
    title = args.title or stem.replace("_", " ")

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    events = []
    for row in raw:
        if row.get("Type", "").strip() not in ALL_TYPES:
            continue
        ev = classify(row, args.include_host_only)
        if ev is None:
            continue
        events.append(ev)

    print(f"[INFO] {len(events)} visible events -> {out}", file=sys.stderr)

    svg_header  = build_svg_header()
    svg_body    = build_svg_body(events)
    detail_data = [{k: str(v) for k, v in ev["csv"].items()} for ev in events]
    page        = _html_page(
        title=_html_escape(title),
        svg_header=svg_header,
        svg_body=svg_body,
        events_json=json.dumps(detail_data, ensure_ascii=False),
    )

    with open(out, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[OK] -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
