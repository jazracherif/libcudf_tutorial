# Symbol Resolution in `backtrace_resource_adaptor`

How the adaptor resolves raw instruction addresses into human-readable function names.

---

## 1. ELF Binary Structure

Every `.so` and executable on Linux is an **ELF** (Executable and Linkable Format) file.
The parts relevant to symbol resolution:

```
libcudf.so
├── .text          — compiled machine code (functions live here)
├── .dynsym        — dynamic symbol table: exported public API names + addresses
├── .dynstr        — string table backing .dynsym
├── .symtab        — full static symbol table: all functions including private ones
│                    (may be stripped in release builds, present in libcudf.so)
├── .strtab        — string table backing .symtab
└── .debug_info    — DWARF debug data: maps addresses → file:line
    .debug_line      (absent in conda release builds of libcudf)
```

**Inspect sections of any ELF file:**
```bash
# List all sections with name, type, offset, and size
readelf -S $CONDA_PREFIX/lib/libcudf.so

# Grep for the symbol-related ones specifically
readelf -S $CONDA_PREFIX/lib/libcudf.so | grep -E '\.symtab|\.dynsym|\.debug'
```

Example output (presence of `.symtab` confirms symbols are not stripped):
```
  [Nr] Name              Type             Address           Offset    Size
  ...
  [26] .dynsym           DYNSYM           0000000000001000  00001000  0x12a40
  [27] .dynstr           STRTAB           0000000000013a40  00013a40  0x0f200
  [42] .symtab           SYMTAB           0000000000000000  01a23000  0x8fa80   ← present = not stripped
  [43] .strtab           STRTAB           0000000000000000  01abc080  0x5c310
```

If `.symtab` is absent, `addr2line` returns `??` for all private symbols.

### .dynsym vs .symtab

| Table | Who reads it | What it contains | Stripped? |
|-------|-------------|------------------|-----------|
| `.dynsym` | `ld.so` at load time, `dladdr()`, `backtrace_symbols()` | Only **exported** symbols — the public API | Never (required for dynamic linking) |
| `.symtab` | `gdb`, `addr2line`, `nm` | **All** symbols including private/internal functions | Often in release builds — **not** in `libcudf.so` (conda aarch64) |

The separation is deliberate:
- **`.dynsym`** is loaded into memory by `ld.so` at process startup for every mapped `.so`. It contains only the symbols visible across library boundaries — functions/variables that other libraries or the main binary may call into. It is intentionally small because it lives in memory for the lifetime of the process.
- **`.symtab`** stays on disk and is never mapped into the process address space at runtime. It is only read by offline tools (`gdb`, `addr2line`, `nm`) that open the `.so` file directly. It can be 10–100× larger than `.dynsym`.

Importantly, `.symtab` is a **superset** of `.dynsym` — every exported symbol is in both. Stripping a release binary (`strip --strip-debug`) removes `.symtab` but **cannot** remove `.dynsym` — doing so would break dynamic linking.

This is why `backtrace_symbols()` alone leaves internal frames unresolved:
it only queries `.dynsym` via `dladdr()`.

---

## 2. Address Spaces and Load Bias


When `ld.so` maps a shared library into the process:

```
Virtual address space:
  [0x0000...]  main binary
  [0xef41...]  libcudf.so   ← mapped at a random base address (ASLR) (the `load_base` address, or `dli_fbase` see below)
  [0xef50...]  librmm.so
```

**ASLR** (Address Space Layout Randomization) is a Linux kernel security feature that maps
shared libraries at a **random base address** every time the process starts, rather than at a
fixed address. This makes buffer-overflow exploits much harder because an attacker cannot
predict where code or data lives in memory.

A **virtual address** (what `backtrace()` returns) for a frame inside `libcudf.so` is:

```
virtual_addr = libcudf_load_base (0xef41...) + file_offset_within_libcudf.so
```

For example, a frame at virtual address `0xef41001a3f00` with `libcudf_load_base = 0xef4100000000`
has `file_offset_within_libcudf = 0x1a3f00` — the position of that instruction in the `.so` file on disk.

`addr2line` expects a **file offset** — the fixed position of the instruction within the `.so`
file on disk, which never changes between runs. You must undo ASLR to get it:

```
file_offset = virtual_addr - load_base   (load_base = dli_fbase from dladdr())
```

The load base is obtained from `dladdr()`, which returns where `ld.so` placed the library
in this run's virtual address space.

---

## 3. APIs Used in the Adaptor

### `backtrace(void** frames, int max)` — `<execinfo.h>`
Walks the current thread's call stack and fills `frames[]` with the
**return addresses** (virtual addresses) of each stack frame.
Returns the number of frames captured.

```
frames[0] = address inside backtrace_resource_adaptor::print_backtrace()
frames[1] = address inside backtrace_resource_adaptor::do_allocate()
frames[2] = address inside rmm pool MR (internal, private symbol)
frames[3] = address inside cudf::groupby::detail::hash::groupby (exported symbol)
...
```

### `backtrace_symbols(void** frames, int n)` — `<execinfo.h>` (Dynamic Symbols only)
Converts virtual addresses to strings by calling `dladdr()` on each one.
Only resolves symbols present in `.dynsym`.

Output format per frame:
```
/path/to/lib.so(mangled_name+0xOFFSET) [0xVIRTUAL_ADDR]
```
For unexported symbols: `lib.so(+0xOFFSET)` — no name.

### `dladdr(void* addr, Dl_info* info)` — `<dlfcn.h>`
Finds which mapped object contains `addr` and returns:

```c
Dl_info {
    dli_fname  // path to the .so file on disk
    dli_fbase  // virtual address where that .so was loaded (the load base)
    dli_sname  // nearest exported symbol name (.dynsym only) — NULL for private symbols
    dli_saddr  // address of that symbol
}
```

The adaptor uses this struct in two ways:

1. **Fast path** — if `dli_sname` is non-null (symbol is in `.dynsym`), demangle it
   with `abi::__cxa_demangle` and return immediately; no subprocess is needed.
2. **Slow path** — if `dli_sname` is null (symbol is private / not in `.dynsym`),
   use `dli_fname` and `dli_fbase` to compute the file offset for `addr2line`:
```cpp
uintptr_t offset = (uintptr_t)addr - (uintptr_t)info.dli_fbase;
```

### `addr2line -e <lib> -C -f 0x<offset>` — external binary
Reads `.symtab` (and `.debug_info` if present) from the ELF file on disk
to resolve a file offset to a function name (and optionally `file:line`).

`<offset>` is the **file offset** computed above: `virtual_addr - dli_fbase`.

- `-e <lib>` — which ELF file to read (`dli_fname` from `dladdr()`)
- `-C` — demangle C++ names
- `-f` — print the function name on the line above the `file:line`
- `0x<offset>` — the file offset (`virtual_addr - dli_fbase`)

**Example** — resolving `cudf::groupby::detail::hash::compute_single_pass_aggs` (file offset
`0xcadb80`, found via `nm -C $CONDA_PREFIX/lib/libcudf.so | grep compute_single_pass_aggs |
grep -v _GLOBAL__ | grep "^[0-9a-f]* t "`):

```bash
addr2line -e $CONDA_PREFIX/lib/libcudf.so -C -f 0xcadb80
```

Expected output (no DWARF in conda `libcudf.so`, so no `file:line`):
```
std::pair<rmm::device_uvector<int>, bool> cudf::groupby::detail::hash::compute_single_pass_aggs<...>
??:0
```

Returns `??` / `??:0` when the symbol is not found.


---

## 4. Resolution Pipeline in the Adaptor

The pipeline for each frame address is implemented in `addr2line_resolve(void* addr)`
(called from `print_backtrace()`) followed by a fallback `backtrace_symbols_resolve()`.

### Step-by-step flow

```
frame_addr
    │
    ▼
① cache hit?  ──► YES ──► return cached string  ✓
    │ NO
    ▼
dladdr(frame_addr, &info)
    │
    ├── dli_fname/dli_fbase NULL? ──► return ""  (not in any mapped .so)
    │
    ├── dli_sname non-null? (symbol is in .dynsym — exported)
    │       │ YES
    │       ▼
    │   abi::__cxa_demangle(dli_sname)  ──► demangled name  ✓  (no subprocess)
    │
    └── dli_sname null (symbol is private — not in .dynsym)
            │
            ▼
② addr2line available? (checked once at startup via `addr2line --version`)
    │
    ├── NO ──► [RMM] WARNING printed once  ──► return ""
    │
    └── YES
            │
            ▼
        offset = frame_addr - dli_fbase
        popen("addr2line -e dli_fname -C -f 0x<offset>")
            │
            ├── popen() failed? ──► [RMM] WARNING with offset  ──► return ""
            │
            └── Success: read line1 (function name) + line2 (file:line)
                    │
                    ├── line1 == "??" ──► .symtab absent  ──► return ""
                    │
                    └── resolved name (+ file:line if not "??:0")  ✓

③ addr2line_resolve() returned ""?
    │ YES
    ▼
backtrace_symbols_resolve(syms[i])
    │
    │  Parses "lib.so(mangled+0xOFFSET) [0xVIRTUAL]" from backtrace_symbols()
    │  [RMM] WARNING printed once (this path should not fire in normal use)
    │
    ├── mangled name found? ──► abi::__cxa_demangle  ──► demangled+offset  ✓
    └── no name in string   ──► raw string ("lib.so(+0xOFFSET)") as last resort
```

### When each step fires in practice

| Step | Symbols resolved | Examples |
|------|-----------------|----------|
| ① cache | any previously seen address | all repeated call paths |
| `dli_sname` fast path | exported symbols in `.dynsym` | `cudf::groupby::aggregate`, libc, libstdc++ |
| `addr2line` | private/internal symbols in `.symtab` | `cudf::groupby::detail::hash::compute_single_pass_aggs` |
| `backtrace_symbols_resolve` | should not fire — stripped .so only | fully stripped `libc.so` (rare) |

### `addr2line` availability check

To avoid the overhead of a failed `fork` on every cache miss when `addr2line` is
not installed, the adaptor probes once at startup:

```cpp
std::call_once(availability_checked, [] {
    FILE* probe = ::popen("addr2line --version 2>/dev/null", "r");
    if (probe) {
        char buf[4];
        addr2line_available = (std::fgets(buf, sizeof(buf), probe) != nullptr);
        ::pclose(probe);
    }
    if (!addr2line_available)
        std::cerr << "[RMM] WARNING: addr2line not found on PATH\n";
});
```

### Cache

Every `(frame_addr → resolved string)` result is stored in a static
`unordered_map` protected by a `mutex`.  On subsequent allocations that
share the same call path, neither the `dladdr()` lookup nor the
`popen()`/`addr2line` subprocess is repeated.

---

## 5. Why Some Frames Still Show as `lib.so(+0xOFFSET)`

1. `addr2line` returned `??` — the `.symtab` was stripped from that library
2. `dladdr()` failed — the address is in an anonymous mapping (vDSO, JIT stub)
3. `addr2line` is not installed on the system (warning printed once on stderr)
4. The symbol was compiled with `-fvisibility=hidden` and genuinely has no
   name in any table

For `libcudf.so` from the rapidsai conda channel (aarch64), `.symtab` is
**present** — so all internal frames resolve.  If you see bare offsets,
they are typically from `libc.so` or `libcuda.so` which are fully stripped.
