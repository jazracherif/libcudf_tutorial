#pragma once

// ---------------------------------------------------------------------------
// backtrace_resource_adaptor
//
// An RMM device_memory_resource adaptor that prints a fully-resolved CPU call
// stack to stdout on every allocate/deallocate call.
//
// Symbol resolution strategy (best available, in order):
//   1. dladdr() dli_sname — free in-process lookup against .dynsym; covers
//                    all exported symbols without forking a subprocess.
//   2. addr2line   — reads .symtab from the mapped .so on disk; resolves
//                    internal/private symbols not present in .dynsym.
//   3. backtrace_symbols() fallback — for frames where addr2line is not
//                    available or returns "??".
//   4. Raw offset  — "library.so(+0xOFFSET)" as a last resort.
//
// Usage:
//   auto* upstream = rmm::mr::get_current_device_resource();
//   backtrace_resource_adaptor bt_mr{upstream};
//   rmm::mr::set_current_device_resource(&bt_mr);
// ---------------------------------------------------------------------------

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cstdio>       // popen(), pclose(), fgets()
#include <cstdlib>      // free()
#include <cxxabi.h>     // abi::__cxa_demangle
#include <dlfcn.h>      // dladdr(), Dl_info
#include <execinfo.h>   // backtrace(), backtrace_symbols()
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class backtrace_resource_adaptor : public rmm::mr::device_memory_resource {
public:
    /// @param upstream     MR to delegate actual allocations to (must outlive this).
    /// @param max_frames   Maximum call-stack depth to capture (default 32).
    /// @param skip_frames  Internal frames to omit from output (default 3).
    explicit backtrace_resource_adaptor(rmm::mr::device_memory_resource* upstream,
                                        int max_frames  = 32,
                                        int skip_frames = 3)
        : upstream_(upstream),
          max_frames_(max_frames),
          skip_frames_(skip_frames)
    {}

private:
    rmm::mr::device_memory_resource* upstream_;
    int max_frames_;
    int skip_frames_;

    // -----------------------------------------------------------------------
    // Resolve one address to a human-readable symbol name.
    // Results are cached — each unique address is only resolved once.
    // Returns an empty string if no name can be determined.
    //
    // Resolution order:
    //   1. dladdr() dli_sname — fast in-process .dynsym lookup; covers all
    //      exported symbols without forking a subprocess.
    //   2. addr2line          — forks a subprocess to read .symtab on disk;
    //      resolves private/internal symbols absent from .dynsym.
    // -----------------------------------------------------------------------
    static std::string addr2line_resolve(void* addr)
    {
        // Cache: resolve each unique address only once.
        static std::unordered_map<void*, std::string> cache;
        static std::mutex cache_mutex;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(addr);
            if (it != cache.end()) return it->second;
        }

        // Compute the result in a lambda so each resolution path can return
        // naturally; the outer function stores in the cache exactly once.
        auto resolve = [&]() -> std::string {
            Dl_info info{};
            if (!::dladdr(addr, &info) || !info.dli_fname) return {};

            // --- Step 1: try dladdr()'s dli_sname (.dynsym — exported symbols).
            // Free in-process lookup; demangle and return immediately if found.
            if (info.dli_sname && info.dli_sname[0] != '\0') {
                int status = 0;
                char* dem = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
                std::string r = (status == 0 && dem) ? dem : info.dli_sname;
                std::free(dem);
                return r;
            }

            // --- Step 2: dli_sname was null — symbol is private (not in .dynsym).
            // Use addr2line to read .symtab from the .so file on disk.
            //   dli_fname — path to the .so  (for -e flag)
            //   dli_fbase — runtime load base (to undo ASLR:
            //               file_offset = virtual_addr - dli_fbase)

            // Check once at startup whether addr2line is available on PATH.
            // If it isn't, skip all popen() calls rather than paying the fork
            // overhead on every cache miss only to get "not found" back.
            static std::once_flag availability_checked;
            static bool addr2line_available = false;
            std::call_once(availability_checked, [] {
                FILE* probe = ::popen("addr2line --version 2>/dev/null", "r");
                if (probe) {
                    char buf[4];
                    addr2line_available = (std::fgets(buf, sizeof(buf), probe) != nullptr);
                    ::pclose(probe);
                }
                if (!addr2line_available)
                    std::cerr << "[RMM] WARNING: addr2line not found on PATH — "
                                 "private symbols will not be resolved.\n";
            });
            if (!addr2line_available) return {};

            auto offset = reinterpret_cast<uintptr_t>(addr)
                        - reinterpret_cast<uintptr_t>(info.dli_fbase);

            char cmd[512];
            std::snprintf(cmd, sizeof(cmd),
                          "addr2line -e '%s' -C -f 0x%lx 2>/dev/null",
                          info.dli_fname,
                          static_cast<unsigned long>(offset));

            // Spawn a new process via popen() to run addr2line.
            // This is the only subprocess fork in the entire adaptor; it is
            // guarded by the per-address cache so each unique PC is resolved
            // at most once for the lifetime of the process.
            FILE* fp = ::popen(cmd, "r");
            if (!fp) {
                // popen() itself failed (e.g. fork limit hit, no shell).
                std::cerr << "[RMM] WARNING: popen(addr2line) failed for offset 0x"
                          << std::hex << offset << std::dec << "\n";
                return {};
            }

            char line1[512]{}, line2[512]{};
            bool got1 = (std::fgets(line1, sizeof(line1), fp) != nullptr);
            bool got2 = (std::fgets(line2, sizeof(line2), fp) != nullptr);
            ::pclose(fp);

            if (!got1) return {};

            for (char* s : {line1, line2})
                for (char* p = s + std::strlen(s) - 1;
                     p >= s && (*p == '\n' || *p == '\r'); --p)
                    *p = '\0';

            if (std::string(line1) == "??") return {};

            std::string r = line1;
            if (got2 && std::string(line2) != "??:0")
                r += "\t(" + std::string(line2) + ")";
            return r;
        };

        std::string result = resolve();

        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[addr] = result;
        return result;
    }

    // -----------------------------------------------------------------------
    // Fallback resolver: parse a backtrace_symbols() string of the form
    //   "lib.so(mangled_name+0xOFFSET) [0xVIRTUAL]"
    // and return the demangled name + offset if possible, or the raw string.
    //
    // This is only reached when addr2line_resolve() returns empty, which means
    // dladdr() found no name in .dynsym AND .symtab had nothing either (fully
    // stripped .so) or addr2line is unavailable.
    // -----------------------------------------------------------------------
    static std::string backtrace_symbols_resolve(const char* raw_sym)
    {
        // Warn once so we know this fallback is actually being hit.
        static std::once_flag warned;
        std::call_once(warned, [] {
            std::cerr << "[RMM] WARNING: backtrace_symbols_resolve fallback triggered "
                         "(addr2line_resolve returned empty — addr2line unavailable or .so fully stripped).\n";
        });

        std::string raw(raw_sym ? raw_sym : "?");
        auto paren_open  = raw.find('(');
        auto paren_plus  = raw.find('+', paren_open);
        auto paren_close = raw.find(')', paren_plus);

        if (paren_open  != std::string::npos &&
            paren_plus  != std::string::npos &&
            paren_close != std::string::npos)
        {
            std::string mangled = raw.substr(paren_open + 1,
                                             paren_plus - paren_open - 1);
            if (!mangled.empty()) {
                int status = 0;
                char* dem = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
                if (status == 0 && dem) {
                    std::string r = dem + raw.substr(paren_plus, paren_close - paren_plus + 1);
                    std::free(dem);
                    return r;
                }
                std::free(dem);
            }
        }

        return raw;  // raw string as last resort
    }

    // -----------------------------------------------------------------------
    // Capture and print the call stack, skipping the first skip_frames_
    // internal frames.
    // -----------------------------------------------------------------------
    void print_backtrace(char const* action, void* ptr, std::size_t bytes) const
    {
        std::vector<void*> frames(static_cast<std::size_t>(max_frames_));
        int n = ::backtrace(frames.data(), max_frames_);
        char** syms = ::backtrace_symbols(frames.data(), n);

        std::cout << "[RMM] " << action
                  << "  ptr=" << ptr
                  << "  bytes=" << bytes << "\n";

        for (int i = skip_frames_; i < n; ++i) {
            int idx = i - skip_frames_;

            // --- Primary resolution (covers virtually all frames in practice):
            //
            //   · Exported symbols (libc, libstdc++, CUDA runtime, cuDF public API,
            //     rmm, the main binary): resolved instantly via dladdr() dli_sname
            //     — no subprocess, no disk I/O.
            //
            //   · Private/internal symbols (e.g. cudf::groupby::detail::hash::*):
            //     not in .dynsym so dli_sname is null; resolved by forking addr2line
            //     which reads .symtab from the .so file on disk.  First-call cost is
            //     amortised by the per-address cache; repeated frames are free.
            //
            std::string resolved = addr2line_resolve(frames[i]);

            // --- Fallback (exceptional cases only — should not fire in normal use):
            //
            //   Reached only when addr2line_resolve() returns empty, i.e.:
            //     · dladdr() fails entirely: the address is not inside any mapped
            //       shared object (e.g. vsyscall/vDSO trampoline, JIT stub, or a
            //       frame captured before .so mapping completes).
            //     · The .so is fully stripped (strip --strip-all removed .symtab)
            //       AND the symbol is not in .dynsym.
            //     · addr2line binary is not installed on the system.
            //
            //   In all these cases backtrace_symbols() has no name either, so this
            //   path produces "lib.so(+0xOFFSET)" at best, or "?" at worst.
            if (resolved.empty())
                resolved = backtrace_symbols_resolve(syms ? syms[i] : nullptr);

            std::cout << "  #" << idx << "  " << resolved << "\n";
        }

        std::free(syms);
        std::cout << "\n";
    }

    // ---- rmm::mr::device_memory_resource interface -------------------------

    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
    {
        void* ptr = upstream_->allocate(stream, bytes);
        print_backtrace("allocate  ", ptr, bytes);
        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t bytes,
                       rmm::cuda_stream_view stream) noexcept override
    {
        upstream_->deallocate(stream, ptr, bytes);
        print_backtrace("deallocate", ptr, bytes);
    }

    bool do_is_equal(rmm::mr::device_memory_resource const& other)
        const noexcept override
    {
        return this == &other;
    }
};
