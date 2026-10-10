"""
The 'cpp' backend: compiles the SAME generated kernel source as the GPU
path into a plain C++ shared library and runs it on the host, one thread per
core, each thread looping over its share of the conditions.

Reusing the generated source byte-for-byte gives full logic parity with the
CPU-matched GPU solver; only the execution substrate differs.
"""
import ctypes
import hashlib
import os
import platform
import shutil
import subprocess

import numpy as np


def _compile_command(lib_path, src_path, defines):
    """Platform-appropriate compile command for the backend library.

    -ffp-contract=off is REQUIRED for parity with the (non-FMA) reference
    Cython CPU build: with contraction enabled, degenerate phase-selection
    decisions can flip (measured on AlCuFe cond 47).

    The backend is SINGLE-THREADED by design (no OpenMP): one pycalphad call
    uses one core, and users parallelize by running multiple pycalphad calls
    in their own threads or processes. This also removes the libgomp/libomp
    runtime dependency (notably simplifying macOS installs).
    """
    system = platform.system()
    extra = os.environ.get('PYCGPU_CPU_EXTRA_CFLAGS', '').split()
    common = ['-std=c++17', '-O3', '-march=native', '-ffp-contract=off',
              '-shared', '-fPIC'] + extra + ['-o', lib_path, src_path]
    if system == 'Darwin':
        cxx = shutil.which('clang++')
        if cxx is None:
            raise RuntimeError("CPU backend needs clang++ (Xcode command line tools) on macOS")
        return [cxx] + common + defines
    cxx = shutil.which('g++') or shutil.which('clang++')
    if cxx is None:
        hint = ("install MinGW-w64 g++ or use WSL" if system == 'Windows'
                else "install g++ (e.g. `apt install g++`)")
        raise RuntimeError(f"CPU backend needs a C++17 compiler on PATH: {hint}")
    if system == 'Windows':
        # Python >= 3.8 removed PATH from the Windows DLL search, so a
        # MinGW-built library whose runtime deps (libstdc++-6.dll,
        # libgcc_s_seh-1.dll, libwinpthread-1.dll) live in the MinGW bin
        # directory fails to load with "Could not find module ... or one of
        # its dependencies". Link the runtimes statically so the produced
        # library is self-contained.
        common = common + ['-static', '-static-libgcc', '-static-libstdc++']
    return [cxx] + common + defines

_CPU_DRIVER_SRC = r"""
#include <thread>
#include <atomic>
#include <vector>
// ===== CPU backend driver (appended by pycalphad.gpu.cpu_backend) =====
// Single-threaded by design: parallelism is the caller's job (multiple
// pycalphad calls in threads/processes; thread_local shims in cpu_compat.h
// keep concurrent calls from different threads safe).

extern "C" void pycgpu_cpu_grid_eval(int model_idx, const double* dof, double* out,
                                     long long n_points, int dof_stride)
{
    init_all_gpu_phase_records();
    for (long long i = 0; i < n_points; ++i) {
        out[i] = g_phase_records_array[model_idx].obj(&dof[i * (long long)dof_stride]);
    }
}

#include <cstdlib>
#include <fenv.h>

// Debug aid (PYCGPU_CPU_SNAN=1, LINUX/GLIBC ONLY): before each condition,
// fill a large region of the stack below the driver frame with
// signaling-NaN doubles. Any arithmetic USE of an unwritten (stale) stack
// double then raises FE_INVALID, promoted to SIGFPE so gdb stops at the
// exact faulting instruction. feenableexcept/fedisableexcept are glibc
// extensions; on other platforms the flag is inert.
#if defined(__GLIBC__)
#define PYCGPU_HAVE_SNAN_DEBUG 1
#include <alloca.h>
static void __attribute__((noinline)) pycgpu_paint_stack(long long nbytes)
{
    unsigned long long* p = (unsigned long long*)alloca(nbytes);
    for (long long i = 0; i < nbytes / 8; ++i) p[i] = 0x7FF0000000000001ull; // sNaN
    __asm__ __volatile__("" :: "r"(p) : "memory");
}
#endif

extern "C" void pycgpu_cpu_point_hull(
    const double* grid_X, const double* grid_GM,
    const long long* x_base_row, const long long* gm_base_row,
    const int* m_points, int num_components,
    const int* fixed_chempot_indices, const int* num_fixed_chempots,
    const double* lincomb_coefs, const double* lincomb_rhs,
    const int* num_lincomb, int max_lincomb,
    double* chemical_potentials, double* out_energy,
    double* result_fractions, int* result_simplex, int n_conditions, int n_threads)
{
    // Threads over conditions: conditions are independent, outputs are
    // per-condition slots, and the CUDA-index shims are thread_local, so
    // results are bit-identical at any thread count.
    if (n_threads <= 0) n_threads = (int)std::thread::hardware_concurrency();
    if (n_threads > n_conditions) n_threads = n_conditions;
    if (n_threads > 1) {
        std::atomic<int> next_t(0);
        auto worker = [&]() {
            for (;;) {
                const int t = next_t.fetch_add(1, std::memory_order_relaxed);
                if (t >= n_conditions) break;
                threadIdx.x = (unsigned int)t;
                blockIdx.x = 0u;
                blockDim.x = 0u;
                point_hull_kernel(grid_X, grid_GM, x_base_row, gm_base_row,
                                  m_points, num_components,
                                  fixed_chempot_indices, num_fixed_chempots,
                                  lincomb_coefs, lincomb_rhs, num_lincomb, max_lincomb,
                                  chemical_potentials, out_energy,
                                  result_fractions, result_simplex, n_conditions);
            }
        };
        std::vector<std::thread> pool;
        pool.reserve((size_t)n_threads);
        for (int i = 0; i < n_threads; ++i) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
        return;
    }
    for (int t = 0; t < n_conditions; ++t) {
        threadIdx.x = (unsigned int)t;
        blockIdx.x = 0u;
        blockDim.x = 0u;
        point_hull_kernel(grid_X, grid_GM, x_base_row, gm_base_row,
                          m_points, num_components,
                          fixed_chempot_indices, num_fixed_chempots,
                          lincomb_coefs, lincomb_rhs, num_lincomb, max_lincomb,
                          chemical_potentials, out_energy,
                          result_fractions, result_simplex, n_conditions);
    }
}

extern "C" void pycgpu_cpu_run_all(
    const void* global_spec_ptr_raw,
    const void* condition_args_list_ptr_raw,
    void* results_list_ptr_raw,
    int num_conditions_total,
    int condition_stride,
    int python_max_statevars,
    const void* initial_phase_data_ptr,
    int initial_phase_data_stride,
    int system_spec_stride,
    const void* grid_data_ptr_raw,
    double* debug_gm_history,
    double* debug_mu_history,
    int* debug_convergence_history,
    int* debug_iteration_count,
    int debug_max_steps,
    const void* work_arrays,
    const int* grid_block_indices,
    long long grid_block_stride_bytes,
    int max_solver_iterations,
    int n_threads)
{
    init_all_gpu_phase_records();
    const int t_start = 0;
    // Threads over conditions (0 = all hardware threads). Safe by
    // construction: each condition is independent, indexes its OWN
    // work-array slot, and the CUDA-index shims (threadIdx/blockIdx) are
    // thread_local, so per-condition results are BIT-IDENTICAL for any
    // thread count.
    if (n_threads <= 0) n_threads = (int)std::thread::hardware_concurrency();
    if (n_threads > num_conditions_total - t_start) n_threads = num_conditions_total - t_start;
    if (n_threads < 1) n_threads = 1;
#ifdef PYCGPU_HAVE_SNAN_DEBUG
    const bool dbg_snan = (std::getenv("PYCGPU_CPU_SNAN") != nullptr);
    if (dbg_snan) {
        feclearexcept(FE_ALL_EXCEPT);
        feenableexcept(FE_INVALID);
    }
#endif
    bool run_parallel = (n_threads > 1);
#ifdef PYCGPU_HAVE_SNAN_DEBUG
    if (dbg_snan) run_parallel = false;  // FP-exception trapping is per-thread; keep the debug aid serial
#endif
    if (run_parallel) {
        std::atomic<int> next_t(t_start);
        auto worker = [&]() {
            for (;;) {
                const int t = next_t.fetch_add(1, std::memory_order_relaxed);
                if (t >= num_conditions_total) break;
                threadIdx.x = (unsigned int)t;
                blockIdx.x = 0u;
                blockDim.x = 0u;
                top_level_equilibrium_kernel(global_spec_ptr_raw, condition_args_list_ptr_raw,
                    results_list_ptr_raw, num_conditions_total, condition_stride,
                    python_max_statevars, initial_phase_data_ptr, initial_phase_data_stride,
                    system_spec_stride, grid_data_ptr_raw, debug_gm_history, debug_mu_history,
                    debug_convergence_history, debug_iteration_count, debug_max_steps,
                    (const WorkArrays*)work_arrays, grid_block_indices, grid_block_stride_bytes,
                    max_solver_iterations);
            }
        };
        std::vector<std::thread> pool;
        pool.reserve((size_t)n_threads);
        for (int i = 0; i < n_threads; ++i) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
    } else
    for (int t = t_start; t < num_conditions_total; ++t) {
#ifdef PYCGPU_HAVE_SNAN_DEBUG
        if (dbg_snan) pycgpu_paint_stack(4ll * 1024 * 1024);
#endif
        // Make tid = blockDim.x * blockIdx.x + threadIdx.x == t
        threadIdx.x = (unsigned int)t;
        blockIdx.x = 0u;
        blockDim.x = 0u;
        top_level_equilibrium_kernel(global_spec_ptr_raw, condition_args_list_ptr_raw,
            results_list_ptr_raw, num_conditions_total, condition_stride,
            python_max_statevars, initial_phase_data_ptr, initial_phase_data_stride,
            system_spec_stride, grid_data_ptr_raw, debug_gm_history, debug_mu_history,
            debug_convergence_history, debug_iteration_count, debug_max_steps,
            (const WorkArrays*)work_arrays, grid_block_indices, grid_block_stride_bytes,
            max_solver_iterations);
    }
#ifdef PYCGPU_HAVE_SNAN_DEBUG
    if (dbg_snan) {
        fedisableexcept(FE_ALL_EXCEPT);  // don't let numpy trap afterwards
        feclearexcept(FE_ALL_EXCEPT);
    }
#endif
}
"""


def build_cpu_library(full_kernel_source: str, define_flags, cache_dir: str, verbose: bool = False,
                      driver_src: str = None):
    """Compile the generated kernel source + driver into a shared library.

    driver_src defaults to the full equilibrium driver; lightweight modules
    (e.g. the property-evaluation module for calculate() outputs) pass their
    own driver since the default one references solver symbols.
    """
    gpu_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(gpu_dir, "cpu_compat.h")) as f:
        compat = f.read()

    source = compat + "\n" + full_kernel_source + "\n" + (_CPU_DRIVER_SRC if driver_src is None else driver_src)
    defines = [d for d in define_flags if d.startswith("-D")]
    extra_cflags = os.environ.get('PYCGPU_CPU_EXTRA_CFLAGS', '')
    # platform token: Windows gained static-runtime link flags (v2) — caches
    # built before that produce unloadable libraries and must not be reused
    plat_token = f"{platform.system()}-v2" if platform.system() == 'Windows' else ''
    tag = hashlib.md5((source + "|".join(sorted(defines)) + extra_cflags
                       + plat_token).encode()).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    src_path = os.path.join(cache_dir, f"{tag}_cpu.cpp")
    lib_path = os.path.join(cache_dir, f"{tag}_cpu.so")

    if not os.path.exists(lib_path):
        with open(src_path, "w") as f:
            f.write(source)
        cmd = _compile_command(lib_path, src_path, defines)
        with open(os.path.join(cache_dir, f"{tag}_cmd.txt"), "w") as f:
            f.write(" ".join(cmd) + "\n")
        if verbose:
            print(f"[CPU-C++] Compiling backend library: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CPU backend compilation failed:\n{result.stderr[-4000:]}")
        if verbose and result.stderr:
            print(f"[CPU-C++] compiler warnings: {result.stderr[-1000:]}")
    elif verbose:
        print(f"[CPU-C++] Using cached backend library {lib_path}")

    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        if platform.system() == 'Windows':
            # fallback: legacy DLL search (PATH included) for environments
            # where static linking was unavailable and the MinGW runtime
            # lives next to g++ on PATH
            lib = ctypes.CDLL(lib_path, winmode=0)
        else:
            raise
    if driver_src is not None:
        # Custom-driver module (e.g. the property-evaluation module): only the
        # driver's own exports exist; the caller sets their prototypes.
        return lib
    hull_fn = lib.pycgpu_cpu_point_hull
    hull_fn.restype = None
    hull_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_int,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    fn = lib.pycgpu_cpu_run_all
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                   ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,
                   ctypes.c_int, ctypes.c_int]
    return lib


def run_cpu_backend(lib, *, system_spec, condition_args_doubles, results,
                    num_conditions, condition_stride, python_max_statevars,
                    initial_phase_data, initial_phase_data_stride, system_spec_stride,
                    grid_data, grid_block_indices, grid_block_stride_bytes,
                    work_arrays_ptr_table, max_solver_iterations=1000, threads=None,
                    verbose=False):
    """Run the single-threaded C++ solver directly on host numpy buffers (zero copies).

    All array arguments are contiguous host numpy arrays allocated by the
    (backend-agnostic) pipeline; `work_arrays_ptr_table` is the uint64 table of
    host addresses in WorkArrays slot order. Results are written in place.
    """
    def _p(a):
        return a.ctypes.data if a is not None else None

    lib.pycgpu_cpu_run_all(
        _p(system_spec), _p(condition_args_doubles), _p(results),
        num_conditions, condition_stride, python_max_statevars,
        _p(initial_phase_data), initial_phase_data_stride, system_spec_stride,
        _p(grid_data),
        None, None, None, None, 0,
        _p(work_arrays_ptr_table),
        _p(grid_block_indices),
        ctypes.c_longlong(grid_block_stride_bytes),
        int(max_solver_iterations), solver_threads() if threads is None else int(threads))

    if verbose:
        print(f"[CPU-C++] Solved {num_conditions} conditions on host")


def solver_threads():
    """Worker threads for the C++ solver and hull (0 = all cores)."""
    from pycalphad.backend import solver_threads as _threads
    return _threads()


def solve_rows(lib, *, sizes, rows, system_spec, condition_args, initial_phase_data, results,
               grid_data, grid_block_indices, grid_block_stride_bytes,
               condition_stride, python_max_statevars, initial_phase_data_stride,
               system_spec_stride, results_per_condition, max_solver_iterations=1000,
               threads=None, memory_budget=None, jansson=None):
    """Solve the conditions listed in `rows` (indices into the packed
    per-condition buffers) and write their results rows in place, and their
    Jansson delta rows into `jansson` ((n, PYJAN_COND_STRIDE)) for libraries
    built with the Jansson epilogue.

    Work arrays are allocated per chunk so the memory stays bounded."""
    from pycalphad.gpu.work_arrays import allocate_work_arrays, per_condition_bytes
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return
    take = lambda flat, stride: np.ascontiguousarray(flat.reshape(-1, stride)[rows]).reshape(-1)
    spec = take(system_spec, system_spec_stride)
    cond = take(condition_args, condition_stride)
    ipd = take(initial_phase_data, initial_phase_data_stride)
    res_view = results.reshape(-1, results_per_condition)
    res = np.ascontiguousarray(res_view[rows]).reshape(-1)
    gbi = (np.ascontiguousarray(np.asarray(grid_block_indices, dtype=np.int32)[rows])
           if grid_block_indices is not None else None)
    budget = memory_budget or (1 << 30)
    chunk = max(1, min(rows.size, budget // per_condition_bytes(sizes)))
    _, table = allocate_work_arrays(np, sizes, chunk, lambda a: a.ctypes.data)
    work = _  # keep the arrays alive while the table is in use
    jansson_stride = jansson.shape[1] if jansson is not None else 0
    for s in range(0, rows.size, chunk):
        e = min(s + chunk, rows.size)
        chunk_res = res[s * results_per_condition:e * results_per_condition]
        if jansson_stride:
            # The library writes the delta blocks after the chunk's result
            # records, so the chunk gets a private buffer with room for them.
            chunk_res = np.zeros((e - s) * (results_per_condition + jansson_stride))
        run_cpu_backend(
            lib, system_spec=spec[s * system_spec_stride:e * system_spec_stride],
            condition_args_doubles=cond[s * condition_stride:e * condition_stride],
            results=chunk_res,
            num_conditions=e - s, condition_stride=condition_stride,
            python_max_statevars=python_max_statevars,
            initial_phase_data=ipd[s * initial_phase_data_stride:e * initial_phase_data_stride],
            initial_phase_data_stride=initial_phase_data_stride,
            system_spec_stride=system_spec_stride, grid_data=grid_data,
            grid_block_indices=gbi[s:e] if gbi is not None else None,
            grid_block_stride_bytes=grid_block_stride_bytes,
            work_arrays_ptr_table=table, max_solver_iterations=max_solver_iterations,
            threads=threads)
        if jansson_stride:
            n_rec = (e - s) * results_per_condition
            res[s * results_per_condition:e * results_per_condition] = chunk_res[:n_rec]
            jansson[rows[s:e]] = chunk_res[n_rec:].reshape(e - s, jansson_stride)
    del work
    res_view[rows] = res.reshape(-1, results_per_condition)
