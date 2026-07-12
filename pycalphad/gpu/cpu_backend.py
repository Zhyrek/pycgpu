"""
CPU backend (initial pass): compiles the SAME generated kernel source as the
GPU path into a plain C++ shared library (single-threaded loop over conditions)
and runs it on the host. Enabled with PYCGPU_CPU=1 alongside gpu=True.

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
    return [cxx] + common + defines

_CPU_DRIVER_SRC = r"""
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
    double* result_fractions, int* result_simplex, int n_conditions)
{
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
    int max_solver_iterations)
{
    init_all_gpu_phase_records();
    // PYCGPU_CPU_TSTART=<t>: start the serial loop at condition t (debug aid
    // for isolating cross-condition state carryover; earlier conditions keep
    // their initialization values in the results buffer).
    const char* tstart_env = std::getenv("PYCGPU_CPU_TSTART");
    const int t_start = tstart_env ? atoi(tstart_env) : 0;
#ifdef PYCGPU_HAVE_SNAN_DEBUG
    const bool dbg_snan = (std::getenv("PYCGPU_CPU_SNAN") != nullptr);
    if (dbg_snan) {
        feclearexcept(FE_ALL_EXCEPT);
        feenableexcept(FE_INVALID);
    }
#endif
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


def build_cpu_library(full_kernel_source: str, define_flags, cache_dir: str, verbose: bool = False):
    """Compile the generated kernel source + OpenMP driver into a shared library."""
    gpu_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(gpu_dir, "cpu_compat.h")) as f:
        compat = f.read()

    source = compat + "\n" + full_kernel_source + "\n" + _CPU_DRIVER_SRC
    defines = [d for d in define_flags if d.startswith("-D")]
    extra_cflags = os.environ.get('PYCGPU_CPU_EXTRA_CFLAGS', '')
    tag = hashlib.md5((source + "|".join(sorted(defines)) + extra_cflags).encode()).hexdigest()
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

    lib = ctypes.CDLL(lib_path)
    hull_fn = lib.pycgpu_cpu_point_hull
    hull_fn.restype = None
    hull_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_int,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    fn = lib.pycgpu_cpu_run_all
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                   ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,
                   ctypes.c_int]
    return lib


def run_cpu_backend(lib, *, system_spec, condition_args_doubles, results,
                    num_conditions, condition_stride, python_max_statevars,
                    initial_phase_data, initial_phase_data_stride, system_spec_stride,
                    grid_data, grid_block_indices, grid_block_stride_bytes,
                    work_arrays_ptr_table, max_solver_iterations=1000, verbose=False):
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
        int(max_solver_iterations))

    if verbose:
        print(f"[CPU-C++] Solved {num_conditions} conditions on host (single-threaded)")
