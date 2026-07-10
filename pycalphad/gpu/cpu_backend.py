"""
CPU backend (initial pass): compiles the SAME generated kernel source as the
GPU path into a plain C++ shared library (OpenMP over conditions) and runs it
on the host. Enabled with PYCGPU_CPU=1 alongside gpu=True.

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
    """Platform-appropriate compile command for the OpenMP backend library.

    -ffp-contract=off is REQUIRED for parity with the (non-FMA) reference
    Cython CPU build: with contraction enabled, degenerate phase-selection
    decisions can flip (measured on AlCuFe cond 47).
    """
    system = platform.system()
    common = ['-std=c++17', '-O3', '-march=native', '-ffp-contract=off',
              '-shared', '-fPIC', '-o', lib_path, src_path]
    if system == 'Darwin':
        cxx = shutil.which('clang++')
        if cxx is None:
            raise RuntimeError("CPU backend needs clang++ (Xcode command line tools) on macOS")
        # Apple clang has no bundled OpenMP runtime; needs `brew install libomp`.
        return [cxx, '-Xpreprocessor', '-fopenmp', '-lomp'] + common + defines
    cxx = shutil.which('g++') or shutil.which('clang++')
    if cxx is None:
        hint = ("install MinGW-w64 g++ or use WSL" if system == 'Windows'
                else "install g++ (e.g. `apt install g++`)")
        raise RuntimeError(f"CPU backend needs a C++17/OpenMP compiler on PATH: {hint}")
    return [cxx, '-fopenmp'] + common + defines

_CPU_DRIVER_SRC = r"""
// ===== CPU backend driver (appended by pycalphad.gpu.cpu_backend) =====
#include <omp.h>
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
    long long grid_block_stride_bytes)
{
    init_all_gpu_phase_records();
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < num_conditions_total; ++t) {
        // Make tid = blockDim.x * blockIdx.x + threadIdx.x == t
        threadIdx.x = (unsigned int)t;
        blockIdx.x = 0u;
        blockDim.x = 0u;
        top_level_equilibrium_kernel(global_spec_ptr_raw, condition_args_list_ptr_raw,
            results_list_ptr_raw, num_conditions_total, condition_stride,
            python_max_statevars, initial_phase_data_ptr, initial_phase_data_stride,
            system_spec_stride, grid_data_ptr_raw, debug_gm_history, debug_mu_history,
            debug_convergence_history, debug_iteration_count, debug_max_steps,
            (const WorkArrays*)work_arrays, grid_block_indices, grid_block_stride_bytes);
    }
}
"""


def build_cpu_library(full_kernel_source: str, define_flags, cache_dir: str, verbose: bool = False):
    """Compile the generated kernel source + OpenMP driver into a shared library."""
    gpu_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(gpu_dir, "cpu_compat.h")) as f:
        compat = f.read()

    source = compat + "\n" + full_kernel_source + "\n" + _CPU_DRIVER_SRC
    defines = [d for d in define_flags if d.startswith("-D")]
    tag = hashlib.md5((source + "|".join(sorted(defines))).encode()).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    src_path = os.path.join(cache_dir, f"{tag}_cpu.cpp")
    lib_path = os.path.join(cache_dir, f"{tag}_cpu.so")

    if not os.path.exists(lib_path):
        with open(src_path, "w") as f:
            f.write(source)
        cmd = _compile_command(lib_path, src_path, defines)
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
    fn = lib.pycgpu_cpu_run_all
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                   ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    return lib


def run_cpu_backend(lib, *, system_spec, condition_args_doubles, results,
                    num_conditions, condition_stride, python_max_statevars,
                    initial_phase_data, initial_phase_data_stride, system_spec_stride,
                    grid_data, grid_block_indices, grid_block_stride_bytes,
                    work_arrays_ptr_table, verbose=False):
    """Run the OpenMP solver directly on host numpy buffers (zero copies).

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
        ctypes.c_longlong(grid_block_stride_bytes))

    if verbose:
        print(f"[CPU-C++] Solved {num_conditions} conditions on host (OpenMP)")
