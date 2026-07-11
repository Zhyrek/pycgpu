#pragma once
// Compatibility shims so the generated GPU kernel source compiles as plain C++
// for the CPU backend (PYCGPU_CPU=1). CUDA qualifiers vanish; the thread-index
// builtins become variables the driver sets per condition so
// `tid = blockDim.x * blockIdx.x + threadIdx.x` yields the condition index.
// They stay thread_local NOT for internal parallelism (the backend is
// single-threaded by design) but so that concurrent pycalphad calls from
// multiple USER threads (ctypes releases the GIL) cannot race on them.
#define __device__
#define __global__
#define __host__
#define __forceinline__ inline

#include <cstring>
#include <cmath>
#include <cstdio>

struct pycgpu_cpu_dim3 { unsigned int x, y, z; };
inline thread_local pycgpu_cpu_dim3 threadIdx = {0u, 0u, 0u};
inline thread_local pycgpu_cpu_dim3 blockIdx  = {0u, 0u, 0u};
inline thread_local pycgpu_cpu_dim3 blockDim  = {0u, 1u, 1u};
inline thread_local pycgpu_cpu_dim3 gridDim   = {1u, 1u, 1u};

#include <chrono>
inline long long clock64() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}
