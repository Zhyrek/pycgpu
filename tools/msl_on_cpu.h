// Compile generated Metal Shading Language as host C++ for debugging.
//
// The Metal backend's kernels are lowered from the shared CUDA/C++ solver
// and run in emulated double precision (metal_compat.h). Metal has no
// printf or debugger hooks for compute kernels, so this shim lets the exact
// generated .metal source build as an ordinary shared library: address
// spaces collapse to nothing and the handful of Metal builtins the prelude
// uses map to libm. See tools/metal_debug.py.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <climits>
#include <algorithm>
#include <cstdio>
#define PYCGPU_MSL_ON_CPU 1
#define thread
#define device
#define constant const
#define threadgroup
#define kernel extern "C"
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long ulong;
struct float2 {
    float x, y;
    float2() = default;
    constexpr float2(float a, float b) : x(a), y(b) {}
    float2 operator-() const { return float2(-x, -y); }
};
using std::fma;
static inline float frexp(float x, int& e) { return std::frexp(x, &e); }
static inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
static inline int clz(ulong x) { return x ? __builtin_clzl(x) : 64; }
using std::fabs; using std::floor; using std::ceil; using std::trunc; using std::rint;
using std::round; using std::sqrt; using std::log; using std::exp; using std::ldexp;
using std::isnan; using std::isinf; using std::isfinite; using std::signbit; using std::max; using std::min;
namespace metal {}
