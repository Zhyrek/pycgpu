"""Build a generated Metal equilibrium module as host C++ for debugging.

    python tools/metal_debug.py path/to/module.metal [--printf] [--asan]
        ->  prints the .dylib path

The library exports run_all(), which loops pycgpu_equilibrium_kernel over
conditions exactly as the GPU would. Remove the `#define printf(...)` line
from metal_compat.h's output (or pass --printf) to get device-side prints.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PRINTF_SHIM = r"""
template <class T> static inline T pyc_print_arg(T v) { return v; }
static inline double pyc_print_arg(pyc_real v) {
    union { ulong u; double d; } c; c.u = pyc_to_bits(v); return c.d;
}
template <class... A> static inline int pyc_printf(const char* f, A... a) {
    return std::fprintf(stdout, f, pyc_print_arg(a)...);
}
#define printf pyc_printf
"""

DRIVER = r"""
extern "C" void run_all(const void* spec, const void* cond, void* res, const void* ipd,
                        const void* grid, const int* gbi, const PycgpuEqParams* p,
                        int first, int count)
{
#if defined(__aarch64__)
    if (std::getenv("PYCGPU_EMU_FTZ")) {  // flush subnormals to zero, as Apple GPUs do
        unsigned long fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        fpcr |= (1ul << 24);
        __asm__ volatile("msr fpcr, %0" :: "r"(fpcr));
    }
#endif
    for (int g = first; g < first + count; ++g) {
        pycgpu_equilibrium_kernel((const pyc_f64*)spec, (const pyc_f64*)cond, (pyc_f64*)res,
                                  (const pyc_f64*)ipd, (const char*)grid, gbi, *p, (uint)g);
    }
}
"""


def build(metal_path, keep_printf=False, opt='-O1', extra_flags=()):
    src = open(metal_path).read()
    src = src.replace('#include <metal_stdlib>', f'#include "{os.path.join(HERE, "msl_on_cpu.h")}"\n#include <cstdlib>')
    src = src.replace('using namespace metal;', '')
    if keep_printf:
        # device printf: pyc_real arguments print as doubles
        src = src.replace('#define printf(...) ((void)0)', PRINTF_SHIM)
    # attributes Metal uses on kernel parameters mean nothing to the host compiler
    src = re.sub(r'\[\[(buffer\(\d+\)|thread_position_in_grid)\]\]', '', src)
    out = metal_path + ('.printf' if keep_printf else '') + '.cpu.cpp'
    with open(out, 'w') as f:
        f.write(src + DRIVER)
    lib = out[:-4] + '.dylib'
    cmd = ['clang++', '-std=c++17', opt, '-g', '-shared', '-fPIC', '-w', '-ffp-contract=off',
           *extra_flags, out, '-o', lib]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-6000:])
    return lib


if __name__ == '__main__':
    print(build(sys.argv[1], keep_printf='--printf' in sys.argv,
                extra_flags=['-fsanitize=address'] if '--asan' in sys.argv else []))
