# Accelerated equilibrium backends

pycalphad ships two optional accelerated backends for `equilibrium()` and
`calculate()`. Both produce results validated against the reference solver
(full test suite green; GM/phase-set parity documented below) and are
selected at runtime — no rebuild of pycalphad is required.

```python
import pycalphad
pycalphad.set_backend('c++')    # single-core compiled backend
pycalphad.set_backend('gpu')    # CUDA backend (falls back with a clear error
                                # if CuPy/driver/nvcc are unavailable)

# or scoped:
with pycalphad.backend('c++'):
    eq = equilibrium(dbf, comps, phases, conds)

# or per call:
eq = equilibrium(dbf, comps, phases, conds, backend='c++')
```

That is the entire API: after `set_backend`, existing scripts work unchanged.
Problems the backends cannot handle exactly (non-standard conditions, custom
Model subclasses, dilute compositions, under/overdetermined inputs) are
routed to the reference solver automatically, including its validation
errors.

## Installation

| backend | install | system requirements |
|---|---|---|
| `c++` | `pip install pycalphad[cpp]` | a C++17 compiler on PATH (g++/clang++; Xcode CLT on macOS; MinGW-w64 or WSL on Windows) |
| `gpu` | `pip install pycalphad[gpu]` | NVIDIA driver; `nvcc` (see below) |

Notes:

- The `cpp` extra installs no additional Python packages — the backend
  compiles generated kernels at runtime with your system compiler and caches
  the shared libraries per system+model fingerprint.
- The `gpu` extra installs `cupy-cuda12x`, whose wheels bundle the CUDA
  runtime libraries (only the NVIDIA *driver* must be present). Kernel
  compilation uses `nvcc` when it is on PATH (conda `cuda-nvcc`, distro
  toolkit, or the `nvidia-cuda-nvcc-cu12` pip wheel); when no `nvcc` is
  found, kernels compile via **NVRTC**, the runtime-compilation library
  the CuPy wheel already ships — i.e. `pip install pycalphad[gpu]` works
  with no CUDA toolkit install at all. Force a choice with
  `PYCGPU_CUDA_COMPILER=nvcc|nvrtc` (the full stock test suite passes
  under both). On ROCm builds of CuPy the same switch maps to
  hipcc/hipRTC (the kernel sources carry guards for both RTC dialects;
  hipRTC is untested in CI, like the ROCm path generally).
- CUDA 11 machines: install `cupy-cuda11x` manually instead of the extra.
- AMD: the kernel sources are HIP-compatible and CuPy publishes experimental
  `cupy-rocm-*` wheels, but the ROCm path is untested in CI; treat it as
  experimental.
- The GPU backend uses everything the C++ backend uses; installing `[gpu]`
  effectively covers `[cpp]` (plus the system compiler requirement).

## What you get (measured, laptop RTX 4070 / single core, vs pycalphad 0.11.1)

| workload | reference | c++ | gpu |
|---|---|---|---|
| AlCuFe 245 equilibria | 7.3 s | 1.05 s | 3.7 s |
| AlCuFe 10,571 equilibria | 134 s | ~13 s | ~12 s |
| AuBi 1,002,000 equilibria | 584 s | 62 s | ~55 s |
| ESPEI ZPF likelihood (Cu-Mg, 7 datasets) | 1.94 s | 0.155 s | — |
| ESPEI 1024-chain MCMC ensemble step | ~33 min | ~45 s | ~12 s |

Correctness: the stock test suite passes under both backends (292 passed
each). Same-phase-set energy agreement is at numerical-precision scale for
typical systems; a small documented set of degenerate conditions differ at
the eps·cond level of the underlying linear algebra with identical phase
sets.

## Tuning (optional environment variables)

- `PYCGPU_DEVICE_HULL=0` — opt out of the compiled starting-point hull
  (default on; bit-identical results, up to 2x total wall at large batches).
- `PYCGPU_CHUNK=N` — conditions per kernel launch (bounds GPU/host memory
  for very large batches).
- `PYCGPU_ROBUST=0` — disable robust phase-removal (default on for the
  accelerated backends).

## Phase diagram mapping

`binplot(..., method='grid')` and `ternplot(..., method='grid')` compute
diagrams from a dense equilibrium grid instead of ZPF-line following —
brute force the accelerated backends make cheap. Every multi-phase grid
point contributes its tie-line endpoint compositions, so boundary points
sit at solver accuracy (verified vs reference tie-lines to ~1e-15);
grid resolution only controls boundary sampling density. Measured: Al-Ni
(8 phases, 12,100-point grid) in 2.9 s on the C++ backend vs 27.1 s for
reference ZPF mapping; Al-Zn in 1.3 s.
