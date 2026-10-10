# Compiled backends

pycalphad can solve equilibria with compiled versions of its solver. Pick a
backend once; every `equilibrium()`, `calculate()` and `Workspace` call
after that uses it, and your scripts stay unchanged:

```python
import pycalphad
pycalphad.set_backend('auto')        # the fastest backend this machine has
eq = pycalphad.equilibrium(dbf, comps, phases, conds)
```

Problems a compiled backend cannot solve exactly (custom `Model`
subclasses, unsupported condition types, vectorized parameter sweeps,
custom solvers, ...) run on the reference implementation automatically, so
setting a backend globally is always safe.

## Backends

| name | runs on | needs |
|---|---|---|
| `'default'` (aliases `'cpu'`, `'reference'`) | the reference implementation (Cython + LAPACK) | nothing |
| `'cpp'` (aliases `'c++'`, `'openmp'`) | the generated solver compiled for the host, on all CPU cores | a C++17 compiler: Xcode command line tools on macOS, `g++` on Linux, MinGW-w64 or WSL on Windows |
| `'cuda'` | NVIDIA GPUs (ROCm experimentally) through CuPy | `pip install pycalphad[gpu]`, an NVIDIA driver |
| `'metal'` | Apple-silicon GPUs together with the CPU | macOS on Apple silicon, Xcode command line tools |
| `'gpu'` | `'cuda'` if CuPy finds a device, else `'metal'` | |
| `'auto'` | `'cuda'` if usable, else `'cpp'`, else `'default'` | |

`pycalphad.available_backends()` reports which backends work on the current
machine and, for those that do not, why.

Selection can also be scoped or per call:

```python
with pycalphad.backend('cpp', threads=4):     # context-local (thread-safe)
    eq = pycalphad.equilibrium(...)

eq = pycalphad.equilibrium(..., backend='metal')
```

The `PYCALPHAD_BACKEND` environment variable sets the initial backend.

### Options

Pass options to `set_backend` / `backend`:

| option | default | meaning |
|---|---|---|
| `threads` | all available cores | C++ solver and hull threads. Use `threads=1` if you already run pycalphad in parallel processes (e.g. ESPEI walkers). Results are bit-identical for any thread count. |
| `max_iters` | 1000 | Newton-iteration budget per condition |
| `chunk` | automatic | conditions per launch (bounds memory for huge batches) |
| `pass1_iters` | backend-specific | iteration cap of the first solving pass (CUDA: 50; Metal: 24); conditions that hit it are re-solved at the full budget |
| `robust_phase_removal` | True | terminate add/remove cycles between near-duplicate composition sets |
| `hull_procs`, `calc_procs` | 1 | worker processes for the reference starting-point hull and for grid sampling over temperatures (0 = automatic) |

## How the compiled backends work

The backends generate C++ from the symbolic `Model` energies (energy,
gradient, Hessian, constraints), combine it with a C++ port of pycalphad's
minimizer and a transliteration of the LAPACK routines it uses, and compile
the result once per system with the local toolchain. Compiled modules are
cached on disk (`~/Library/Caches/pycalphad` on macOS, `~/.cache/pycalphad`
on Linux; override with `PYCGPU_CACHE_DIR`), so only the first use of a
system pays the compile time (seconds for C++, ~10 s to a few minutes for
GPU kernels).

All three compiled backends run the same solver source
(`pycalphad/gpu/solve_condition.h` and the headers it includes); they differ
only in how conditions are spread over hardware.

### Metal (Apple silicon)

Apple GPUs have no double-precision units, and the solver needs ~1e-14
relative precision over double's full exponent range (magnetic models carry
intermediates like `T**25`). The Metal build therefore runs the solver in
emulated double precision: every `double` becomes a pair of floats with an
extra exponent field (`pycalphad/gpu/metal_compat.h`), giving ~48
significant bits over double's range. The CUDA/C++ source is lowered to
Metal Shading Language automatically (`metal_backend.lower_to_msl`), so the
GPU runs the same code as the other backends.

One GPU thread is far slower than a CPU core on this serial, branchy code;
the GPU only helps when thousands of similar conditions move in lockstep.
The Metal backend therefore uses the GPU and the CPU together:

1. Batches smaller than a few thousand conditions run on the C++ solver.
   The GPU kernels are compiled in the background when the first larger
   batch for a system arrives (tens of seconds to a few minutes, once per
   system; macOS caches the result), and that batch runs on the CPU.
2. Larger batches are split between the GPU and the CPU threads in
   proportion to their measured throughput, and both run at once.
3. The GPU pass uses a small iteration cap; conditions it leaves
   unconverged or capped are finished by the C++ solver at full budget and
   full double precision. Conditions with a prescribed mole fraction within
   1e-6 of 0 or 1 always run on the CPU: the dilute component's chemical
   potential needs more than ~48 bits.
4. Systems whose per-condition solver state does not fit a GPU thread
   (Metal allows 256 KB) run on the C++ solver entirely. The limit depends
   mostly on the most complex phase: roughly 25 phases when every phase has
   at most 4 site fractions, 14 phases with an 8-site-fraction phase (e.g. a
   four-sublattice ordered model), 9 with 12.

Results agree with the reference at the solver's convergence tolerance (see
below); the conditions finished on the GPU carry the emulated precision
(GM typically within 1e-6 J/mol of the C++ backend, worst cases ~1e-2 J/mol
near degenerate phase boundaries).

**Should you use it?** On the Apple M4 (10 GPU cores) the GPU adds about
10% throughput to the `'cpp'` backend on batches of tens of thousands of
conditions and nothing below that, because emulated double precision costs
~20x and one GPU thread is far slower than a CPU core; `'auto'` therefore
picks `'cpp'`. Chips with larger GPUs (Max/Ultra) should shift the balance
toward the GPU; the Metal backend measures both sides on every call and
adapts its split.

## Measured performance

Apple M4 (4 performance + 6 efficiency cores, 10-core GPU), Al-Ni
(8 phases), temperature-composition grids, kernels already compiled
(`examples/5_Accelerated_Backends/diagnostics/compare_backends.py`):

| conditions | `'default'` | `'cpp'` 1 thread | `'cpp'` 10 threads | `'metal'` |
|---|---|---|---|---|
| 10,000 (100 x 100) | 7.6 s | 1.63 s | 0.36 s | 0.37 s (CPU only: below the GPU threshold) |
| 90,000 (300 x 300) | ~69 s (extrapolated) | — | 3.02 s | 2.72 s |
| 250,000 (500 x 500) | — | — | 8.17 s | 7.37 s |

On these grids all backends report the same failed (non-converged)
conditions. That is not guaranteed in general: where the phase-amount
convergence test sits at the solver's numerical noise floor (e.g. the
two-composition-set FCC_L12 region of Al-Ni), iteration counts vary from
condition to condition for every backend, and an isolated condition can
exceed the 1000-iteration budget on one backend but not another. GM from
`'cpp'` agrees with the reference to 5e-4 J/mol; `'metal'` agrees with
`'cpp'` to 7e-3 J/mol (99th percentile 6e-5 J/mol). The times include the
whole `equilibrium()` call (grid sampling, starting point, result assembly).

Earlier measurements on a laptop RTX 4070 / one CPU core, vs pycalphad
0.11.1 (from the CUDA backend's development):

| workload | reference | cpp (1 thread) | cuda |
|---|---|---|---|
| AlCuFe 245 equilibria | 7.3 s | 1.05 s | 3.7 s |
| AlCuFe 10,571 equilibria | 134 s | ~13 s | ~12 s |
| AuBi 1,002,000 equilibria | 584 s | 62 s | ~55 s |
| ESPEI ZPF likelihood (Cu-Mg, 7 datasets) | 1.94 s | 0.155 s | — |

Consumer NVIDIA cards have low FP64 throughput (1/64 of FP32); data-center
cards (A100/H100) change the CUDA numbers substantially.

## Accuracy

The compiled solvers reproduce the reference minimizer's algorithm
(iteration dynamics, convergence tests, phase addition and removal); final
energies agree with the reference to ~1e-4 J/mol with identical stable
phase sets across the test suite. The remaining differences come from
floating-point evaluation order in the generated energy functions, which
can move a condition to a slightly different point inside the same
convergence tolerance band. Composition conditions at or beyond the dilute
limit are clamped into `[1e-10, 1 - 1e-10]` exactly as in the reference;
at the clamp, GM agrees to ~1e-5 J/mol while the chemical potential of the
clamped component itself (an RT/y-amplified quantity) can differ by
~1e3 J/mol.

## Derivatives (Jansson derivatives)

`Workspace.get('HM.T')`, `Workspace.get(JanssonDerivative('GM', v.X('AL')))`
and `equilibrium(..., output='HM.T')` work on every backend. On a compiled
backend, derivatives of system-level Model properties (`GM`, `HM`, `SM`,
`CPM`, ...) with respect to the temperature or a mole-fraction condition are
computed for the whole grid at once:

* the derivative deltas (Sundman et al. 2015, Eqs. 74/78) are solved inside
  the compiled solver right after each condition converges (C++, CUDA, and
  Metal, where the GPU computes them in emulated double precision);
* the property values and gradients come from generated, compiled property
  functions;
* only the final chain-rule assembly (Eq. 73) runs in Python.

Other derivatives (phase-specific numerators such as `X(FCC_A1,AL).T` or
`NP(BCC_B2).T`, pressure denominators, and problems with MU, W,
linear-combination or phase-local conditions) are evaluated per condition by
the reference code on the backend's converged states.

The upstream reference implementation pairs phase-amount deltas in moles of
formula units with per-mole-of-atoms properties, so derivatives that involve
phase amounts are wrong for phases with more or fewer than one atom per
formula unit, or a composition-dependent number of them (e.g. the Cu-Mg
Laves phase, B2 with vacancies). This fork patches the reference
implementation at import (`pycalphad/_upstream_patches.py`) until the
upstream fix is merged; the compiled backends compute the deltas in moles of
atoms directly.

## Running the test suite on a backend

```shell
pytest pycalphad/tests                     # reference
pytest pycalphad/tests --backend=cpp
pytest pycalphad/tests --backend=metal
pytest pycalphad/tests --backend=cuda
```

The flag sets the global backend for the whole session, so every test runs
through it. Tests that compare bitwise against the reference report xfail
(never skip) when the documented evaluation-order differences trip them.

## Phase diagram mapping

`binplot(..., method='grid')` and `ternplot(..., method='grid')` compute
diagrams from a dense equilibrium grid instead of ZPF-line following, which
the compiled backends make cheap. Every multi-phase grid point contributes
its tie-line endpoints, so boundary points sit at solver accuracy; grid
resolution only controls how densely boundaries are sampled.

The default ZPF-line mapping (`binplot`, `ternplot`, isopleth and step
strategies) follows boundaries one point at a time: its energy sampling
(`calculate()`) uses the compiled evaluator, while its single-point solves
use the reference solver (they are small and sequential; see below).

## What runs on a compiled backend

| path | compiled | notes |
|---|---|---|
| `equilibrium()`, `Workspace` solves | yes | plain `Model` and MQMQA (quasichemical) models; custom `Model` subclasses and custom `Solver` objects use the reference |
| `calculate()` | yes | energies and properties at sampled points, multi-threaded |
| `Workspace.get` / `equilibrium(output=...)`: X, NP, Y, MU, T/P/N, Model properties | yes | whole-grid evaluation, bit-identical to the per-condition path (grids of 50+ conditions) |
| Jansson derivatives of Model properties w.r.t. T or X | yes | see above |
| ZPF mapping (`binplot`, `ternplot`, isopleths, step) | partly | sampling compiled, point solves on the reference |
| `IsolatedPhase`, `DormantPhase`, `T0`, other derivatives | no | per-condition reference solves (warm-started from the previous condition) |

## Troubleshooting and developer settings

* `PYCGPU_TIME=1` prints where accelerated time goes (kernel wall time,
  GPU/CPU split for Metal).
* `PYCGPU_COUNT_DISPATCH=<file>` logs every routing decision (compiled
  backend or reference fallback, and why).
* `PYCGPU_FALLBACK=1` falls back to the reference implementation on any
  compiled-backend error instead of raising.
* `PYCGPU_CACHE_DIR` moves the kernel cache; deleting it forces recompiles.
* `tools/metal_debug.py` builds a generated Metal module as host C++ (with
  working `printf`) for debugging the emulated-precision kernels on the CPU.

Developer notes on individual components live in `docs/backends/`.
