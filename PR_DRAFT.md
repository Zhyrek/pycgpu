# [DRAFT — do not submit yet] Accelerated equilibrium backends (C++ / CUDA)

**Target:** `pycalphad/pycalphad` `develop` ← `Zhyrek/pycgpu` `accelerated-backends-pr` (curated 3-commit branch: engine / dispatch+tests / packaging+docs+plotting; `gpu-port-sync` is the full development history)

## Summary

This PR adds two optional, runtime-selectable backends for `equilibrium()`
and `calculate()` that reproduce the reference solver's results at
numerical-precision-class agreement while running 5–10x faster on one CPU
core and scaling to million-condition batches:

```python
import pycalphad
pycalphad.set_backend('c++')     # or 'gpu'; also `with pycalphad.backend(...)`
# ...every existing script now runs accelerated, unchanged...
```

No pycalphad rebuild is involved: the backends generate C/CUDA source from
the symbolic `Model` energies at runtime, compile once per (system, model
fingerprint) with the user's toolchain, and disk-cache the artifacts.
Supported conditions cover the standard shapes: N=1/P/T/X grids, W
mass-fraction and linear-combination conditions (difference and ratio
forms; eps-class GM with matching phase sets, and the backend converges
on the W-grid conditions the reference fails), and MU chemical-potential
conditions (scalars and arrays), plus dilute/boundary compositions with
the reference's clamping semantics.
Anything the backends cannot handle exactly (custom `Model` subclasses,
phase-local or under/overdetermined conditions, dilute compositions) is
routed to the reference code path automatically, including its validation
errors — `set_backend` is safe to enable globally.

## Correctness

- **Synced to current `develop` and the full stock test suite passes under
  both backends** (default 319 passed / c++ 318 / gpu 318, the difference
  being one backend-aware skip), including the tests added upstream since
  this work began — among them the new never-disorder model feature (#651),
  which the backends support (generated energies match the reference to
  1e-15 on CoV-20Wan.tdb; its test also exposed and fixed a general
  vertex-slot compaction defect in result processing).
- The starting-point hull is a device/compiled port of `hyperplane.pyx`
  verified **bit-identical** (GM, chemical potentials, simplex fractions
  and indices) on ~1,900 mixed binary/ternary/phase-restricted cases, and
  end-to-end GM is bit-identical to the CPU-hull path at 10k, 100k, and
  1,002,000-condition batches.
- Energy agreement vs the current reference solver: the 37-phase AlCuFe
  census at 245 conditions gives **240/240 mutually converged conditions
  with max |dGM| = 8.8e-6 J/mol and matching stable phase sets; the
  backend additionally converges on all 5 conditions the reference fails**
  (composition-sum corner cases), with zero backend-only failures. The
  census is guarded against silent fallbacks (a dispatch audit asserts
  the backend actually ran — an earlier undefined-database-symbol
  compile failure had made interim census results vacuous, caught and
  fixed via reference-matching undefined-symbol semantics in codegen).
  Solver kernels match the reference's iteration dynamics (step-size
  ramp, convergence gates, NaN vertex padding), verified by
  per-iteration trajectory traces agreeing to ~1e-12 on the sensitive
  cases (5-component single-phase miscibility gap gh-589, ill-conditioned
  magnetic Hessian, 9-component rose, pure-vacancy suspension gh-503,
  charged-species alfeo). (Against the older solver this branch
  originally targeted there was a small degenerate-basin mismatch family;
  upstream's solver improvements since then eliminated it.) Two test
  adjustments are flagged inline (a bitwise comparison relaxed to
  rtol=1e-10; one degenerate-tie test made backend-aware, 0.02 J/mol).
- The reference tree is byte-for-byte upstream: workspace.py, solver.py,
  minimizer.pyx/.pxd, eqsolver.pyx, lower_convex_hull.py and
  starting_point.py are unmodified. The only reference-file changes are the
  capability-gated dispatch in core/equilibrium.py and an evaluator hook in
  core/calculate.py. The `robust_phase_removal` kwarg applies to the
  accelerated kernels only (accepted and ignored on the reference path).
- Determinism: identical inputs give bit-identical outputs run-to-run on
  both backends (`PYTHONHASHSEED` caveats of the reference `calculate()`
  sampling are documented separately and predate this PR).

## Performance (laptop RTX 4070 / one CPU core, vs upstream 0.11.1)

Caveat: reference timings depend on the BLAS stack. These were measured
in a conda/MKL environment; a pip/OpenBLAS install runs the *reference*
solver ~2.2x faster on small-matrix-heavy systems (measured on AlCuFe
245: 2.95 s vs 6.5 s), which shrinks the backend speedup accordingly
there (backend: 1.1 s). Large batches are BLAS-insensitive (Al-Ni
12,100 conditions: reference 14.2 s in both environments; c++ 2.9 s,
gpu 1.6 s).

| workload | 0.11.1 | c++ | gpu |
|---|---|---|---|
| AlCuFe 245 equilibria | 7.3 s | 1.05 s (6.9x) | 3.7 s |
| AlCuFe 10,571 | 134 s | ~13 s (9.7x) | ~12 s (10.5x) |
| AuBi 100,250 | 58.6 s | ~10 s | ~9 s |
| AuBi 1,002,000 | 584 s | 62 s (9.4x) | ~55 s |

The stock test suite itself runs ~3.7x faster under the c++ backend.

## What's in the change

- `pycalphad/gpu/` — code generation from `Model` energies (`gpu_codegen`),
  the solver kernels (`eqsolver.h`, `minimizer.h`, `comp_set.h`,
  `phase_rec.h`, linear algebra in `svd.c`/`lu_solver.h`, device hull in
  `hyperplane.h`), the C++ single-threaded driver (`cpu_backend`), a
  point-list batch solver (`point_solver`), and generic walker-ensemble
  machinery (`ensemble.EnsemblePointBatcher`) for evaluating many
  parameter vectors against fixed condition points — the hook the
  companion ESPEI PR builds its batched likelihoods on. No
  ESPEI-specific code lives in pycalphad.
- `pycalphad/backend.py` — `set_backend` / `backend()` / per-call kwarg,
  eager validation, capability gate.
- `calculate()` accelerates any symbolic `Model` property output (GM, HM,
  SM, CPM, `_MIX`/`_FORM` variants), including runtime fit parameters;
  non-GM outputs compile a lightweight property module (~2x the reference
  callables on HM/CPM) and reproduce reference NaN semantics (e.g. `_MIX`
  on partitioned models). Unsupported outputs fall back silently.
- Minimal reference-code touches: an `accelerated=` hook in
  `_compute_phase_values` (grid energy evaluation only) and the `gpu=`
  dispatch in `core/equilibrium.py`. The reference solver itself is
  unmodified and remains the ground truth the backends are validated
  against.
- Packaging: `pip install pycalphad[cpp]` / `pycalphad[gpu]`; kernel
  sources ship as package data. See `BACKENDS.md` for the dependency
  story (c++: system C++17 compiler; gpu: CuPy wheels + nvcc, CUDA 11/ROCm
  notes).

## Known limitations / discussion points

- GPU backend requires `nvcc` at runtime (CuPy `RawModule(backend='nvcc')`);
  a pure-pip story via NVRTC or the `nvidia-cuda-nvcc-cu12` wheel is a
  possible follow-up.
- AMD/HIP: sources are HIP-compatible by construction but untested in CI.
- Windows native needs MinGW-w64 or WSL for the c++ backend.
- The branch history is a development log (~40 commits incl. investigation
  notes); happy to squash/curate into reviewable units before marking
  ready.

## Checklist before un-drafting

- [x] Squash/curate commit history (`accelerated-backends-pr`: 3 commits,
      triple suite green — default 319 / c++ 318 / gpu 318)
- [x] Development docs, dev TDBs, CLAUDE.md files and important_tests/
      excluded from the curated branch
- [ ] CI story for the backends (compiler availability on runners)
- [ ] Maintainer decision on robust-phase-removal defaults
