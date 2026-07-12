# [DRAFT — do not submit yet] Accelerated equilibrium backends (C++ / CUDA)

**Target:** `pycalphad/pycalphad` `develop` ← `Zhyrek/pycgpu` `gpu-port-sync` (develop-synced; a squash-curated branch will replace this before un-drafting)

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
Anything the backends cannot handle exactly (custom `Model` subclasses,
non-standard or under/overdetermined conditions, dilute compositions) is
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
- Same-phase-set energy agreement vs the current reference solver:
  21-phase AlCuFe over a 245-condition grid — where both solvers converge
  (236 conditions), **236/236 stable phase sets match with max |dGM| =
  3.5e-5 J/mol**; AuBi 55/55 at <= 9e-6. Convergence census on the
  remaining 9 AlCuFe conditions: 2 fail on both solvers, 3 fail on the
  reference only (the backend converges), 4 fail on the backend only
  (reference converges) — i.e. roughly symmetric ~2% failure rates on
  this deliberately hard system, in the known degenerate add/remove
  cycling class (not iteration-cap limited; flagged as a follow-up). (Against the
  older solver this branch originally targeted there was a small
  degenerate-basin mismatch family; upstream's solver improvements since
  then eliminated it.) Two test adjustments are flagged inline (a bitwise
  comparison relaxed to rtol=1e-10; one degenerate-tie test made
  backend-aware, 0.02 J/mol).
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

- [ ] Squash/curate commit history
- [ ] Remove/relocate development docs (POINT_SOLVER_DESIGN.md,
      ESPEI_INTEGRATION.md, PR_DRAFT.md) as maintainers prefer
- [ ] CI story for the backends (compiler availability on runners)
- [ ] Maintainer decision on robust-phase-removal defaults
