# Accelerated pycalphad backends + ESPEI integration — change log for standup

Two branches, both CI-green on every platform, both ready as draft PRs:
`Zhyrek/pycgpu` (`accelerated-backends-pr`, curated from `gpu-port-sync`)
and `Zhyrek/ESPEI` (`accelerated-backends`). Everything below is measured,
not projected. Reference hardware: one laptop CPU core / RTX 4070.

## The one-paragraph version

pycalphad gained two optional, runtime-selectable compute backends —
`set_backend('c++')` and `set_backend('gpu')` — that generate C/CUDA code
from the symbolic thermodynamic models at runtime, compile once, disk-cache
the kernels, and reproduce the reference solver's results at
numerical-precision-class agreement (max |dGM| 8.8e-6 J/mol on the
37-phase AlCuFe census, with the backend converging on conditions the
reference fails) while running ~10x faster on one core and scaling to
million-condition batches. On top of them, ESPEI's MCMC parameter fitting
got batched likelihoods for three of its four residual types and
whole-ensemble walker evaluation: ~40x on a real emcee run, ~110x per
thermochemical likelihood, ~10 ms/walker/step flat to 1000+ chains on GPU.
Opt-in is one YAML line (`mcmc.backend: c++`); default behavior is
untouched, and anything the backends can't handle exactly falls back
silently to the reference path.

## Phase 1 — the engine (pre-handoff)

- **Runtime code generation**: symbolic `Model` energies (symengine) →
  C/CUDA device functions (objective, gradients, Hessians, mass/mole
  functions, internal constraints) with CSE, piecewise→ternary lowering,
  and strength-reduced integer powers. Compiled via CuPy RawModule (GPU)
  or the system C++ compiler (single-threaded C++ driver), disk-cached per
  (system, model fingerprint).
- **Solver port**: the full equilibrium minimizer (composition sets,
  equilibrium matrix assembly, SVD/LU linear algebra, phase add/remove
  logic, starting-point construction) as per-thread device code — one
  equilibrium condition per GPU thread.
- **Bit-parity discipline**: line-by-line trajectory comparison against
  the Cython reference; fixed a GM/NP scale bug, the formula-unit
  phase-amount convention, a VA sublattice-ordering bug class, and
  documented pycalphad's own cross-process nondeterminism
  (PYTHONHASHSEED, 1-ulp jitter — PYCALPHAD_NONDETERMINISM.md).
- **Full stock test suite passing under the C++ backend** (290/290 at the
  time; 318 post-merge), plus grid-method phase diagram plotting and
  pipeline parallelization.
- **Runtime fit parameters**: parameter symbols map to trailing
  degrees-of-freedom slots read live from the phase records, so one
  compiled kernel serves every MCMC step without recompilation (~8e-15
  agreement vs the reference callables).

## Phase 2 — productization + ESPEI (first stretch of this session)

- **ESPEI integration analysis**: profiled a real ZPF likelihood (1.63 s =
  714 tiny per-vertex equilibria + 476 sampling calls) and designed the
  batching tiers; fixed a C++ driver-loop bug found on the way.
- **Point-list batch solver** (`pycalphad.gpu.point_solver`): arbitrary
  lists of fixed-condition equilibria in one launch, with per-point phase
  restriction and per-point parameter vectors.
- **Device lower convex hull** (`gpu/hyperplane.h`): compiled port of
  `hyperplane.pyx`, verified **bit-identical** on ~2,200 cases including
  fixed-chemical-potential sets.
- **Walker-ensemble machinery** (`pycalphad.gpu.ensemble`): walker-stacked
  grids, device-resident energy buffers, external-pointer grid blocks —
  ~10 ms/walker/step flat from 8 to 1024 walkers on CUDA (165–200x).
- **Batched ZPF residual in ESPEI**: ~12.6x per likelihood; real emcee run
  16 walkers × 30 steps: 837 s → 20.7 s (40x) with matching acceptance.
- **Dead-code parity study**: proved the historical outer phase-addition
  loop never executed (read NaN padding as zero), compiled it out with a
  study flag.
- **Develop-branch sync**: merged upstream pycalphad develop; purged ~800
  lines of debug residue so reference files are byte-for-byte upstream;
  supported the new never-disorder model feature on both backends (found
  and fixed a general vertex-slot compaction defect in the process).
- **MU (chemical potential) conditions** on the accelerated path, hull
  bit-identical on 287 fixed-MU cases.
- **CI on all platforms**: macOS (glibc-only guards), Windows (static
  MinGW linking + DLL search fallback), ESPEI CI against stock pycalphad
  (optional imports). 9/9 pycalphad platform jobs, 7/7 ESPEI jobs.
- **Packaging**: `pip install pycalphad[cpp] / pycalphad[gpu]`,
  `espei[cpp] / espei[gpu]`; YAML opt-in `mcmc.backend: default|c++|gpu`.
- **Demo package** for a side-by-side laptop demonstration: stock ~97 s vs
  accelerated ~9 s (later 3.8 s), plus a 16-core dask variant — which
  caught and fixed a real dask-pickling bug (ctypes handles can't cross
  process boundaries; residuals now rebuild per worker).
- **Honesty audits** for the collaborator brief: dispatch audit (which
  equilibrium calls actually accelerate), AlCuFe convergence census,
  measured speedup decompositions.

## Phase 3 — coverage, correctness tail, and completion (this session)

- **`calculate()` property outputs beyond GM**: HM, SM, CPM and the
  `_MIX`/`_FORM` variants (any symbolic model property) compile into
  lightweight property modules — ~2x the reference callables on HM/CPM,
  eps-class agreement (≤3e-10) across 4 databases × 8 outputs on both
  backends, including NaN-semantics parity for partitioned models.
- **Batched thermochemical residual in ESPEI**: the stock residual builds
  one Workspace *per data sample per MCMC step*; the batched version
  precomputes sample rows once and evaluates each data group in one
  compiled call. **~110x per likelihood, likelihoods identical to stock.**
- **Correctness tail — every silent fallback investigated and fixed.**
  Removing the masks exposed six real defects the fallbacks had hidden:
  charged-species symbol mapping (FE+3 vs FE_POS3), 9-component
  starting-point extraction, a numpy dtype form that disabled the
  metastable-addition grid for single-phase systems, composition-set
  capacity not scaling with component count, and — the big one — solver
  dynamics drift from the upstream merge (the reference ramps its Newton
  step and requires 10 quiet iterations before convergence; the backend
  had a constant step and a 5-iteration gate). After matching, backend
  trajectories track the reference to ~1e-12 per iteration; the
  fallback-guarded AlCuFe census gives 240/240 converged at max |dGM|
  8.8e-6 with the backend also converging on all 5 reference failures. Tests that had never truly run
  on the backend (rose_nine, issue589's 3-way miscibility gap, gh-503
  pure-vacancy suspension, ill-conditioned magnetic Hessian, charged
  alfeo) all pass on both backends now.
- **Batched activity residual**: per-dataset subsystem machinery built
  once, all sample equilibria in one launch; residuals identical to stock
  with and without fit parameters (speedup scales with dataset size).
- **Ternary+ ZPF vertices**: multi-composition vertices batch (grouped by
  constraint count); exact ternary tie-line vertices match stock to 4
  decimals at ~5.8x; also fixed a latent subsystem-dataset bug (binary
  data inside a ternary fit now solves in its own region system, matching
  stock to 3e-9).
- **emcee ensemble hook**: `EnsembleLikelihoodPool` evaluates whole
  proposal half-batches through the batched residuals — installed
  automatically from a stock YAML config; same-seed demo trace
  bit-identical to per-walker evaluation, fit time 8.7 s → 3.8 s at just
  4 walkers.
- **Curated 3-commit PR branch** cut clean from upstream develop
  (engine / dispatch + tests / packaging + docs + plotting), triple test
  gate green: default 319 / c++ 318 / gpu 318.
- **Cleanup**: planning documents converted to implementation reports,
  scratch files removed, ~180 alarm-style comment markers and leftover
  generation artifacts rewritten as concise descriptive comments.

## Validation posture (how we know it's right)

- Reference solver untouched: reference tree byte-for-byte upstream except
  a ~40-line capability gate and one evaluator hook; the reference remains
  the ground truth.
- Full stock test suite under all three backends on every change
  (default 319 / c++ 318 / gpu 318; the delta is one backend-aware skip).
- Component-level bit-parity where meaningful (hull bit-identical;
  per-iteration solver trajectory traces); censuses guarded against
  silent fallbacks so parity claims are never vacuous.
- ESPEI: 289 tests passing with and without the accelerated pycalphad;
  every batched residual A/B'd against its stock counterpart (identical or
  documented eps-class); same-seed MCMC traces compared bit-wise.

## Headline numbers

| workload | stock | accelerated |
|---|---|---|
| pycalphad equilibrium, AlCuFe 10,571 conditions | 134 s | ~13 s (C++), ~12 s (GPU) |
| pycalphad equilibrium, AuBi 1,002,000 conditions | 584 s | 62 s (C++), ~55 s (GPU) |
| stock pycalphad test suite runtime | 1x | ~3.7x faster under C++ |
| ESPEI ZPF likelihood | 1.94 s | 0.13–0.16 s |
| ESPEI thermochemical likelihood | 27 ms | 0.24 ms |
| real emcee run (16 walkers × 30 steps) | 837 s | 20.7 s |
| GPU walker ensembles (8–1024 chains) | — | ~10 ms/walker/step, flat |
| end-to-end demo fit (4 walkers × 15 iter) | 97 s | 3.8 s |
