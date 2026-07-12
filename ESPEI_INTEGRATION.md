# ESPEI on the pycalphad accelerated backends — integration report

ESPEI's MCMC evaluates a *fixed* set of small equilibria and samples every
step — identical conditions, only the parameter vector changes. The
integration exploits that: build the machinery once, poke parameters in
place each step, and evaluate whole batches (all data points of a
residual; all walkers of a proposal) in single launches. The ESPEI-side
code lives on the `Zhyrek/ESPEI` `accelerated-backends` branch; pycalphad
exposes only generic hooks (`pycalphad.gpu.point_solver`,
`pycalphad.gpu.ensemble`, `pycalphad.gpu.gpu_calculate`).

## Measured baseline that motivated the design

ESPEI 0.9.1, Cu-Mg, 2 fit parameters, 7 ZPF datasets: one walker's ZPF
likelihood costs **1.63 s** — ~65% in 714 tiny per-vertex Workspace
equilibria (~2.2 ms each), ~30% in 476 single-phase sampling calls, ~5%
glue. `set_backend` alone does not help (ESPEI's shadow functions bypass
the dispatch, and the per-vertex solves are per-call-overhead-bound);
batching is the win.

## What was implemented

**pycalphad side**
- Runtime fit parameters in the generated kernels: parameter symbols map
  to trailing dof slots read live from `PhaseRecordFactory.param_values`,
  so one compiled kernel serves every MCMC step (validated ~8e-15 vs the
  reference LLVM callables, zero recompilation). Both the grid-evaluation
  and equilibrium-solver paths carry them.
- Point-list batch solver + compiled lower convex hull (see
  POINT_SOLVER_DESIGN.md).
- Walker-ensemble machinery (`ensemble.EnsemblePointBatcher`):
  walker-stacked grids, device-resident energy buffers, external-pointer
  grid blocks — the substrate for whole-ensemble evaluation.
- Accelerated `calculate()` property outputs (GM, HM, SM, CPM,
  `_MIX`/`_FORM`) with fit parameters, used by the thermochemical
  residual.

**ESPEI side (three batched residuals + the sampler hook, all opt-in)**
- `BatchedZPFResidual`: all tie-line vertex equilibria of a likelihood in
  a few batched launches; binary and ternary+ vertices (grouped by
  prescribed-composition count); subsystem datasets and other exact-path
  exceptions fall back to ESPEI's own code paths; `sample_df='reference'`
  reproduces the stock log-likelihood exactly.
- `BatchedFixedConfigurationPropertyResidual`: per-sample dof rows built
  once, each data group's property evaluated in one compiled call
  (reference-state shifts and excluded contributions included).
- `BatchedActivityResidual`: per-dataset subsystem machinery built once,
  all sample equilibria in one batched launch; reference states and
  boundary compositions stay on the stock path.
- `EnsembleLikelihoodPool`: emcee proposal half-batches evaluate through
  ensemble-capable residuals in one launch set (mirrors
  `EmceeOptimizer.predict` exactly); installed automatically when an
  accelerated backend is active and no dask scheduler is configured.
- One YAML line opts in: `mcmc.backend: c++` (or `gpu`); default behavior
  is untouched, and all residuals are pickle-safe for dask workers.

## Measured outcomes (Cu-Mg unless noted; one CPU core / laptop RTX 4070)

| workload | stock | accelerated |
|---|---|---|
| ZPF likelihood (476 driving forces) | 1.94 s | 0.13–0.16 s (~12.6x) |
| real emcee run, 16 walkers × 30 steps | 837 s | 20.7 s (40x), matching acceptance |
| whole-ensemble step, 8–1024 walkers (CUDA) | — | ~10 ms/walker/step, flat (165–200x) |
| fixed-config thermochemical likelihood | 27 ms | 0.24 ms (~110x), likelihoods identical |
| activity likelihood (10-point fixture) | 241 ms | 200 ms (~1.2x; scales with points/dataset) |
| demo fit, 4 walkers × 15 iterations | 97 s | 3.8 s (ensemble pool), trace bit-identical to per-walker |

Fidelity: batched ZPF differs from stock by ~0.08 out of −3934 total logL
(sampling-estimate vertices on the shared grid; exact in
reference-sampling mode); thermochemical and activity likelihoods are
identical to stock on all fixtures; the ESPEI test suite passes both with
and without the accelerated pycalphad installed.
