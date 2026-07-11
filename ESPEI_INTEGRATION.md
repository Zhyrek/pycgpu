# Accelerating ESPEI with the pycalphad C++/GPU backends

## Measured baseline (ESPEI 0.9.1, Cu-Mg test system, 2 fit parameters, 7 ZPF datasets)

One walker's ZPF likelihood evaluation costs **1.63 s** and decomposes into:

| component | count | share |
|---|---|---|
| tie-line vertex equilibria (`estimate_hyperplane` + `driving_force_to_hyperplane` Workspace solves) | 714 tiny solves (~2.2 ms each) | ~65% |
| single-phase sampling (`calculate_`, pdens=50) | 476 calls | ~30% |
| glue | — | ~5% |

An MCMC run is `n_walkers x n_steps` likelihood calls (walkers =
`n_params x chains_per_parameter`, typically 2-6/parameter). Even this toy
system costs ~3.6 h for 8 walkers x 1000 steps on one core; production
systems (tens of parameters, hundreds of datasets) are cluster-scale.

**The key structural fact:** the workload per likelihood is a *fixed* set of
small equilibria and samples — identical conditions every step — with only
the parameter vector changing. ESPEI already exploits this on the reference
path: `PhaseRecordFactory.param_values[:] = params` updates compiled LLVM
callables in place, no rebuild.

## What has been implemented (this repository)

**Runtime fit parameters in the generated kernels.** Fit-parameter symbols
(e.g. `VV0000`) now map to *trailing dof slots* `x[num_statevars +
phase_dof + j]` instead of being baked in as constants:

- no kernel-signature changes anywhere (parameters ride in the dof vector);
- the same compiled kernel serves every MCMC step (`param_values` read live
  at call time, matching ESPEI's in-place update convention; str-sorted
  parameter order matches `extract_parameters`);
- gradients/Hessians are untouched (differentiation lists never included the
  trailing slots);
- validated on Cu-Mg: three parameter sets, all phases, generated-vs-LLVM
  energies agree to ~8e-15 relative with zero recompilation.

Currently wired into the `calculate()`/grid-evaluation path
(`pycalphad.gpu.gpu_calculate`). The equilibrium-solver path needs the same
trailing slots filled at composition-set init (parameters stored in the
per-condition SystemSpecification) — mechanical, not yet done.

## Integration tiers

### Tier 1 — drop-in today: minor
`pycalphad.set_backend('c++')` does NOT accelerate ESPEI as-is: ESPEI's
`shadow_functions.calculate_` calls `_compute_phase_values` directly,
bypassing the backend hook, and the per-vertex Workspace equilibria are
per-call-overhead-bound anyway. Expect ~1x. (This is why the tiers below
exist.)

### Tier 2 — one-line ESPEI patch: ~1.2-1.4x per walker
Pass an accelerated evaluator into `_compute_phase_values` from
`shadow_functions.calculate_` (built once per residual context from the
factory; parameters update live). Accelerates the ~30% sampling share and
removes LLVM-callable overhead. Cheap, low risk, worth doing but not the
prize.

### Tier 3 — batched ZPF likelihood: ~8-15x per walker (C++)
Replace the 714 sequential Workspace solves with ONE batched backend call
per likelihood: the vertex conditions/starting configurations are fixed per
run (extracted once from `zpf_data`), so build the per-condition input
arrays once, poke the parameter slots, and launch. The batch solver already
handles per-condition heterogeneous starting phases, and single-phase
vertex solves need `grid_data=NULL` (add-phase search disabled), which the
kernel supports. C++ single-core estimate: 714 x ~0.15 ms = ~0.1 s vs
1.63 s.

### Tier 4 — walker-batched GPU ensemble: the "more chains" win
Because parameters are per-condition inputs (trailing dof slots filled from
the per-condition spec), a single GPU launch can evaluate the ENTIRE
ensemble: `n_walkers x n_vertices` threads, each with its walker's
parameter vector. One launch per emcee step replaces `n_walkers` likelihood
calls:

- 100 walkers x 714 vertices = 71k threads — comfortably inside the
  measured batch regime (100k conditions in ~5-10 s wall, and vertex
  solves are far cheaper than full equilibria: warm-started, 1-3 phases,
  no grid search);
- estimated ensemble step cost ~1-3 s vs 100 x 1.63 s serial (or /cores
  under dask) — **effective 50-100x at large ensembles**, and the cost is
  ~flat in walker count until the GPU saturates. That directly enables the
  strategy of many more chains with fewer, larger steps.
- emcee integration point: a `pool`-like mapper that intercepts the whole
  ensemble proposal batch (emcee's `vectorize=True` / custom
  `EnsembleSampler.log_prob_fn` accepting the (n_walkers, n_params)
  matrix), so no emcee fork is needed.

## Status update (2026-07-11)

**Step 1 (equilibrium-path runtime parameters) is COMPLETE and validated**:
the batch NaN bug traced to unfilled trailing fit-param dof slots in the
initial-phase-data energy evaluation (stack-local array; CUDA silently read
zeros, C++ read stale stack). Fixed in gpu_codegen.py; c++/CUDA match the
Python reference to ~2e-7 max |dGM| across parameter sets; full stock suite
green under PYCALPHAD_BACKEND=c++.

**Tier 3 v1 exists** (`pycalphad.gpu.espei_batch.BatchedZPFCalculator`):
reproduces `calculate_zpf_driving_forces` semantics on batched backend
calls. Correctness on Cu-Mg (7 real ZPF datasets, 476 driving forces):
469/476 within 1 J/mol of reference (7 outliers <= 17 J/mol ~ 0.02 sigma).
Performance: 10.7 s warm vs 1.94 s reference — cartesian grouping solves
42,770 conditions for 476 vertices (182 unique T x 235 unique X). Verdict:
**a point-list batch entry is mandatory**; cartesian batching cannot serve
ZPF data shapes (dense unique temperatures).

### Point-list entry design (v2, the real Tier 3)
The pipeline already flattens grids to per-condition rows, so build those
rows directly from a vertex list:

- per-point condition args / spec rows (fit_params live in each spec row's
  core tail -> per-point/per-walker parameter vectors are free);
- per-point phase restriction WITHOUT kernel changes: grid_block_indices is
  already per-condition, so an IsolatedPhase vertex points at a grid block
  filtered to its single phase (grid search can then only re-add that
  phase) — mixed all-phase + single-phase vertices ride ONE launch;
- starting points: run the existing hull machinery per unique-T block and
  extract the needed rows;
- per-likelihood costs after prep caching: grid re-eval on the CACHED
  sample dof matrix (points are parameter-independent; only energies
  change), vectorized hull, param poke into cached spec rows, one launch,
  flat-array extraction (no xarray). Estimated ~0.5-0.7 s/likelihood on
  C++ (~3-4x); Tier 4 walker batching then amortizes launch+glue across
  the ensemble where CUDA wins big.

## Suggested order of work

1. Equilibrium-path runtime parameters (trailing slots filled at
   composition-set init from per-condition spec storage) + capability-gate
   update. Unlocks Tiers 3/4.
2. Tier 3 batched-ZPF residual as an opt-in ESPEI residual class
   (`residual_objs` are pluggable — ESPEI's `ResidualRegistry` means this
   can live outside ESPEI core initially).
3. Tier 4 ensemble mapper on top (same batch machinery, walker-major
   condition layout).
4. Tier 2 one-liner along the way.

## Risks / notes

- Vertex equilibria in ESPEI use `approximate_equilibrium` and property
  framework details; the batched residual must reproduce
  `estimate_hyperplane`'s exact chempot extraction semantics (NaN handling
  for underdetermined vertices).
- Backend eps-class arithmetic shifts likelihoods by ~1e-6-1e-3 in dGM
  units; MCMC is stochastic and driving-force sigma is 1000 J/mol-scale, so
  this is far below the noise floor — but a same-seed comparison run
  (reference vs batched likelihood traces) is the acceptance test.
- Thermochemical/activity residuals are `calculate_`-based and get Tier 2
  treatment; ZPF dominates cost in phase-boundary-rich fits.
