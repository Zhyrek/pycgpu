# Point-list batch solver — implementation report

`pycalphad/gpu/point_solver.py` solves many independent fixed-condition
equilibria in one launch, replacing one-Workspace-per-point loops in
consumers like ESPEI's batched residuals. This document records what was
built and how it is validated; the public surface is `PointList`,
`point_hull` / `device_point_hull`, `PointBatchSolver` (via
`get_point_solver`), and `build_spec_row0` / `build_spec_rows`.

Motivation (measured before building it): Cartesian condition grids cannot
serve scattered condition lists — the Cu-Mg ZPF workload's 476 vertices
inflate to 42,770 grid solves (10.7 s) against 1.94 s for the stock
per-point reference, because dense unique temperatures fill the grid with
unwanted points. A true point list is required.

## What it does

- **PointList**: per-point conditions (T, P, N, any number of prescribed
  mole fractions with a component mask, optional per-point phase
  restriction, optional fixed chemical potentials, per-point fit-parameter
  rows). Batches must be uniform in the *number* of prescribed mole
  fractions; `build_spec_rows` enforces this and rewrites the per-point
  constraint blocks (starting chemical potentials from the hull, mole
  fraction rhs/coefficients, fit-parameter tails) onto a tiled spec
  template built by `build_spec_row0` — one template per constraint count.
- **point_hull / device_point_hull**: starting points from a shared
  `calculate()` grid via the compiled lower-convex-hull
  (`gpu/hyperplane.h`), a port of `hyperplane.pyx` verified bit-identical
  (GM, chemical potentials, simplex fractions and indices) on ~2,200 mixed
  binary/ternary/phase-restricted/fixed-chempot cases.
- **PointBatchSolver.solve**: one batched kernel/driver launch over all
  points; returns flat per-point results (GM, MU, NP, X, Y, phase ids,
  converged flag). Work-array memory is bounded by chunked launches
  (`PYCGPU_POINT_CHUNK`); the C++ driver and CUDA kernel share the same
  generated source.

## Fidelity

Each point reproduces the reference `equilibrium()` result for the same
conditions: the solver kernels match the reference minimizer's iteration
dynamics (step-size ramp, convergence gates, phase-change rules), and
end-to-end agreement on validated systems is bit-identical or eps-class
with identical stable-phase sets. Failed points report `converged=False`
so consumers can reproduce reference NaN semantics or fall back
per-point.

## Steady-state cost per consumer call (measured, Cu-Mg ZPF workload)

One `calculate()` grid over the unique temperatures, one hull launch, and
one solver launch per (phase-restriction group × constraint-count group).
Net: ~12.6x per ESPEI ZPF likelihood on one CPU core vs the stock
per-vertex Workspace loop, and ~10 ms/walker/step flat to 1000+ walkers on
CUDA for whole-ensemble evaluation.
