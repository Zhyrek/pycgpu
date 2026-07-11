# Point-list batch solver — implementation notes

Goal: solve an arbitrary LIST of conditions (T, P, N, X-vector) in ONE kernel
launch, with optional per-point phase restriction and per-point fit-parameter
vectors. This is the enabler for the batched-ZPF ESPEI residual (tier 3) and
walker-batched MCMC ensembles (tier 4). Cartesian batching measured 10.7 s vs
1.94 s reference on Cu-Mg ZPF (42,770 solves for 476 vertices) — dense unique
temperatures make grids unusable for this workload.

## Per-condition input formats (verified against gpu_equilibrium.py, 2026-07-11)

All arrays are row-per-condition, C-contiguous; dims from `dynamic_sizes`
(Cu-Mg: MC=4, MP=6, MSV=4, MDOF=4, MFIX=4, MAX_PARAMS=2).

1. `condition_args` (n, MSV + MC) float64:
   `[statevar values in state_variables order | X value per wks.components
   entry]` (built by `_condition_column`, gpu_equilibrium.py:370).

2. spec rows (n, padded stride; 88 doubles for Cu-Mg): built by
   `_populate_system_specification` -> `create_flat_system_specification`
   (gpu_systemspec_flat.py, CORE layout) -> `apply_safe_padding` (appends
   AFTER core). Per-point overwrites on a tiled row 0 (fast-path pattern,
   gpu_systemspec_array.py:169):
   - `off_mu = 3` — starting chemical potentials (from hull MU per point;
     fixed-MU conditions keep spec0 values);
   - `off_rhs = 3 + MC + MC*MC` — prescribed_mole_fraction_rhs (X per point,
     constraint enumeration order = conditions dict order, nonvacant only);
   - fit_params at `core_len - (MAX_PARAMS+1)` .. — per-point parameter
     vectors + num_params at `core_len - 1` (core_len = 61 for Cu-Mg; the
     tail of the CORE, not of the padded row).

3. `initial_phase_data` struct rows (stride 80 doubles Cu-Mg) packed by
   `_create_initial_phase_data_struct_array` from dict arrays
   (gpu_equilibrium.py:473): phase_indices (int32 -> model idx via
   py_phase_name_to_unique_idx_map; -1 invalid), phase_amounts (clamped to
   MIN_PHASE_FRACTION), site_fractions (MP, MDOF), compositions (MP, MC),
   chemical_potentials (MC), num_phases. Source = hull output per point:
   valid = model_idx>=0 & NP>1e-10, stable-compacted (vectorized fast path
   gpu_equilibrium.py:486-529 is the template).

4. Grid: `calculate()` per unique statevar combo (use `parallel_calculate`
   with T-key); `_prepare_grid_data_for_gpu_from_calculate_result` packs
   self-describing DeviceGrid blocks; `grid_block_indices` int32 per
   condition selects the block. PER-POINT PHASE RESTRICTION: build an extra
   block per (T, allowed-phase-subset) by filtering the calculate result
   rows to that phase before packing; single-phase points index that block —
   the add-search can then only re-add the allowed phase. No kernel change.

5. Hull per point: pycalphad's `lower_convex_hull` is a serial per-condition
   loop calling Cython `hyperplane()` (~25 us/cond) pulling condition values
   from coords — write a NEW point-list twin in gpu/ (do NOT touch CPU code)
   that iterates the point arrays directly: per point, grid slice = its
   (T-block [+ phase filter]), lincomb rows = X conditions (coef 1 at
   component, rhs = X value) + N row (ones, rhs 1), fills MU/NP/Phase/X/Y
   result rows with a single flat point dim. Feed those into (3).

6. Launch: work arrays (23 slots, per-thread rows) + results
   (n, results_per_condition), `results_per_condition = 7 + MC + MP + MP*MDOF
   + MP*MC + MP`; c++ path `run_cpu_backend` / CUDA kernel with identical
   argument list. Results offsets: GM at 0, MU at 1..MC, phase amounts at
   1+MC.., see `_process_gpu_results`.

7. Kernel acquisition: same codegen/cache as `calculate_equilibrium_gpu`
   (`_run_model_codegen` + kernel cache keyed on model GM expressions +
   statevar layout). Factor or replicate the acquisition block; models with
   fit-parameter symbols produce param-slot-aware kernels (str-sorted order
   = `extract_parameters`).

## Per-likelihood loop (tier 3 steady state, after one-time prep)
1. re-evaluate grid energies for the new parameter vector on the CACHED
   sample dof matrix (points are parameter-independent; the c++/CUDA grid
   evaluators already accept runtime params in trailing dof slots);
2. re-run the point hull (vectorizable, ~25 us/point);
3. poke fit_params into cached spec rows (per point = per walker if tier 4);
4. one launch over all points (hyperplane vertices all-phase + isolated
   single-phase, mixed);
5. extract GM/MU flat; assemble driving forces (semantics already
   implemented and validated in gpu/espei_batch.py v1).

## Acceptance
- driving forces vs ESPEI reference on Cu-Mg 7-dataset ZPF: n=476, target
  << 1 sigma (1000 J/mol); v1 cartesian implementation achieved 469/476
  within 1 J/mol (7 outliers <= 17 J/mol, eps-class basin differences).
- timing target: <= 0.5 s/likelihood C++ (vs 1.94 s reference measured
  2026-07-11 on this machine, PYTHONHASHSEED=0).
