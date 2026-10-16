# pycalphad-cg

A column-generation equilibrium solver for [pycalphad](https://github.com/pycalphad/pycalphad), packaged as a
`Workspace` subclass. It needs no changes to pycalphad.

This is a working example. It shows how an external solver can plug into pycalphad today, and which pycalphad
internals it has to rely on to do so. That second list is meant as input for a stable solver API.

## Use

```python
from pycalphad import Database, variables as v
from pycalphad_cg import CGWorkspace, equilibrium

dbf = Database('alfe.tdb')
conds = {v.T: (500, 2000, 10), v.X('AL'): (0.01, 0.99, 0.01), v.P: 101325, v.N: 1}

# same arguments and result as pycalphad.equilibrium
eq = equilibrium(dbf, ['AL', 'FE', 'VA'], list(dbf.phases), conds, output='HM')

# or as a Workspace: everything after the equilibrium (get, properties, derivatives) is pycalphad's
wks = CGWorkspace(dbf, ['AL', 'FE', 'VA'], list(dbf.phases), conds)
hm, cp = wks.get('HM', 'HM.T')
```

`CGWorkspace` takes two extra fields:

- `cg_threads`: worker threads for the conditions. The default is the `PYCALPHAD_CG_THREADS` environment variable,
  or all cores if that is not set.
- `cg_fallback`: what happens with problems column generation does not handle (listed under "Limits"). If True (the
  default), they are solved by pycalphad's own `Workspace.recompute`, with a warning. If False, they raise
  `CGCapabilityError`.

## Install

The compiled core includes pycalphad's `phase_rec.pxd`, so it is built against the installed pycalphad:

```bash
pip install --no-build-isolation .
```

For a pycalphad source checkout instead of an installed one:

```bash
PYTHONPATH=/path/to/pycalphad python setup.py build_ext --inplace
```

## How it plugs in

pycalphad's `Workspace.recompute()` returns the equilibrium dataset (`LightDataset`). It does this by sampling
every phase on a grid, taking the lower convex hull of the samples as starting points, and refining them with a
solver object. `CGWorkspace` overrides `recompute()` only.

Everything that reads `wks.eq` works on its result unchanged: `wks.get(...)`, computed properties, Jansson
derivatives, reference states, and the mapping strategies' starting points. `recompute()` itself:

1. runs the input checks pycalphad makes in `calculate` and `starting_point`, which column generation bypasses, so
   the same errors are raised (`ConditionError`; a `ValueError` about the degrees of freedom);
2. falls back to `Workspace.recompute()` for unsupported problems;
3. otherwise solves every condition by column generation and writes the dataset in pycalphad's layout.

How the solver works:

- **Linear program.** The equilibrium is a linear program over a set of points, each a phase at given site
  fractions.
- **Pricing.** Each round looks for the point of every phase with the lowest energy relative to the current
  chemical potentials, by local minimization from several starts, and adds it.
- **Polish.** When no phase has a point below the tangent plane, Newton refines the solution on the equations
  that define the equilibrium.
- **Certification.** One more pricing round confirms that no phase lies below the final tangent plane.
- **Grid.** No energy grid is computed. Conditions that share state variables are solved in sequence, each
  starting from the previous one's points, on worker threads that release the GIL.

## Validation

**pycalphad's own test suite with `CGWorkspace` in place of `Workspace`.** `tests/upstream_suite_plugin.py` swaps
the class in for the whole suite. On upstream develop (6f9e2ce6):

```bash
cd /path/to/pycalphad
PYTHONPATH=/path/to/pycalphad:/path/to/pycalphad-cg python -m pytest pycalphad/tests -p tests.upstream_suite_plugin
```

| | passed | skipped | xfailed | xpassed |
|---|---|---|---|---|
| pycalphad's `Workspace` | 335 | 2 | 1 | 1 |
| `CGWorkspace` | 335 | 2 | 1 | 1 |

Four equilibrium calls fell back to pycalphad's solver. All four are tests that pass their own solver object.

**This package's tests** (`python -m pytest tests`) compare against pycalphad on binaries and ternaries from
pycalphad's test databases:

- column generation's Gibbs energy is never above pycalphad's by more than 0.01 J/mol, and where the two agree
  the chemical potentials do too;
- `wks.get` properties and the `equilibrium` wrapper's output match;
- input errors match;
- fallback works, and 1 or 4 threads give the same result.

## Performance

`examples/timing.py`, on an Apple M4 (10 cores) against pycalphad develop (6f9e2ce6). Times are wall time per
call, warm, including building the Workspace (its models and phase records): about 0.35 s of each call for the
COST507 systems, the same for both solvers.

| System | Conditions | pycalphad | cg, 1 thread | cg, 10 threads | cg lower by, at most (J/mol) |
|---|---|---|---|---|---|
| Al-Ni (alni_dupin_2001), 1200 K | 500 | 0.27 s | 0.09 s | 0.06 s | 7e-5 |
| Al-Fe (Al-Fe_sundman2009), 900 K | 500 | 1.47 s | 0.40 s | 0.18 s | 39 |
| Al-Fe (alfe), 500–1900 K | 15 × 25 | 0.20 s | 0.06 s | 0.05 s | 22 |
| Al-Cr-Ni (alcrni), 1273 K | 25 × 25 | 0.57 s | 0.17 s | 0.09 s | 152 |
| COST507 Al-Mg-Si-Cu-Zn, 800 K | 100 | 1.60 s | 1.06 s | 1.06 s | 4e-6 |
| COST507, 10 elements, 800 K | 100 | 7.93 s | 2.91 s | 3.00 s | 5e-4 |

- **Energies.** Column generation's Gibbs energy is never above pycalphad's in these runs.
- **Where it is lower.** pycalphad's default grid (pdens 60) misses an ordered state there. At the largest
  difference in each system, pycalphad at pdens 3000 gives the same energy as column generation:
  - Al-Cr-Ni: L12 + L12 at x(Al) = 0.14, x(Cr) = 0.18;
  - Al-Fe: B2 at 700 K, x(Al) = 0.14;
  - Sundman's Al-Fe: BCC at x(Al) = 0.245.
- **Threads.** They gave no gain on the 100-condition COST507 runs. The fork's own measurements show a gain at
  500 conditions (0.34 → 0.14 ms per condition for the 5-element case).

## What it relies on in pycalphad

Nothing here is public API, and each item can break with a pycalphad release. The list is meant as a starting
point for what a solver protocol would need to expose:

- `Workspace.recompute()` as the override point, and `Workspace`'s `TypedField` for the extra settings.
- The `LightDataset` layout of `wks.eq`:
  - variables GM, MU, NP, X, Y, Phase;
  - `vertex` length C+1;
  - Y padded to the largest phase;
  - empty conditions (mole fractions summing above 1) left NaN.
- `PhaseRecordFactory`: `nonvacant_elements`, `state_variables`, `param_values`, `molar_masses`, `models`,
  `comps`, `param_symbols`, `update_parameters`, and `prf[name]`.
- `PhaseRecord`:
  - `variables`, `phase_dof`, `num_statevars`, `parameters`;
  - `num_internal_cons`, `internal_cons_func`, `internal_cons_jac`;
  - `function_factory`.
- `FastFunctionFactory` and `FastFunction`: the compiled function pointers for G, its gradient and Hessian, and the
  per-element mole formulas. These come through `cimport pycalphad.core.phase_rec`, the only C-level dependency.
- The input checks in `calculate` and `starting_point`, re-implemented here because column generation does not
  call them.
- `Solver`, compared by type, to decide whether a passed solver object is a custom one (custom ones fall back).

## Limits

- **Fallback.** These problems are passed to pycalphad's own solver:
  - `calc_opts` other than `pdens`;
  - custom solver objects (pycalphad's grid-and-hull starting points would be skipped, so the solver would never
    see them);
  - custom phase-record factories;
  - a phase without a temperature state variable, or with non-linear internal constraints;
  - conditions other than T, P, N, mole and mass fractions, chemical potentials, and linear combinations of mole
    fractions;
  - phase-local conditions other than site fractions and mole fractions, or a phase-local mole fraction in a phase
    whose formula moles depend on its site fractions (ionic liquids).
- **Mapping.** The ZPF stepping in `pycalphad.mapping` creates `pycalphad.core.solver.Solver` directly, so only the
  mapping's starting equilibria come from column generation. Redirecting the stepping would need a patch.
- **`equilibrium` wrapper.** `pycalphad_cg.equilibrium` is a copy of `pycalphad.equilibrium` with `CGWorkspace`
  in place of `Workspace`. A `workspace_class` argument (or a protocol) upstream would remove the copy.
