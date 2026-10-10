"""
Compiled evaluation of IsolatedPhase properties over a condition grid.

IsolatedPhase (pycalphad/property_framework/metaproperties.py) solves its
composition set alone at every condition with Solver.solve, which updates
the composition set in place: each solve starts from the result of the
previous one. Workspace.get evaluates conditions in order and, within a
condition, its properties in order, and every evaluation of an
IsolatedPhase property is one solve. This module replays exactly that
sequence on the compiled solver (one condition per call, no grid, so no
phase additions: Solver.solve semantics), feeding each step's result into
the next, and evaluates each property with its own compute_property on the
composition set the step produced. A step that does not converge is redone
with the reference Solver.solve from the same starting state, since the
reference uses its final state either way.
"""
import numpy as np

from pycalphad import variables as v

__all__ = ['isolated_phase_steps']


def _condition_supported(key):
    return key in (v.N, v.P, v.T) or (type(key) is v.MoleFraction and key.phase_name is None)


def _compset_compatible(wks, compset):
    pr = compset.phase_record
    if pr.phase_name not in wks.phases:
        return False
    if pr.phase_record_factory is wks.phase_record_factory:
        return True
    # find_first_compset's fallback builds a single-phase factory around the
    # same Model object
    models = getattr(pr.phase_record_factory, 'models', {})
    return models.get(pr.phase_name) is wks.models[pr.phase_name]


def chain_supported(wks, compset):
    """None if an IsolatedPhase chain for `compset` can run compiled, else why not."""
    if compset is None:
        return 'no composition set'
    if not _compset_compatible(wks, compset):
        return 'composition set from a different phase record factory'
    if any(not _condition_supported(key) for key in wks.conditions):
        return 'unsupported condition types'
    if len(wks.parameters) > 0:
        return 'fit parameters'
    return None


def isolated_phase_steps(wks, compset, step_conditions):
    """Solve `compset` alone at each step's conditions in sequence, each step
    starting from the previous result (IsolatedPhase semantics).

    step_conditions: list of Workspace.get_dict's cur_conds (condition object
    -> value) per step. Returns a list of (site_fractions, NP, chemical_potentials)
    per step; `compset` is left in the final state, as Solver.solve leaves it.
    """
    from pycalphad.core.solver import Solver
    from pycalphad.gpu.cpu_backend import run_cpu_backend
    from pycalphad.gpu.gpu_equilibrium import _create_initial_phase_data_struct_array
    from pycalphad.gpu.point_solver import (PointList, build_condition_args, build_initial_phase_data,
                                            build_spec_rows, get_point_solver)
    from pycalphad.gpu.work_arrays import allocate_work_arrays

    pr = compset.phase_record
    phase_name = pr.phase_name
    nonvacant = list(pr.nonvacant_elements)
    ncomp = len(nonvacant)
    state_variables = [str(sv) for sv in pr.state_variables]
    n = len(step_conditions)
    values = [{str(key): float(value) for key, value in conds.items()} for conds in step_conditions]
    solver = get_point_solver(wks.components, sorted(wks.phases), dict(wks.models),
                              wks.phase_record_factory, backend='cpp')
    ds = solver.dynamic_sizes
    MP, MC, MDOF = solver.MP, solver.MC, solver.MDOF
    pdof = pr.phase_dof

    # Point list: state variables and prescribed mole fractions per step
    T = np.array([c['T'] for c in values], dtype=np.float64)
    P = np.array([c['P'] for c in values], dtype=np.float64)
    N = np.array([c.get('N', 1.0) for c in values], dtype=np.float64)
    prescribed = [el for el in nonvacant if f'X_{el}' in values[0]]
    X = np.zeros((n, ncomp))
    mask = np.zeros((n, ncomp), dtype=bool)
    for el in prescribed:
        X[:, nonvacant.index(el)] = [c[f'X_{el}'] for c in values]
        mask[:, nonvacant.index(el)] = True
    free = [i for i in range(ncomp) if not mask[0, i]]
    if len(free) == 1:  # balance component, as PointList expects full compositions
        X[:, free[0]] = 1.0 - X.sum(axis=1)
    points = PointList(T=T, P=P, N=N, X=X, x_cond_mask=mask)

    # Starting data: the composition set's current state (step 0); later steps
    # overwrite the amount and site fractions from the previous result.
    # Solver.solve starts from zero chemical potentials.
    hull = {'Phase': np.array([[phase_name]] * n, dtype=object), 'NP': np.ones((n, 1)),
            'Y': np.zeros((n, 1, MDOF)), 'X': np.zeros((n, 1, ncomp)), 'MU': np.zeros((n, ncomp))}
    hull['NP'][:, 0] = compset.NP
    hull['Y'][:, 0, :pdof] = np.asarray(compset.dof)[len(state_variables):len(state_variables) + pdof]
    spec_row0 = solver.build_spec_row0({'N': N[0], 'P': P[0], 'T': T[0]},
                                       x_conditions={el: X[0, nonvacant.index(el)] for el in prescribed})
    specs = np.ascontiguousarray(build_spec_rows(points, spec_row0, hull, ds, nonvacant))
    cond_args = np.ascontiguousarray(build_condition_args(points, pr.state_variables,
                                                          [str(c) for c in solver.shim.components],
                                                          solver.MSV, MC, nonvacant))
    ipd = np.ascontiguousarray(_create_initial_phase_data_struct_array(
        build_initial_phase_data(points, hull, solver.name_to_idx, ds), n, ds, False))
    _, table = allocate_work_arrays(np, ds, 1, lambda a: a.ctypes.data)
    work = _  # keep the work arrays alive
    results = np.zeros(solver.results_per_condition)
    y0 = 6 + MC + MP

    reference = Solver()
    steps = []
    y = hull['Y'][0, 0, :pdof].copy()
    amount = float(compset.NP)
    for k in range(n):
        row = ipd[k]
        row[MP] = amount
        row[2 * MP:2 * MP + pdof] = y
        results[:] = 0.0
        run_cpu_backend(solver.lib, system_spec=specs[k], condition_args_doubles=cond_args[k],
                        results=results, num_conditions=1, condition_stride=cond_args.shape[1],
                        python_max_statevars=solver.MSV, initial_phase_data=row,
                        initial_phase_data_stride=ipd.shape[1], system_spec_stride=specs.shape[1],
                        grid_data=None, grid_block_indices=None, grid_block_stride_bytes=0,
                        work_arrays_ptr_table=table, threads=1)
        converged = results[1 + MC + MP] > 0.5
        y_new = results[y0:y0 + pdof].copy()
        amount_new = results[1 + MC]
        mu = results[1:1 + ncomp].copy()
        if not (converged and np.all(np.isfinite(y_new)) and np.isfinite(amount_new)):
            # reference Solver.solve from the same starting state
            statevars = np.array([values[k][sv] for sv in state_variables], dtype=np.float64)
            compset.update(y, amount, statevars)
            res = reference.solve([compset], step_conditions[k])
            y_new = np.asarray(compset.dof)[len(state_variables):len(state_variables) + pdof].copy()
            amount_new = compset.NP
            mu = np.asarray(res.chemical_potentials, dtype=np.float64).copy()
        y, amount = y_new, float(amount_new)
        steps.append((y.copy(), amount, mu))
    del work
    statevars = np.array([values[-1][sv] for sv in state_variables], dtype=np.float64)
    compset.update(y, amount, statevars)
    return steps
