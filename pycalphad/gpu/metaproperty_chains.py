"""
Compiled evaluation of IsolatedPhase and DormantPhase properties over a
condition grid.

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
with the reference Solver.solve from the same starting state. If that does
not converge either, the step is retried once from the IsolatedPhase's
starting composition set, as the reference does (gh-670); a step that still
does not converge evaluates its properties without a composition set.

DormantPhase relaxes its composition set at every condition under that
condition's equilibrium chemical potentials and state variables, by a fixed
number of steps, again starting from where the previous evaluation left it;
the same sequence runs on the compiled solver's dormant mode.
"""
import numpy as np

from pycalphad import variables as v

__all__ = ['chain_supported', 'isolated_phase_steps', 'dormant_phase_steps', 'relax_dormant_compsets']


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


class _StepSolver:
    """The compiled point solver set up for single-phase steps: row k solves
    or relaxes one composition set of phase `phase_names[k]` at the state
    variables and prescribed mole fractions in `values[k]` (str -> float),
    with no grid (no phase additions)."""

    def __init__(self, components, phases, models, phase_record_factory, phase_names, values, n_slots=1):
        from pycalphad.gpu.gpu_equilibrium import _create_initial_phase_data_struct_array
        from pycalphad.gpu.gpu_systemspec_flat import spec_layout
        from pycalphad.gpu.point_solver import (PointList, build_condition_args, build_initial_phase_data,
                                                build_spec_rows, get_point_solver)
        from pycalphad.gpu.work_arrays import allocate_work_arrays

        records = [phase_record_factory[name] for name in phase_names]
        nonvacant = list(records[0].nonvacant_elements)
        ncomp = len(nonvacant)
        self.state_variables = [str(sv) for sv in records[0].state_variables]
        n = len(values)
        self.values = values
        solver = get_point_solver(components, sorted(phases), dict(models), phase_record_factory, backend='cpp')
        ds = solver.dynamic_sizes
        MP, MC, MDOF = solver.MP, solver.MC, solver.MDOF
        self.solver, self.ncomp, self.MP, self.MC = solver, ncomp, MP, MC
        self.pdofs = [pr.phase_dof for pr in records]

        # Point list: state variables and prescribed mole fractions per row
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

        # Starting data: the row's phase; the amount and site fractions are
        # written per call.
        hull = {'Phase': np.array([[name] for name in phase_names], dtype=object), 'NP': np.ones((n, 1)),
                'Y': np.zeros((n, 1, MDOF)), 'X': np.zeros((n, 1, ncomp)), 'MU': np.zeros((n, ncomp))}
        spec_row0 = solver.build_spec_row0({'N': N[0], 'P': P[0], 'T': T[0]},
                                           x_conditions={el: X[0, nonvacant.index(el)] for el in prescribed})
        self.specs = np.ascontiguousarray(build_spec_rows(points, spec_row0, hull, ds, nonvacant))
        self.mu_offset = spec_layout(ds)['initial_chemical_potentials']
        self.cond_args = np.ascontiguousarray(build_condition_args(points, records[0].state_variables,
                                                                   [str(c) for c in solver.shim.components],
                                                                   solver.MSV, MC, nonvacant))
        self.ipd = np.ascontiguousarray(_create_initial_phase_data_struct_array(
            build_initial_phase_data(points, hull, solver.name_to_idx, ds), n, ds, False))
        self._work, self.table = allocate_work_arrays(np, ds, n_slots, lambda a: a.ctypes.data)
        self.results = np.zeros((n, solver.results_per_condition))

    def statevars(self, k):
        return np.array([self.values[k][sv] for sv in self.state_variables], dtype=np.float64)

    def set_chemical_potentials(self, k, chemical_potentials):
        self.specs[k, self.mu_offset:self.mu_offset + self.ncomp] = chemical_potentials

    def set_start(self, k, y, amount):
        row = self.ipd[k]
        row[self.MP] = amount
        row[2 * self.MP:2 * self.MP + self.pdofs[k]] = y

    def result(self, k):
        """(converged, site fractions, amount, chemical potentials) of row k."""
        MP, MC = self.MP, self.MC
        r = self.results[k]
        y0 = 6 + MC + MP
        return (r[1 + MC + MP] > 0.5, r[y0:y0 + self.pdofs[k]].copy(), float(r[1 + MC]),
                r[1:1 + self.ncomp].copy())

    def _run(self, rows, max_solver_iterations, threads):
        from pycalphad.gpu.cpu_backend import run_cpu_backend
        self.results[rows] = 0.0
        run_cpu_backend(self.solver.lib, system_spec=self.specs[rows], condition_args_doubles=self.cond_args[rows],
                        results=self.results[rows], num_conditions=self.specs[rows].shape[0],
                        condition_stride=self.cond_args.shape[1], python_max_statevars=self.solver.MSV,
                        initial_phase_data=self.ipd[rows], initial_phase_data_stride=self.ipd.shape[1],
                        system_spec_stride=self.specs.shape[1], grid_data=None, grid_block_indices=None,
                        grid_block_stride_bytes=0, work_arrays_ptr_table=self.table,
                        max_solver_iterations=max_solver_iterations, threads=threads)

    def run(self, k, y, amount, max_solver_iterations=1000):
        """Solve row k alone from site fractions `y` and amount `amount`."""
        self.set_start(k, y, amount)
        self._run(slice(k, k + 1), max_solver_iterations, 1)
        return self.result(k)

    def run_all(self, max_solver_iterations=1000):
        """Solve every row from its starting data (one work slot per row)."""
        self._run(slice(None), max_solver_iterations, None)


def _workspace_step_solver(wks, compset, step_conditions):
    values = [{str(key): float(value) for key, value in conds.items()} for conds in step_conditions]
    return _StepSolver(wks.components, wks.phases, wks.models, wks.phase_record_factory,
                       [compset.phase_record.phase_name] * len(values), values)


def _site_fractions(compset):
    nsv = len(compset.phase_record.state_variables)
    return np.asarray(compset.dof)[nsv:nsv + compset.phase_record.phase_dof].copy()


def isolated_phase_steps(wks, compset, step_conditions, initial_compset=None):
    """Solve `compset` alone at each step's conditions in sequence, each step
    starting from the previous result (IsolatedPhase semantics). A step that
    does not converge is retried once from `initial_compset` (the
    IsolatedPhase's starting composition set), if given.

    step_conditions: list of Workspace.get_dict's cur_conds (condition object
    -> value) per step. Returns a list of (site_fractions, NP,
    chemical_potentials, converged) per step; `compset` is left in the final
    state, as Solver.solve leaves it.
    """
    from pycalphad.core.solver import Solver
    steps_solver = _workspace_step_solver(wks, compset, step_conditions)
    reference = Solver()

    def solve(k, y, amount):
        converged, y_new, amount_new, mu = steps_solver.run(k, y, amount)
        if converged and np.all(np.isfinite(y_new)) and np.isfinite(amount_new):
            return True, y_new, float(amount_new), mu
        # reference Solver.solve from the same starting state
        compset.update(y, amount, steps_solver.statevars(k))
        res = reference.solve([compset], step_conditions[k])
        return (bool(res.converged), _site_fractions(compset), float(compset.NP),
                np.asarray(res.chemical_potentials, dtype=np.float64).copy())

    steps = []
    y, amount = _site_fractions(compset), float(compset.NP)
    for k in range(len(step_conditions)):
        converged, y, amount, mu = solve(k, y, amount)
        if not converged and initial_compset is not None:
            converged, y, amount, mu = solve(k, _site_fractions(initial_compset), float(initial_compset.NP))
        steps.append((y.copy(), amount, mu, converged))
    compset.update(y, amount, steps_solver.statevars(len(step_conditions) - 1))
    return steps


def dormant_phase_steps(wks, compset, step_conditions, step_chemical_potentials, iterations):
    """Relax `compset` at each step's state variables and chemical potentials
    in sequence, each step starting from the previous result (DormantPhase
    semantics: `iterations` fixed-potential steps, see the dormant mode of
    solve_equilibrium_at_condition).

    Returns a list of (site_fractions, NP) per step; `compset` is left in the
    final state, as DormantPhase leaves it.
    """
    steps_solver = _workspace_step_solver(wks, compset, step_conditions)
    steps = []
    y, amount = _site_fractions(compset), float(compset.NP)
    for k in range(len(step_conditions)):
        steps_solver.set_chemical_potentials(k, step_chemical_potentials[k])
        _, y, amount, _ = steps_solver.run(k, y, amount, max_solver_iterations=-int(iterations))
        steps.append((y.copy(), amount))
    compset.update(y, amount, steps_solver.statevars(len(step_conditions) - 1))
    compset.fixed = False  # (SystemState sets it)
    return steps


def relax_dormant_compsets(phase_record_factory, compsets, statevars, chemical_potentials, iterations):
    """DormantPhase's relaxation of each of `compsets` (phase records from
    `phase_record_factory`), independently, at the state variables
    `statevars` (in the phase records' order) and chemical potentials
    `chemical_potentials`. Updates the composition sets in place as
    DormantPhase does. Returns False, changing nothing, where unsupported."""
    if not compsets:
        return True
    models = phase_record_factory.models
    components = [v.Species(el) for el in phase_record_factory.pure_elements]
    state_variables = [str(sv) for sv in phase_record_factory.state_variables]
    if not set(state_variables) <= {'N', 'P', 'T'} or len(phase_record_factory.param_symbols) > 0:
        return False
    if any(cs.phase_record.phase_record_factory is not phase_record_factory for cs in compsets):
        return False
    nonvacant = list(compsets[0].phase_record.nonvacant_elements)
    if len(chemical_potentials) != len(nonvacant):
        return False
    point = dict(zip(state_variables, (float(x) for x in statevars)))
    # (any feasible composition: the relaxation does not use it)
    point.update({f'X_{el}': 1.0 / len(nonvacant) for el in nonvacant[:-1]})
    if 'P' not in point or 'T' not in point:
        return False
    solver = _StepSolver(components, sorted(models), models, phase_record_factory,
                         [cs.phase_record.phase_name for cs in compsets], [point] * len(compsets),
                         n_slots=len(compsets))
    for k, cs in enumerate(compsets):
        solver.set_chemical_potentials(k, chemical_potentials)
        solver.set_start(k, _site_fractions(cs), float(cs.NP))
    solver.run_all(max_solver_iterations=-int(iterations))
    sv_array = np.asarray(statevars, dtype=np.float64)
    for k, cs in enumerate(compsets):
        _, y, amount, _ = solver.result(k)
        cs.update(y, amount, sv_array)
        cs.fixed = False  # (SystemState sets it)
    return True
