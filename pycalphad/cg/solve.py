"""
Driver for the column-generation solver: phases from the workspace's PhaseRecords, condition rows, the sweep over
the condition grid (warm-started; points are kept while the state variables stay the same), and the equilibrium
dataset in the layout of the reference implementation.
"""
import itertools
from collections import OrderedDict

import numpy as np

from pycalphad import variables as v
from pycalphad.backend import AcceleratedCapabilityError
from pycalphad.core.constants import COMP_DIFFERENCE_TOL
from pycalphad.core.light_dataset import LightDataset

__all__ = ['solve_workspace', 'CGCapabilityError']


class CGCapabilityError(AcceleratedCapabilityError):
    """A problem the column-generation solver does not handle (yet); the reference implementation is used."""


# Phases where every sublattice can be all vacancies: the energy per mole of atoms is unbounded below as the formula
# unit empties of atoms (about RT ln(atoms per site)). Points must keep at least this fraction of the atom sites.
MIN_FILLED_FRACTION = 0.25
# Ordered start configurations per phase (end members mixed with the overall composition), at most
MAX_ORDERED_SEEDS = 16
# End-member pair midpoints per phase, at most (as pycalphad's sampling of end-member pairs: a miscibility gap or a
# strongly non-ideal multicomponent solution can have minima only along such edges), and random interior points
MAX_PAIR_SEEDS = 64
import os as _os
_DEBUG = bool(_os.environ.get('PYCALPHAD_CG_DEBUG'))
RANDOM_SEEDS = 8


class _Phase:
    """Static structure of one phase for the solver: the compiled core object, sublattice blocks, species."""
    def __init__(self, pr, elements):
        from pycalphad.cg._core import CPhase
        variables = list(pr.variables)
        self.name = pr.phase_name
        self.D = len(variables)
        self.variables = variables
        self.subl = np.array([y.sublattice_index for y in variables], dtype=np.int32)
        self.blocks = [np.flatnonzero(self.subl == s) for s in range(int(self.subl.max()) + 1)]
        species = [y.species for y in variables]
        self.species_names = [str(sp.name) for sp in species]
        elem_of = []
        for sp in species:
            consts = {str(k): float(val) for k, val in sp.constituents.items()}
            name = str(sp.name)
            elem_of.append(elements.index(name) if name in elements and consts == {name: 1.0} else -1)
        self.elem_of = np.array(elem_of, dtype=np.int32)
        self.A, self.linear = _formula_moles(pr, self.blocks, len(elements))
        self.E, self.e = _extra_constraints(pr, self.blocks)
        sv_names = [str(s) for s in pr.state_variables]
        if 'T' not in sv_names:
            raise CGCapabilityError('no temperature state variable')
        self.core = CPhase(pr, sv_names.index('T'), self.subl, self.elem_of, self.A, self.linear, self.E, self.e)
        if self.linear and all(any(self.species_names[j] == 'VA' for j in blk) for blk in self.blocks):
            asum = self.A.sum(axis=0)
            m_max = sum(asum[blk].max() for blk in self.blocks)
            self.core.set_bounds(MIN_FILLED_FRACTION * m_max)
        self.m_lo = self.core.m_lo
        self.clones = []   # copies of core for worker threads (own buffers, shared compiled functions)
        self.stoichiometric = all(len(b) == 1 for b in self.blocks)

    def core_for(self, worker):
        """The phase's core object for worker thread `worker` (0: the original)."""
        if worker == 0:
            return self.core
        while len(self.clones) < worker:
            self.clones.append(self.core.clone())
        return self.clones[worker - 1]

    def _norm(self, y):
        y = np.maximum(y, 1e-14)
        for b in self.blocks:
            y[b] /= y[b].sum()
        return y

    def composition_start(self, comp, elements):
        y = np.full(self.D, 1e-6)
        for b in self.blocks:
            for j in b:
                e = self.elem_of[j]
                if e >= 0:
                    y[j] = max(comp[e], 1e-6)
                elif len(b) == 1:
                    y[j] = 1.0
        return self._norm(y)

    def seeds(self, z, elements, rng):
        """Start configurations: the composition z, z enriched in each element, and ordered end members (sampled
        when there are many) mixed with z."""
        C = len(elements)
        out = [self.composition_start(z, elements)]
        for k in range(C):
            out.append(self.composition_start(0.4 * np.asarray(z) + 0.6 * np.eye(C)[k], elements))
        ems = list(itertools.product(*[range(len(b)) for b in self.blocks]))
        if len(ems) > MAX_ORDERED_SEEDS:
            ems = [ems[i] for i in rng.choice(len(ems), MAX_ORDERED_SEEDS, replace=False)]
        end_members = []
        for em in ems:
            y = np.full(self.D, 1e-6)
            for si, b in enumerate(self.blocks):
                y[b[em[si]]] = 1.0
            end_members.append(self._norm(y))
            out.append(self._norm(0.5 * end_members[-1] + 0.5 * out[0]))
        if not self.stoichiometric:
            pairs = list(itertools.combinations(range(len(end_members)), 2))
            if len(pairs) > MAX_PAIR_SEEDS:
                pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIR_SEEDS, replace=False)]
            for i, j in pairs:
                out.append(self._norm(0.5 * end_members[i] + 0.5 * end_members[j]))
            for _ in range(RANDOM_SEEDS):
                y = np.empty(self.D)
                for b in self.blocks:
                    y[b] = rng.dirichlet(np.ones(len(b)))
                out.append(self._norm(y))
        if self.m_lo > 0:
            out = [y for y in out if self.A.sum(axis=0) @ y >= self.m_lo]
        return out


def _sample_dof(pr, blocks, rng):
    ns, D = pr.num_statevars, pr.phase_dof
    y = rng.random(D) + 0.1
    for b in blocks:
        y[b] /= y[b].sum()
    sv = np.array([1.0, 101325.0, 1000.0]) if ns == 3 else np.full(ns, 1000.0)
    return y, np.concatenate([sv, y])


def _formula_moles(pr, blocks, C):
    """(A, linear): if the PhaseRecord's moles of each element per formula unit are linear in the site fractions,
    A with m_i = A[i] . y; else (ionic liquids, whose site ratios depend on y) linear is False and the solver calls
    the formula-mole functions."""
    ns, D = pr.num_statevars, pr.phase_dof
    rng = np.random.default_rng(0)
    A = np.zeros((C, D))
    for trial in range(2):
        y, dof = _sample_dof(pr, blocks, rng)
        for i in range(C):
            grad = np.zeros(ns + D)
            pr.formulamole_grad(grad, dof, i)
            val = np.zeros(1)
            pr.formulamole_obj(val, dof, i)
            if trial == 0:
                A[i] = grad[ns:]
            if not np.allclose(grad[ns:], A[i], atol=1e-12) or abs(A[i] @ y - val[0]) > 1e-10:
                return np.zeros((C, D)), False
    return A, True


def _extra_constraints(pr, blocks):
    """(E, e): the PhaseRecord's internal constraints other than the site-fraction balances (charge balance), as
    E y = e. They must be linear in the site fractions."""
    ns, D = pr.num_statevars, pr.phase_dof
    nc = pr.num_internal_cons
    rng = np.random.default_rng(1)
    rows = []
    jacs = []
    for trial in range(2):
        y, dof = _sample_dof(pr, blocks, rng)
        jac = np.zeros((nc, ns + D)); pr.internal_cons_jac(jac, dof)
        val = np.zeros(nc); pr.internal_cons_func(val, dof)
        jacs.append((jac[:, ns:], val, y))
    J0, v0, y0 = jacs[0]
    J1, v1, y1 = jacs[1]
    for k in range(nc):
        row = J0[k]
        support = np.flatnonzero(np.abs(row) > 0)
        is_balance = any(set(support) == set(b) and np.allclose(row[b], row[b[0]]) for b in blocks)
        if is_balance:
            continue
        if not np.allclose(J1[k], row, atol=1e-12):
            raise CGCapabilityError(f'{pr.phase_name}: non-linear internal constraint')
        rows.append((row, row @ y0 - v0[k]))
    if not rows:
        return np.zeros((0, D)), np.zeros(0)
    return np.array([r for r, _ in rows]), np.array([c for _, c in rows])


# Phases across calls: every equilibrium() call builds new Models and a new PhaseRecordFactory, and pycalphad's
# build_functions cache (100 entries) does not hold a large system's functions, so they would be recompiled each
# time. Keyed by what determines the compiled functions; parameter values are passed in at solve time.
_PHASE_CACHE = OrderedDict()
_PHASE_CACHE_SIZE = 1024


def _phase_key(prf, name, elements):
    mod = prf.models[name]
    return (type(mod), name, tuple(str(c) for c in prf.comps), tuple(elements),
            tuple(str(sv) for sv in prf.state_variables), tuple(str(p) for p in prf.param_symbols),
            tuple(str(y) for y in mod.site_fractions), mod.ast)


def _phases_for(prf, elements):
    """_Phase objects for the workspace's phase records (cached across calls, see _PHASE_CACHE)."""
    phases = []
    for name in sorted(prf.keys()):
        try:
            key = _phase_key(prf, name, elements)
            ph = _PHASE_CACHE.get(key)
        except TypeError:   # (an unhashable model attribute)
            key, ph = None, None
        if ph is None:
            ph = _Phase(prf[name], list(elements))
            if key is not None:
                _PHASE_CACHE[key] = ph
                if len(_PHASE_CACHE) > _PHASE_CACHE_SIZE:
                    _PHASE_CACHE.popitem(last=False)
        else:
            _PHASE_CACHE.move_to_end(key)
        phases.append(ph)
    return phases


def _gauge_directions(phases, C):
    """Orthonormal directions w with w.x = 0 for every admissible composition of every phase (see Solver.set_gauge).
    With linear formula moles m = A y on the affine set {B y = c} (site balances and the phase's constraints),
    w.x vanishes everywhere iff w is orthogonal to A y0 (y0 any point of the set) and to A N (N spanning B's null
    space). Phases with non-linear formula moles are taken to allow no such direction."""
    rows = []
    for ph in phases:
        if not ph.linear:
            return np.zeros((0, C))
        B = np.vstack([np.isin(np.arange(ph.D), blk).astype(float) for blk in ph.blocks] + ([ph.E] if len(ph.E) else []))
        c = np.concatenate([np.ones(len(ph.blocks)), ph.e if len(ph.E) else np.zeros(0)])
        y0 = np.linalg.lstsq(B, c, rcond=None)[0]
        _, sv, vt = np.linalg.svd(B)
        rank = int(np.sum(sv > 1e-10 * max(sv[0], 1.0)))
        N = vt[rank:].T
        rows.append(ph.A @ y0)
        if N.shape[1]:
            rows.extend((ph.A @ N).T)
    M = np.array(rows)
    _, sv, vt = np.linalg.svd(M, full_matrices=True)
    rank = int(np.sum(sv > 1e-10 * max(sv[0], 1.0)))
    return vt[rank:]


def _condition_rows(point, elements, molar_masses):
    """Rows (R, b) of sum_p n_p R x_p = b, fixed chemical potentials and their flags, for one condition point
    (mirrors pycalphad.core.solver.Solver's prescribed mole-fraction rows)."""
    C = len(elements)
    N = float(point.get(v.N, 1.0))
    rows = [np.ones(C)]; rhs = [N]
    mu_fixed = np.zeros(C); fixed = np.zeros(C, dtype=np.int32)
    for cond, value in point.items():
        value = float(value)
        if cond in (v.N, v.P, v.T) or getattr(cond, 'phase_name', None) is not None:
            continue   # (phase-local conditions constrain their phase's points: _local_constraints)
        if isinstance(cond, v.MoleFraction) and cond.phase_name is None:
            c = np.zeros(C); c[elements.index(str(cond)[2:])] = 1.0
            rows.append(c); rhs.append(value * N)
        elif isinstance(cond, v.MassFraction) and cond.phase_name is None:
            c = np.full(C, -value); c[elements.index(str(cond)[2:])] += 1.0
            rows.append(c * molar_masses); rhs.append(0.0)
        elif isinstance(cond, v.ChemicalPotential):
            i = elements.index(str(cond)[3:])
            fixed[i] = 1; mu_fixed[i] = value
        elif str(cond).startswith('LinComb_'):
            c = np.zeros(C); constant = 0.0
            for symbol, coef in zip(cond.symbols, cond.coefs):
                if symbol == 1:
                    constant = coef
                    continue
                c[elements.index(str(symbol)[2:])] = coef
            if cond.denominator == 1:
                rows.append(c); rhs.append((value - float(constant)) * N)
            else:
                c[cond.symbols.index(cond.denominator)] -= value
                rows.append(c); rhs.append(-float(constant) * N)
        else:
            raise CGCapabilityError(f'condition {cond}')
    return np.array(rows), np.array(rhs), mu_fixed, fixed


def _local_constraints(ph, point, elements):
    """(E, e, pinned) for the phase-local conditions on this phase: mole fractions X(phase, el) = c as
    (A_el - c sum(A)) y = 0, site fractions 0 < Y = c < 1 as y_j = c, and Y = 0 (or 1, for the sublattice's other
    species) as species held at the floor."""
    E = []; e = []; pinned = np.zeros(ph.D, dtype=np.int32)
    for cond, value in point.items():
        if getattr(cond, 'phase_name', None) != ph.name:
            continue
        value = float(value)
        if isinstance(cond, v.SiteFraction):
            j = [k for k, y in enumerate(ph.variables) if y.sublattice_index == cond.sublattice_index
                 and y.species == cond.species]
            if not j:
                raise CGCapabilityError(f'site fraction condition {cond} not in the phase')
            j = j[0]
            block = [k for k in ph.blocks[ph.subl[j]] if k != j]
            if value <= 1e-12:
                pinned[j] = 1
            elif value >= 1 - 1e-12:
                pinned[block] = 1
            else:
                row = np.zeros(ph.D); row[j] = 1.0
                E.append(row); e.append(value)
        elif isinstance(cond, v.MoleFraction):
            if not ph.linear:
                raise CGCapabilityError(f'{ph.name}: phase-local mole fraction with non-linear formula moles')
            el = elements.index(str(cond.species))
            E.append(ph.A[el] - value * ph.A.sum(axis=0)); e.append(0.0)
        else:
            raise CGCapabilityError(f'phase-local condition {cond}')
    return np.array(E).reshape(-1, ph.D), np.array(e), pinned


def _nominal_composition(point, elements):
    """A composition for generating start configurations (the given mole fractions, the rest shared equally)."""
    C = len(elements)
    z = np.full(C, np.nan)
    for cond, value in point.items():
        if isinstance(cond, v.MoleFraction) and cond.phase_name is None:
            z[elements.index(str(cond)[2:])] = float(value)
    free = np.isnan(z)
    z[free] = max(1.0 - np.nansum(z[~free]), 0.0) / max(free.sum(), 1)
    z = np.clip(z, 1e-3, None)
    return z / z.sum()


def _consolidate(result):
    """Merge points of the same phase whose compositions agree within COMP_DIFFERENCE_TOL (one composition set)."""
    phs, Y, X, n, G = result
    keep = []
    for a in range(len(phs)):
        for k in keep:
            if phs[k] == phs[a] and np.max(np.abs(X[k] - X[a])) < COMP_DIFFERENCE_TOL:
                if n[a] > n[k]:
                    Y[k], X[k], G[k] = Y[a], X[a], G[a]
                n[k] += n[a]
                break
        else:
            keep.append(a)
    return [phs[k] for k in keep], [Y[k] for k in keep], [X[k] for k in keep], [n[k] for k in keep], [G[k] for k in keep]


# Mole-fraction conditions below this are trace conditions: the master problem's duals for such a row are set by
# amounts ~1e-10 and are unusable, so the point is first solved with the trace values raised to TRACE_START and then
# followed down to them by Newton (dropping phases whose amounts reach zero), and certified there.
TRACE_COMPOSITION = 1e-6
TRACE_START = 1e-4
# Fixed chemical potentials: the point is solved with trial mole fractions for those components, finished by Newton
# under the actual conditions and certified; the trial values are corrected from the potential mismatch.
MAX_POTENTIAL_TRIALS = 40
R_GAS = 8.3145


def _is_global_x(cond):
    return isinstance(cond, v.MoleFraction) and cond.phase_name is None


def _solve_trace(solver, point, elements, molar_masses, seeds):
    """Solve with trace mole fractions raised, then follow them down by Newton and certify; None if that fails."""
    trace = [k for k, val in point.items() if _is_global_x(k) and float(val) < TRACE_COMPOSITION]
    start = OrderedDict(point)
    for k in trace:
        start[k] = TRACE_START
    if sum(float(val) for k, val in start.items() if _is_global_x(k)) >= 1:
        return None
    R, b, mu_fixed, fixed = _condition_rows(start, elements, molar_masses)
    solver.set_conditions(R, b, mu_fixed, fixed)
    out = solver.solve(seeds)
    if out[0] != 0 or not solver.last_polished:
        return None
    decades = max(np.log10(TRACE_START / max(float(point[k]), 1e-300)) for k in trace)
    nsteps = max(int(np.ceil(decades / 2)), 1)
    for step in range(1, nsteps + 1):
        frac = step / nsteps
        cur = OrderedDict(point)
        for k in trace:
            cur[k] = float(np.exp((1 - frac) * np.log(TRACE_START) + frac * np.log(max(float(point[k]), 1e-300))))
        R, b, mu_fixed, fixed = _condition_rows(cur, elements, molar_masses)
        if not solver.polish_from_state(R, b, mu_fixed, fixed):
            return None
    return solver.certify()


def _solve_fixed_potentials(solver, point, elements, molar_masses, seeds, guess):
    """Fixed chemical potentials: solve with trial mole fractions in their place, finish under the actual conditions
    by Newton and certify; correct the trial values from the potential mismatch until that succeeds."""
    R, b, mu_fixed, fixed = _condition_rows(point, elements, molar_masses)
    # direct: the fixed potentials in the costs, the other conditions as rows (always admits a composition, e.g.
    # with charge balance fixing it); the polish solves the actual equations, and only a polished (certified)
    # result is taken
    solver.set_conditions(R, b, mu_fixed, fixed)
    out = solver.solve(seeds)
    seeds = None
    if out[0] == 0 and solver.last_polished:
        return out
    T = float(point[v.T])
    fixed_idx = np.flatnonzero(fixed)
    given_x = sum(float(val) for k, val in point.items() if _is_global_x(k))
    xg = np.array([guess.get(i, (1 - given_x) / (len(fixed_idx) + 1)) for i in fixed_idx])
    out = (6, None, None, None, None, None, None, 0)
    for trial in range(MAX_POTENTIAL_TRIALS):
        xg = np.clip(xg, 1e-10, None)
        if given_x + xg.sum() >= 1:
            xg *= (1 - given_x) * 0.999 / xg.sum()
        trial_point = OrderedDict((k, val) for k, val in point.items() if not isinstance(k, v.ChemicalPotential))
        for i, x in zip(fixed_idx, xg):
            trial_point[v.X(elements[i])] = x
        Rx, bx, mx, fx = _condition_rows(trial_point, elements, molar_masses)
        solver.set_conditions(Rx, bx, mx, fx)
        out = solver.solve(seeds)
        seeds = None
        if out[0] != 0:
            return out
        res = solver.finish(R, b, mu_fixed, fixed)
        if res is not None:
            for i in fixed_idx:
                guess[i] = float(sum(n * x[i] for n, x in zip(res[4], res[3])) / sum(res[4]))
            return res
        mu = out[6]
        step = np.clip((mu_fixed[fixed_idx] - mu[fixed_idx]) / (R_GAS * T), -5, 5)
        xg = xg * np.exp(step)
    return (6, None, None, None, None, None, None, MAX_POTENTIAL_TRIALS)


def _solve_point(solver, point, elements, molar_masses, seeds, guess):
    R, b, mu_fixed, fixed = _condition_rows(point, elements, molar_masses)
    if fixed.any():
        return _solve_fixed_potentials(solver, point, elements, molar_masses, seeds, guess)
    if any(_is_global_x(k) and float(val) < TRACE_COMPOSITION for k, val in point.items()):
        out = _solve_trace(solver, point, elements, molar_masses, seeds)
        if out is not None:
            return out
    solver.set_conditions(R, b, mu_fixed, fixed)
    return solver.solve(seeds)


# Conditions per worker thread below which threads are not worth starting
MIN_CONDITIONS_PER_THREAD = 8
# Conditions per contiguous part of a sequence handed to a thread, at least
MIN_CONDITIONS_PER_PART = 4
PARTS_PER_THREAD = int(_os.environ.get("PYCALPHAD_CG_PARTS", 2))


def _thread_count(n_conditions):
    import os
    from pycalphad.backend import _current
    requested = os.environ.get('PYCALPHAD_CG_THREADS') or _current()[1].threads or os.cpu_count() or 1
    return max(1, min(int(requested), n_conditions // MIN_CONDITIONS_PER_THREAD))


class _Worker:
    """A solver over its own copies of the phases (one per thread)."""
    def __init__(self, index, phases, C, gauge):
        from pycalphad.cg._core import Solver
        self.index = index
        self.cores = [ph.core_for(index) for ph in phases]
        self.solver = Solver(self.cores, C)
        self.solver.set_gauge(gauge)
        self.guess = {}   # trial mole fractions for fixed-potential components, carried between points
        self.state_key = None

    def prepare(self, phases, point, sv_values, params, elements, snapshot, rng):
        """State variables and phase-local conditions for a group of conditions (unless already set); the point
        store from `snapshot`, or reset and seeded (returns the seeds to price in the first solve)."""
        key = (tuple(sv_values), tuple(params),
               tuple((k, float(val)) for k, val in point.items() if getattr(k, 'phase_name', None) is not None))
        if key != self.state_key:
            for ph, core in zip(phases, self.cores):
                E, e, pinned = _local_constraints(ph, point, elements)
                core.set_state(sv_values, params)
                core.set_local_constraints(E, e)
                core.set_pinned(pinned)
            self.state_key = key
        if snapshot is not None:
            self.solver.reset_points()
            self.solver.import_points(snapshot)
            return None
        return self.reseed(phases, point, elements, rng)

    def reseed(self, phases, point, elements, rng):
        self.solver.reset_points()
        z0 = _nominal_composition(point, elements)
        seeds = []
        for i, ph in enumerate(phases):
            S = np.array(ph.seeds(z0, elements, rng)).reshape(-1, ph.D)
            self.solver.add_points(i, S)
            seeds.append(S)
        return seeds


def solve_workspace(wks):
    """The equilibrium dataset (reference layout) for a Workspace, computed by column generation. Conditions are
    solved in sequences that share state variables (warm-started); sequences, or contiguous parts of one, run on
    worker threads (pricing and the LP release the GIL)."""
    import queue
    from concurrent.futures import ThreadPoolExecutor
    from pycalphad import __version__ as pycalphad_version
    from pycalphad.property_framework.units import as_quantity
    conds = OrderedDict((key, np.atleast_1d(np.asarray(as_quantity(key, value).to(key.implementation_units).magnitude,
                                                       dtype=float)))
                        for key, value in wks.conditions.items())
    prf = wks.phase_record_factory
    elements = [str(e) for e in prf.nonvacant_elements]
    C = len(elements)
    state_variables = list(prf.state_variables)
    for sv in state_variables:
        if sv not in conds and sv != v.N:
            raise CGCapabilityError(f'free state variable {sv}')
    phases = _phases_for(prf, elements)
    names = [ph.name for ph in phases]
    params = np.asarray(prf.param_values, dtype=float)
    molar_masses = np.asarray(prf.molar_masses, dtype=float)
    gauge = _gauge_directions(phases, C)

    keys = list(conds.keys())
    shape = tuple(len(conds[k]) for k in keys)
    nvert = C + 1
    maxdof = max(ph.D for ph in phases)
    GM = np.full(shape, np.nan); MU = np.full(shape + (C,), np.nan)
    NP = np.full(shape + (nvert,), np.nan); X = np.full(shape + (nvert, C), np.nan)
    Y = np.full(shape + (nvert, maxdof), np.nan)
    Phase = np.full(shape + (nvert,), '', dtype='U%d' % max(max(len(n) for n in names), 6))
    points = np.full(shape + (nvert,), -1, dtype=np.int32)

    # sequences of conditions sharing state variables and phase-local conditions, in a warm-starting order
    sv_pos = [i for i, k in enumerate(keys) if k in (v.N, v.P, v.T)]
    other_pos = [i for i in range(len(keys)) if i not in sv_pos]
    order = sorted(itertools.product(*[range(s) for s in shape]),
                   key=lambda idx: tuple(idx[i] for i in sv_pos) + tuple(idx[i] for i in other_pos))
    groups = []
    for idx in order:
        point = OrderedDict((k, conds[k][i]) for k, i in zip(keys, idx))
        x_sum = sum(float(val) for k, val in point.items() if isinstance(k, v.MoleFraction) and k.phase_name is None)
        if x_sum > 1:
            continue   # sum of independent mole fractions greater than one: left empty, as by the reference
        sv_values = np.array([float(point.get(sv, 1.0)) for sv in state_variables])
        local = tuple((k, float(val)) for k, val in point.items() if getattr(k, 'phase_name', None) is not None)
        key = (tuple(sv_values), local)
        if not groups or groups[-1][0] != key:
            groups.append((key, sv_values, []))
        groups[-1][2].append((idx, point))

    def write(idx, out):
        status, phs, Ys, Xs, ns, Gs, mu, rounds = out
        phs, Ys, Xs, ns, Gs = _consolidate((list(phs), list(Ys), list(Xs), list(ns), list(Gs)))
        # vertex order as the reference's (its hull lists grid points, which are grouped by sorted phase name)
        perm = sorted(range(len(phs)), key=lambda a: (names[phs[a]], tuple(-Xs[a])))
        phs, Ys, Xs, ns, Gs = ([seq[a] for a in perm] for seq in (phs, Ys, Xs, ns, Gs))
        total = sum(ns)
        GM[idx] = sum(n * g for n, g in zip(ns, Gs)) / total
        MU[idx] = mu
        for a in range(min(len(phs), nvert)):
            Phase[idx + (a,)] = names[phs[a]]
            NP[idx + (a,)] = ns[a]
            X[idx + (a,)] = Xs[a]
            Y[idx + (a, slice(0, len(Ys[a])))] = Ys[a]

    def run(worker, sv_values, items, snapshot):
        """Solve `items` in order on `worker`, starting from `snapshot` or freshly seeded. Returns the point store
        after the first condition (for starting other workers from it)."""
        rng = np.random.default_rng(0)
        seeds = worker.prepare(phases, items[0][1], sv_values, params, elements, snapshot, rng)
        rows = [_condition_rows(point, elements, molar_masses) for _, point in items]
        plain = all(not r[3].any() for r in rows) and not any(
            _is_global_x(k) and float(val) < TRACE_COMPOSITION for _, point in items for k, val in point.items())
        if plain and len(items) > 1:
            # ordinary conditions: the whole sequence in C without the GIL; failures are retried below
            st, na, phs, amt, Xs, Ys, Gs, mus = worker.solver.solve_sequence(
                np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                np.array([r[3] for r in rows]), seeds)
            after_first = None
            for t, (idx, point) in enumerate(items):
                if st[t] == 0:
                    k = na[t]
                    write(idx, (0, list(phs[t, :k]), [Ys[t, a, :phases[phs[t, a]].D] for a in range(k)],
                                [Xs[t, a] for a in range(k)], list(amt[t, :k]), list(Gs[t, :k]), mus[t], 0))
                else:
                    seeds = worker.reseed(phases, point, elements, rng)
                    out = _solve_point(worker.solver, point, elements, molar_masses, seeds, worker.guess)
                    if out[0] == 0:
                        write(idx, out)
            return worker.solver.export_points()
        after_first = None
        for n, (idx, point) in enumerate(items):
            out = _solve_point(worker.solver, point, elements, molar_masses, seeds, worker.guess)
            seeds = None
            if out[0] != 0:
                # retry from scratch (fresh seeds at this composition)
                seeds = worker.reseed(phases, point, elements, rng)
                out = _solve_point(worker.solver, point, elements, molar_masses, seeds, worker.guess)
                seeds = None
            if out[0] == 0:
                write(idx, out)
            if n == 0:
                after_first = worker.solver.export_points()
        return after_first

    n_conditions = sum(len(g[2]) for g in groups)
    nthreads = _thread_count(n_conditions)
    workers = [_Worker(i, phases, C, gauge) for i in range(nthreads)]
    if nthreads == 1:
        for key, sv_values, items in groups:
            run(workers[0], sv_values, items, None)
    else:
        idle = queue.Queue()
        for w in workers:
            idle.put(w)

        def task(sv_values, items, snapshot):
            w = idle.get()
            try:
                return run(w, sv_values, items, snapshot)
            finally:
                idle.put(w)

        with ThreadPoolExecutor(nthreads) as pool:
            if len(groups) >= nthreads:
                # whole sequences in parallel (e.g. one per temperature)
                for f in [pool.submit(task, sv_values, items, None) for key, sv_values, items in groups]:
                    f.result()
            else:
                # one sequence at a time: seed with its first condition, then contiguous parts in parallel, each
                # starting from the seeded point store
                for key, sv_values, items in groups:
                    snapshot = pool.submit(task, sv_values, items[:1], None).result()
                    rest = items[1:]
                    # several parts per thread, taken in turn, so faster cores (e.g. performance vs efficiency cores)
                    # take more of them
                    nparts = min(PARTS_PER_THREAD * nthreads, max(1, len(rest) // MIN_CONDITIONS_PER_PART))
                    bounds = np.linspace(0, len(rest), nparts + 1).astype(int)
                    futures = [pool.submit(task, sv_values, rest[bounds[i]:bounds[i + 1]], snapshot)
                               for i in range(nparts) if bounds[i + 1] > bounds[i]]
                    for f in futures:
                        f.result()

    coords = OrderedDict((str(k), conds[k]) for k in keys)
    coords['vertex'] = np.arange(nvert)
    coords['component'] = elements
    dims = [str(k) for k in keys]
    data = {'NP': (dims + ['vertex'], NP), 'GM': (dims, GM), 'MU': (dims + ['component'], MU),
            'X': (dims + ['vertex', 'component'], X), 'Y': (dims + ['vertex', 'internal_dof'], Y),
            'Phase': (dims + ['vertex'], Phase), 'points': (dims + ['vertex'], points)}
    return LightDataset(data, coords=coords, attrs={'engine': 'pycalphad %s' % pycalphad_version})
