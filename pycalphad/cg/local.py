"""
Local equilibrium and stability checks on composition sets for the column-generation backend, for the callers that
work with composition sets directly (mapping):

- :class:`CGSolver`, a drop-in for pycalphad.core.solver.Solver: Newton on the equilibrium conditions with the
  composition sets' site fractions at their local driving-force minima (unknowns: chemical potentials, free phase
  amounts, free state variables; equations: zero driving force for every composition set, the condition rows, the
  fixed chemical potentials).
- :func:`global_min_candidate`, the composition set with the largest driving force at a point's chemical
  potentials, from local minimizations per phase (no energy grid).
"""
import numpy as np

from pycalphad import variables as v
from pycalphad.core.constants import COMP_DIFFERENCE_TOL
from pycalphad.core.solver import SolverBase, SolverResult
from pycalphad.cg.solve import (_phases_for, _condition_rows, _local_constraints, CGCapabilityError)

__all__ = ['CGSolver', 'global_min_candidate']

R_GAS = 8.3145
import os as _os
_DEBUG = bool(_os.environ.get('PYCALPHAD_CG_DEBUG'))
MAX_ITERATIONS = 60
# Step limits: chemical potentials (in RT) and temperature (K) per Newton step
MAX_MU_STEP = 5.0
MAX_T_STEP = 50.0
# A composition set whose site fractions change by more than JUMP (and more than JUMP_RATIO times its composition)
# in one solve changed constitution discontinuously (e.g. ordered -> disordered across a first-order transition): the
# step crossed a two-phase field it should have resolved, so the solve is reported as not converged and the caller
# (mapping) takes a smaller step. A continuous change shrinks with the step; a first-order jump does not.
JUMP = 0.3
JUMP_RATIO = 10.0
# Likewise a composition set whose composition moves by more than JUMP_X and by more than JUMP_X_RATIO times the
# change of the overall composition in one solve switched to another basin of its phase (it then takes the place of
# another composition set of that phase, e.g. at a node)
JUMP_X = 0.1
JUMP_X_RATIO = 5.0
# global_min_candidate: a candidate close to a stable composition set's composition with a tiny driving force is not
# a new phase: within TIE_X_NEAR (mole fraction) and below TIE_DG (J/mol), that set's state in another description (a
# tie at rounding level); within TIE_X and below TIE_DG_FAR, the other side of a miscibility gap that has just closed
# at its critical point. A phase that really appears there gains driving force quickly and is found at the next step
TIE_X_NEAR = 1e-3
TIE_DG = 1.0
TIE_X = 0.03
TIE_DG_FAR = 0.01
# Two composition sets of one phase may meet in a solve (a miscibility gap closing) only if they started closer than
# this (site fractions)
COLLAPSE = 0.02


def _phase_lookup(compsets):
    prf = compsets[0].phase_record.phase_record_factory
    elements = [str(e) for e in prf.nonvacant_elements]
    phases = _phases_for(prf, elements)
    return prf, elements, {ph.name: ph for ph in phases}


def _as_statevar_conditions(conditions):
    out = {}
    for key, value in conditions.items():
        key = v.N if str(key) == 'N' else v.P if str(key) == 'P' else v.T if str(key) == 'T' else key
        out[key] = float(np.asarray(value).reshape(-1)[0])
    return out


def _initial_potentials(compsets, phases, mu_fixed, fixed):
    """Chemical potentials (J/mol) from the composition sets as they are: least squares over each set's zero
    driving force and the stationarity of its site fractions (J (dGM/dy - sum_i mu_i dx_i/dy) = 0), with the fixed
    potentials as hard rows."""
    rows = []; rhs = []
    C = len(mu_fixed)
    for cs in compsets:
        ph = phases[cs.phase_record.phase_name]
        y = np.array(cs.dof[cs.phase_record.num_statevars:], dtype=float)
        GM, x = ph.core.evaluate(y)
        rows.append(x); rhs.append(GM)
        if ph.core.ne == 0 and not ph.stoichiometric:
            # stationarity: per sublattice, dGM/dy_j - mean = sum_i mu_i (dx_i/dy_j - mean)
            eps = 1e-7
            g = np.zeros(ph.D); dx = np.zeros((C, ph.D))
            for j in range(ph.D):
                yp = y.copy(); yp[j] += eps
                Gp, xp = ph.core.evaluate(yp)
                g[j] = (Gp - GM) / eps; dx[:, j] = (xp - x) / eps
            for blk in ph.blocks:
                if len(blk) < 2:
                    continue
                w = y[blk]
                gb = g[blk] - w @ g[blk]
                dxb = dx[:, blk] - (dx[:, blk] @ w)[:, None]
                for k, j in enumerate(blk):
                    rows.append(dxb[:, k] * np.sqrt(max(y[j], 1e-12)))
                    rhs.append(gb[k] * np.sqrt(max(y[j], 1e-12)))
    A = np.array(rows); b = np.array(rhs)
    for i in np.flatnonzero(fixed):
        row = np.zeros(C); row[i] = 1e6
        A = np.vstack([A, row]); b = np.append(b, 1e6 * mu_fixed[i])
    return np.linalg.lstsq(A, b, rcond=None)[0]


_EQUIV = None   # set per solve: (phases, state variables, index of T) for _same_state


def _same_state(sets, ys, xs, a, b):
    """Whether composition sets a and b ended in the same state: of one phase at the same site fractions, or of two
    descriptions of a phase (different names, e.g. an ordered model in its disordered state and the disordered
    model) at the same composition with equal energies also at another temperature (two different phases at one
    composition, e.g. solid and liquid at a melting point, differ there)."""
    if sets[a].phase_record.phase_name == sets[b].phase_record.phase_name:
        return np.allclose(ys[a], ys[b], atol=10 * COMP_DIFFERENCE_TOL)
    if np.max(np.abs(np.asarray(xs[a]) - np.asarray(xs[b]))) >= 10 * COMP_DIFFERENCE_TOL or _EQUIV is None:
        return False
    phases, sv, it = _EQUIV
    pa, pb = phases[sets[a].phase_record.phase_name], phases[sets[b].phase_record.phase_name]
    other = sv.copy(); other[it] += 100.0
    try:
        pa.core.set_statevars(other); pb.core.set_statevars(other)
        ga = pa.core.evaluate(ys[a])[0]; gb = pb.core.evaluate(ys[b])[0]
    finally:
        pa.core.set_statevars(sv); pb.core.set_statevars(sv)
    # (J/mol: two descriptions of one phase differ only by a slight ordering there; two phases by tens of J/mol)
    return abs(ga - gb) < 5.0


def _collapsed(xs0, xs, ys, sets, active):
    """Whether two composition sets ended in the same state in a way that is a trivial solution (one state twice; any
    free state variable value satisfies it), not a two-phase equilibrium closing at a critical point: they started
    more than COLLAPSE apart, or one of them jumped onto the other (at a critical point both approach it)."""
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            if not _same_state(sets, ys, xs, a, b):
                continue
            d0 = float(np.max(np.abs(xs0[a] - xs0[b])))
            da = float(np.max(np.abs(np.asarray(xs[a]) - xs0[a])))
            db = float(np.max(np.abs(np.asarray(xs[b]) - xs0[b])))
            if (d0 > COLLAPSE or min(da, db) < 0.2 * max(da, db) or max(da, db) > d0 + COMP_DIFFERENCE_TOL
                    or d0 < COMP_DIFFERENCE_TOL):
                # (a critical point lies between the two: neither moves farther than they were apart; and with the
                # last, one state twice from the start, no two-phase equilibrium at all)
                return True
    return False


def _coincident_pair(sets, ys, xs, active):
    """Whether two of the active composition sets ended in the same state (see _same_state)."""
    return any(_same_state(sets, ys, xs, a, b) for i, a in enumerate(active) for b in active[i + 1:])


def _coincident(sets, ys, xs, active):
    """Whether two of three or more active composition sets ended in the same state (see _same_state; also two
    descriptions of one phase, e.g. at an order-disorder transition of second order): such a node is no equilibrium
    with that many sets. Two sets meeting is left to the caller: that is how a miscibility gap closes at its critical
    point."""
    if len(active) < 3:
        return False
    return _coincident_pair(sets, ys, xs, active)


def _composition_step(sets, conds, elements):
    """Largest change of a prescribed overall mole fraction from the composition sets' current overall composition."""
    total = sum(float(cs.NP) for cs in sets)
    if total <= 0:
        return 0.0
    z = sum(float(cs.NP) * np.asarray(cs.X, dtype=float) for cs in sets) / total
    step = 0.0
    for key, val in conds.items():
        if isinstance(key, v.MoleFraction) and key.phase_name is None and str(key.species) in elements:
            step = max(step, abs(float(val) - z[elements.index(str(key.species))]))
    return step


def _jumped(ys0, ys, xs0, xs, active, is_fixed, step):
    """Whether an active composition set changed discontinuously (see JUMP and JUMP_X)."""
    for k, a in enumerate(active):
        dy = float(np.max(np.abs(ys[a] - ys0[a]))) if len(ys[a]) else 0.0
        dx = float(np.max(np.abs(np.asarray(xs[k]) - xs0[a])))
        if dy > JUMP and dy > JUMP_RATIO * dx:
            return True
        if dx > JUMP_X and dx > JUMP_X_RATIO * step:
            return True
    return False


class CGSolver(SolverBase):
    """Newton on the equilibrium conditions for given composition sets (see the module docstring); the interface
    of pycalphad.core.solver.Solver. Composition sets are updated in place; with remove_metastable, free sets whose
    amounts reach zero are removed."""

    def __init__(self, verbose=False, remove_metastable=True, **options):
        self.verbose = verbose
        self.remove_metastable = remove_metastable
        self.options = options

    def solve(self, composition_sets, conditions):
        """Newton from the composition sets as they are; if two sets of one phase collapsed onto each other (see
        _collapsed), once more with their starting site fractions exchanged: stepping a two-phase line over a
        critical point, the sets change sides, and each then starts on the side it ends on."""
        saved = [(cs, np.array(cs.dof, dtype=float), float(cs.NP)) for cs in composition_sets]
        result = self._solve(composition_sets, conditions)
        if result.converged or self._why != 'composition sets collapsed onto each other':
            return result
        ns = len(composition_sets[0].phase_record.state_variables) if composition_sets else 0
        pairs = [(i, j) for i in range(len(saved)) for j in range(i + 1, len(saved))
                 if saved[i][0].phase_record.phase_name == saved[j][0].phase_record.phase_name]
        if len(pairs) != 1:
            return result
        composition_sets[:] = [cs for cs, _, _ in saved]
        for cs, dof, NP in saved:
            cs.update(dof[ns:], NP, dof[:ns])
        i, j = pairs[0]
        (ci, di, ni), (cj, dj, nj) = saved[i], saved[j]
        ci.update(dj[ns:], ni, di[:ns]); cj.update(di[ns:], nj, dj[:ns])
        swapped = self._solve(composition_sets, conditions)
        if swapped.converged:
            return swapped
        composition_sets[:] = [cs for cs, _, _ in saved]
        for cs, dof, NP in saved:
            cs.update(dof[ns:], NP, dof[:ns])
        return result

    def _solve(self, composition_sets, conditions):
        from pycalphad.gpu.dispatch import _audit
        _audit('cg_local_solve')
        self._why = None
        if len(composition_sets) == 0:
            return SolverResult(converged=False, x=np.zeros(0), chemical_potentials=np.zeros(0))
        prf, elements, phases = _phase_lookup(composition_sets)
        C = len(elements)
        conds = _as_statevar_conditions(conditions)
        state_variables = list(composition_sets[0].phase_record.state_variables)
        ns = len(state_variables)
        sv_index = {str(sv): i for i, sv in enumerate(state_variables)}
        free_sv = [i for i, sv in enumerate(state_variables) if sv not in conds and str(sv) != 'N']
        sv = np.array(composition_sets[0].dof[:ns], dtype=float)
        for key, val in conds.items():
            if str(key) in sv_index:
                sv[sv_index[str(key)]] = val
        if 'N' in sv_index and v.N not in conds:
            sv[sv_index['N']] = 1.0
        sv0 = sv.copy()
        global _EQUIV
        R, b, mu_fixed, fixed = _condition_rows(conds, elements, np.asarray(prf.molar_masses, dtype=float))
        K = R.shape[0]
        params = np.asarray(prf.param_values, dtype=float)
        for ph in phases.values():
            E, e, pinned = _local_constraints(ph, conds, elements)
            ph.core.set_state(sv, params)
            ph.core.set_local_constraints(E, e)
            ph.core.set_pinned(pinned)
        sets = list(composition_sets)
        ys = [np.array(cs.dof[ns:], dtype=float) for cs in sets]
        ys0 = [y.copy() for y in ys]
        xs0 = [np.array(cs.X, dtype=float) for cs in sets]
        step = _composition_step(sets, conds, elements)
        n = np.array([float(cs.NP) for cs in sets])
        is_fixed = [bool(cs.fixed) for cs in sets]
        active = list(range(len(sets)))
        mu = _initial_potentials(sets, phases, mu_fixed, fixed)
        converged = False
        _why = 'iteration limit'
        _EQUIV = (phases, sv, sv_index['T'])
        last_res = np.inf
        for it in range(self.options.get('max_iterations', MAX_ITERATIONS)):
            for ph in phases.values():
                ph.core.set_statevars(sv)
            RT = R_GAS * sv[sv_index['T']]
            loc = []
            for a in active:
                ph = phases[sets[a].phase_record.phase_name]
                r = ph.core.local_point(mu, ys[a])
                ys[a] = r[1]
                loc.append(r)
            free_cs = [a for a in active if not is_fixed[a]]
            na = len(active)
            nunk = C + len(free_cs) + len(free_sv)
            neq = na + K + int(fixed.sum())
            if nunk != neq:
                _why = f'phase rule: {nunk} unknowns, {neq} equations'
                break   # Gibbs phase rule not satisfied by these composition sets and conditions
            F = np.zeros(neq); J = np.zeros((neq, nunk))
            X = np.array([r[2] for r in loc])
            for k, a in enumerate(active):
                F[k] = loc[k][4]
                J[k, :C] = -X[k]
                for f, t in enumerate(free_sv):
                    J[k, C + len(free_cs) + f] = loc[k][7][t]
            for kk in range(K):
                row = na + kk
                F[row] = sum(n[a] * (R[kk] @ X[k]) for k, a in enumerate(active)) - b[kk]
                J[row, :C] = sum(n[a] * (R[kk] @ loc[k][5]) for k, a in enumerate(active))
                for c, a in enumerate(free_cs):
                    J[row, C + c] = R[kk] @ X[active.index(a)]
                for f, t in enumerate(free_sv):
                    J[row, C + len(free_cs) + f] = sum(n[a] * (R[kk] @ loc[k][6][:, t]) for k, a in enumerate(active))
            row = na + K
            for i in np.flatnonzero(fixed):
                F[row] = mu[i] - mu_fixed[i]; J[row, i] = 1.0; row += 1
            scale = np.concatenate([np.full(na, RT), np.maximum(np.abs(b), 1e-12 * max(abs(b[0]), 1.0)),
                                    np.full(int(fixed.sum()), RT)])
            res = np.max(np.abs(F) / scale) if len(F) else 0.0
            if res < 1e-10 or (res < 1e-7 and res > 0.1 * last_res):
                jumped = _jumped(ys0, ys, xs0, [r[2] for r in loc], active, is_fixed, step)
                xs_now = {a: np.asarray(loc[k][2]) for k, a in enumerate(active)}
                collapsed = _collapsed(xs0, xs_now, ys, sets, active)
                coincident = _coincident(sets, ys, xs_now, active)
                converged = all(r[0] for r in loc) and not coincident and not jumped and not collapsed
                _why = None if converged else ('coincident composition sets' if coincident else
                                               'constitution jump' if jumped else
                                               'composition sets collapsed onto each other' if collapsed else
                                               'local minimization not converged')
                break
            last_res = res
            try:
                step = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(J, -F, rcond=1e-12)[0]
            dmu = step[:C]; dn = step[C:C + len(free_cs)]; dsv = step[C + len(free_cs):]
            sc = 1.0
            if np.max(np.abs(dmu)) > MAX_MU_STEP * RT:
                sc = MAX_MU_STEP * RT / np.max(np.abs(dmu))
            for f, t in enumerate(free_sv):
                if str(state_variables[t]) == 'T' and abs(dsv[f]) * sc > MAX_T_STEP:
                    sc = MAX_T_STEP / abs(dsv[f])
            # a free amount reaching zero: cut the step there and drop that composition set
            drop = None
            for c, a in enumerate(free_cs):
                if n[a] + sc * dn[c] < 0 and dn[c] < 0:
                    alpha = n[a] / -dn[c]
                    if alpha < sc:
                        sc = alpha; drop = a
            mu = mu + sc * dmu
            for c, a in enumerate(free_cs):
                n[a] += sc * dn[c]
            for f, t in enumerate(free_sv):
                sv[t] += sc * dsv[f]
            if drop is not None:
                n[drop] = 0.0
                active.remove(drop)
                last_res = np.inf
                if not active:
                    break
        if converged and _coincident_pair(sets, ys, {a: np.asarray(sets[a].X) if a not in active else np.asarray(loc[active.index(a)][2]) for a in active}, active):
            # two composition sets met (a miscibility gap closing): the Newton system is singular there, so the
            # potentials are those of the single phase at that point, its tangent plane, and free state variables
            # are undetermined (they keep their starting values, the closest point to the critical point)
            sv = sv0.copy()
            for ph in phases.values():
                ph.core.set_statevars(sv)
            for a in active:
                sets[a].update(ys[a], max(n[a], 0.0) if not is_fixed[a] else sets[a].NP, sv)
            mu = _initial_potentials([sets[active[0]]], phases, mu_fixed, fixed)
        # write back
        for a, cs in enumerate(sets):
            cs.update(ys[a], max(n[a], 0.0) if not is_fixed[a] else cs.NP, sv)
        if self.remove_metastable:
            for a in reversed(range(len(sets))):
                if (a not in active or sets[a].NP <= 0.0) and not is_fixed[a]:
                    del composition_sets[a]
        if len(composition_sets) == 0:
            return SolverResult(converged=False, x=np.zeros(0), chemical_potentials=np.array(mu))
        x = composition_sets[0].dof
        for cs in composition_sets[1:]:
            x = np.r_[x, cs.dof[ns:]]
        x = np.r_[x, [cs.NP for cs in composition_sets]]
        self._why = None if converged else _why
        if _DEBUG and not converged:
            print(f'CGSolver failed ({_why}): {[(c.phase_record.phase_name, np.round(c.X, 4).tolist(), c.NP) for c in sets]}, '
                  f'conditions {conds}, last residual {last_res:.3g}')
        if self.verbose:
            print(f'CGSolver: converged={converged}, chemical potentials {np.round(mu, 5).tolist()}, {composition_sets}')
        return SolverResult(converged=converged, x=x, chemical_potentials=np.array(mu))


def global_min_candidate(point, system_info, tol=1e-5):
    """(composition set, driving force) of the phase with the largest driving force (J/mol, > tol) at the point's
    state variables and chemical potentials, or None: local minimizations of every phase's driving force from its
    start configurations, the point's composition sets of that phase, and a start at the overall composition."""
    from pycalphad.core.composition_set import CompositionSet
    from pycalphad.gpu.dispatch import _audit
    _audit('cg_global_min_check')
    prf = system_info['phase_records']
    elements = [str(e) for e in prf.nonvacant_elements]
    phases = _phases_for(prf, elements)
    stable = point.stable_composition_sets
    if not stable:
        return None
    ns = len(prf.state_variables)
    sv = np.array(stable[0].dof[:ns], dtype=float)
    params = np.asarray(prf.param_values, dtype=float)
    mu = np.asarray(point.chemical_potentials, dtype=float).reshape(-1)
    # overall composition from the stable sets
    z = np.zeros(len(elements)); total = 0.0
    for cs in stable:
        z += cs.NP * np.asarray(cs.X); total += cs.NP
    z = z / total if total > 0 else np.full(len(elements), 1 / len(elements))
    rng = np.random.default_rng(0)
    best = None
    stable_x = [np.asarray(cs.X, dtype=float) for cs in stable]
    for ph in phases:
        ph.core.set_state(sv, params)
        ph.core.set_local_constraints(np.zeros((0, ph.D)), np.zeros(0))
        ph.core.set_pinned(np.zeros(ph.D, dtype=np.int32))
        starts = [np.array(cs.dof[ns:], dtype=float) for cs in stable if cs.phase_record.phase_name == ph.name]
        starts += ph.seeds(z, elements, rng, state=(tuple(sv), tuple(params), ()))
        for y0 in starts:
            tpd, y, x, G, conv = ph.core.py_price(mu, y0)
            if not np.isfinite(G):
                continue
            dG = -tpd
            near = min((np.max(np.abs(np.asarray(x) - xs)) for xs in stable_x), default=np.inf)
            if (near < TIE_X_NEAR and dG < TIE_DG) or (near < TIE_X and dG < TIE_DG_FAR):
                # a state already present in another description of equal energy (e.g. an ordered model in its
                # disordered state and the disordered model), not a new phase
                continue
            if best is None or dG > best[1]:
                best = (ph.name, dG, y)
    if best is None or best[1] < tol:
        return None
    cs = CompositionSet(prf[best[0]])
    cs.update(np.asarray(best[2], dtype=np.float64), 1.0, sv)
    return cs, best[1]
