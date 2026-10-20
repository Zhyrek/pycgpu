"""
Routing between the reference solver and the compiled backends.

:func:`solve_workspace`, called from ``Workspace.recompute``: when a
compiled backend is selected (see :mod:`pycalphad.backend`) and the problem
is one it supports exactly, the equilibrium is computed by the compiled
pipeline; otherwise (or on a declared capability limit) the caller runs the
reference implementation.

:func:`workspace_properties`, called from ``Workspace.get_dict``: for a
Workspace solved by a compiled backend, evaluates the common properties
(state variables, chemical potentials, X, NP, Y, Model properties; see
grid_properties) and Jansson derivatives of Model properties for the whole
condition grid at once; the caller evaluates the rest per condition.

Set ``PYCGPU_COUNT_DISPATCH=<file>`` to append one line per routing
decision to a file (used to audit test-suite coverage of the backends).
"""
import logging
import os

import numpy as np

from pycalphad import variables as v

__all__ = ['problem_supported', 'solve_workspace', 'workspace_properties',
           'equilibrium_outputs']

_log = logging.getLogger(__name__)


def _audit(line):
    path = os.environ.get('PYCGPU_COUNT_DISPATCH')
    if path:
        test = os.environ.get('PYTEST_CURRENT_TEST', '')
        with open(path, 'a') as f:
            f.write(f'{line} <{test}>\n')


def problem_supported(conditions, parameters, dbf=None, comps=None, phases=None):
    """(True, None) if the compiled backends solve this problem exactly,
    else (False, reason).

    Supported: fully determined N=1/P/T problems with X, W, MU,
    linear-combination and scalar phase-local composition conditions, and
    scalar parameter overrides.
    """
    if parameters:
        try:
            for pv in dict(parameters).values():
                if np.asarray(pv, dtype=np.float64).size != 1:
                    return False, 'vectorized parameter sweep'
        except Exception:
            return False, 'parameter extraction failed'
    try:
        global_x = [c for c in conditions
                    if isinstance(c, v.MoleFraction) and getattr(c, 'phase_name', None) is None]
        global_w = [c for c in conditions
                    if isinstance(c, v.MassFraction) and getattr(c, 'phase_name', None) is None]
        mu = [c for c in conditions if isinstance(c, v.ChemicalPotential)]
        lincomb = [c for c in conditions if str(c).startswith('LinComb_')]
        statevars = [c for c in conditions if c in (v.N, v.P, v.T)]
        local = [c for c in conditions
                 if isinstance(c, (v.MoleFraction, v.SiteFraction))
                 and getattr(c, 'phase_name', None) is not None]
        n_composition = len(global_x) + len(global_w) + len(mu) + len(lincomb)
        # Invalid inputs must reach the reference path, which raises the
        # canonical validation errors (missing components, no active phases,
        # under/overdetermined composition conditions).
        if dbf is not None and comps is not None:
            comp_names = {str(c).upper() for c in comps} - {'VA', '/-'}
            if not comp_names <= {str(e).upper() for e in dbf.elements}:
                return False, 'component not in database'
            if n_composition != len(comp_names) - 1:
                return False, 'not a fully determined composition set'
            if phases is not None:
                from pycalphad.core.utils import filter_phases, unpack_species
                candidates = [p for p in phases if p in dbf.phases]
                if not filter_phases(dbf, unpack_species(dbf, list(comps)), candidates):
                    return False, 'no active phases'
        if n_composition + len(statevars) + len(local) != len(conditions):
            return False, 'unsupported condition type present'
        for cond, value in conditions.items():
            if getattr(cond, 'phase_name', None) is not None:
                if not isinstance(cond, (v.MoleFraction, v.SiteFraction)):
                    return False, 'unsupported phase-local condition class'
                if np.asarray(value, dtype=object).size != 1:
                    return False, 'array-valued phase-local condition'
            elif cond == v.N:
                if np.any(np.atleast_1d(np.asarray(value, dtype=object)).astype(float) != 1.0):
                    return False, 'N != 1'
    except Exception:
        return False, 'condition inspection failed'
    return True, None


def solve_workspace(wks):
    """Compute `wks`'s equilibrium on the selected compiled backend.

    Returns the properties LightDataset, or None when the reference
    implementation should be used instead.
    """
    from pycalphad.backend import _current, AcceleratedCapabilityError
    from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    from pycalphad.core.solver import Solver
    from pycalphad.property_framework.units import as_quantity

    wks._solved_by_backend = False
    backend_name, options = _current()
    if backend_name == 'default':
        return None
    conds = {key: as_quantity(key, value).to(key.implementation_units).magnitude
             for key, value in wks.conditions.items()}
    if backend_name == 'cg':
        return _solve_workspace_cg(wks, conds)
    if wks.calc_opts and set(wks.calc_opts) - {'pdens'}:
        ok, reason = False, 'unsupported calc_opts'
    elif type(wks.solver) is not Solver:
        ok, reason = False, 'non-default solver'
    elif type(wks.phase_record_factory) is not PhaseRecordFactory:
        ok, reason = False, 'non-standard phase records'
    else:
        ok, reason = problem_supported(conds, wks.parameters.unwrap(), dbf=wks.database,
                                       comps=[str(c) for c in wks.components],
                                       phases=list(wks.phases))
    if not ok:
        _audit(f'gate_fallback: {reason}')
        _log.debug("reference solver used (%s)", reason)
        return None
    _audit('gate_pass')
    try:
        wks.phase_record_factory.update_parameters(wks.parameters.unwrap())
        from pycalphad.gpu.gpu_equilibrium import calculate_equilibrium_gpu
        result = calculate_equilibrium_gpu(wks, to_xarray=False)
        wks._solved_by_backend = True
        return result
    except Exception as err:
        fall_back = (isinstance(err, AcceleratedCapabilityError)
                     or bool(os.environ.get('PYCGPU_FALLBACK')))
        _audit(f"{'runtime_fallback' if fall_back else 'runtime_error'}: {str(err)[:100]}")
        if not fall_back:
            raise RuntimeError(
                f"the {backend_name!r} backend failed (chained below). Use "
                "set_backend('default') for the reference solver, or set "
                "PYCGPU_FALLBACK=1 to fall back automatically on such errors.") from err
        _log.debug("reference solver used (capability limit: %r)", err)
        return None


def _solve_workspace_cg(wks, conds):
    """The column-generation backend. Post-processing (properties, derivatives) stays on the reference code paths,
    which evaluate them from the equilibrium state, so `_solved_by_backend` is left False."""
    from pycalphad.backend import AcceleratedCapabilityError
    from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    from pycalphad.core.solver import Solver
    if wks.calc_opts and set(wks.calc_opts) - {'pdens'}:
        ok, reason = False, 'unsupported calc_opts'
    elif type(wks.solver) is not Solver:
        ok, reason = False, 'non-default solver'
    elif type(wks.phase_record_factory) is not PhaseRecordFactory:
        ok, reason = False, 'non-standard phase records'
    else:
        ok, reason = problem_supported(conds, wks.parameters.unwrap(), dbf=wks.database,
                                       comps=[str(c) for c in wks.components],
                                       phases=list(wks.phases))
    if not ok:
        _audit(f'cg_gate_fallback: {reason}')
        return None
    try:
        wks.phase_record_factory.update_parameters(wks.parameters.unwrap())
        from pycalphad.cg import solve_workspace as cg_solve
        result = cg_solve(wks)
    except AcceleratedCapabilityError as err:
        _audit(f'cg_runtime_fallback: {str(err)[:100]}')
        return None
    _audit('cg_solved')
    return result


def _batched_jansson_supported(wks, prop):
    """None if `prop` is a Jansson derivative the compiled pipeline computes
    for a whole grid, else the reason it is not."""
    from pycalphad.property_framework.computed_property import (JanssonDerivative,
                                                                ModelComputedProperty)
    if not isinstance(prop, JanssonDerivative):
        return 'not a Jansson derivative'
    numerator, denominator = prop.numerator, prop.denominator
    if type(numerator) is not ModelComputedProperty or numerator.phase_name is not None:
        return 'numerator is not a system-level Model property'
    if denominator not in wks.conditions:
        return 'denominator is not a condition'
    if denominator == v.T:
        pass
    elif not (type(denominator) is v.MoleFraction and denominator.phase_name is None):
        # P needs the property gradients' P column, which the generated
        # property modules do not emit yet.
        return f'denominator {denominator} not supported'
    for cond in wks.conditions:
        if cond in (v.N, v.P, v.T):
            continue
        if type(cond) is v.MoleFraction and cond.phase_name is None:
            continue
        # MU, W, linear-combination and phase-local conditions: the batched
        # deltas are only verified against the reference for N/P/T/X.
        return f'condition {cond} not supported'
    return None


def _workspace_cur_conds(wks):
    """Workspace.get_dict's per-condition `cur_conds`, in its condition order."""
    from collections import OrderedDict
    eq = wks.eq
    str_conds_keys = [str(k) for k in eq.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
    conds_keys = [None] * len(str_conds_keys)
    for k in wks.conditions.keys():
        conds_keys[str_conds_keys.index(str(k))] = k
    return [OrderedDict(zip(conds_keys, [np.asarray(eq.coords[b][a], dtype=np.float64)
                                         for a, b in zip(index, str_conds_keys)]))
            for index in np.ndindex(eq.GM.shape)]


def _metaproperty_of(prop):
    """('isolated' | 'dormant', meta-property object) for IsolatedPhase and
    DormantPhase properties, else None."""
    for kind in ('isolated', 'dormant'):
        meta = getattr(prop, kind + '_phase', None)
        if meta is not None:
            return kind, meta
    return None


def _metaproperty_chain_properties(wks, props, display_units):
    """IsolatedPhase and DormantPhase properties for the whole grid: one
    compiled chain per meta-property object, replaying Workspace.get_dict's
    evaluation order (conditions in order; within a condition, that object's
    properties in order, one solve or relaxation each) so every step starts
    where the reference would."""
    from pycalphad.core.composition_set import CompositionSet
    from pycalphad.property_framework.units import Q_
    from pycalphad.gpu.metaproperty_chains import chain_supported, dormant_phase_steps, isolated_phase_steps
    from pycalphad.core.solver import Solver
    groups = {}
    for prop in props:
        tagged = _metaproperty_of(prop)
        if tagged is not None:
            groups.setdefault(id(tagged[1]), (tagged[0], tagged[1], []))[2].append(prop)
    if not groups:
        return {}
    cur_conds = _workspace_cur_conds(wks)
    # chains of different objects sharing one composition set would interleave
    compset_users = {}
    for _, meta, _ in groups.values():
        compset_users[id(meta._compset)] = compset_users.get(id(meta._compset), 0) + 1
    out = {}
    for kind, meta, group in groups.values():
        compset = meta._compset
        reason = chain_supported(wks, compset)
        if reason is None and compset_users[id(compset)] > 1:
            reason = 'composition set shared by several meta-properties'
        if reason is None and kind == 'isolated' and type(meta.solver) is not Solver:
            reason = 'non-default solver'
        if reason is None and kind == 'dormant':
            phase = np.asarray(wks.eq.Phase).reshape(len(cur_conds), -1)
            if not np.all(((phase != '') & (phase != '_FAKE_')).any(axis=1)):
                reason = 'conditions without composition sets'
        convert = [display_units and str(p.implementation_units) != str(p.display_units) for p in group]
        if reason is None:
            try:
                for p, needed in zip(group, convert):
                    if needed:
                        Q_(1.0, p.implementation_units).to(p.display_units)
            except Exception:
                reason = 'unit conversion needs composition sets'
        if reason is not None:
            for p in group:
                _audit(f'property_reference: {p} ({reason})')
            continue
        start = (np.array(compset.dof), compset.NP, compset.fixed)
        step_conditions = [c for c in cur_conds for _ in group]
        try:
            if kind == 'isolated':
                steps = isolated_phase_steps(wks, compset, step_conditions,
                                             getattr(meta, '_initial_compset', None))
            else:
                mu = np.asarray(wks.eq.MU, dtype=np.float64).reshape(len(cur_conds), -1)
                steps = dormant_phase_steps(wks, compset, step_conditions,
                                            np.repeat(mu, len(group), axis=0), meta.max_iterations)
            pr = compset.phase_record
            eq_mu = np.asarray(wks.eq.MU, dtype=np.float64).reshape(len(cur_conds), -1)
            state_variables = [str(sv) for sv in pr.state_variables]
            values = [np.full((len(cur_conds),) + p.shape, np.nan) for p in group]
            for c, conds in enumerate(cur_conds):
                by_name = {str(k): float(val) for k, val in conds.items()}
                statevars = np.array([by_name[sv] for sv in state_variables], dtype=np.float64)
                for j, p in enumerate(group):
                    step = steps[c * len(group) + j]
                    if kind == 'isolated' and not step[3]:
                        # unconverged: IsolatedPhase evaluates without a composition set
                        values[j][c, ...] = p.inner_property.compute_property([], conds, eq_mu[c])
                        continue
                    # IsolatedPhase evaluates with the step's chemical
                    # potentials, DormantPhase with the equilibrium's
                    chemical_potentials = step[2] if kind == 'isolated' else mu[c]
                    step_compset = CompositionSet(pr)
                    step_compset.update(step[0], step[1], statevars)
                    values[j][c, ...] = p.inner_property.compute_property([step_compset], conds,
                                                                          chemical_potentials)
        except Exception as err:
            nsv = len(compset.phase_record.state_variables)
            compset.update(start[0][nsv:nsv + compset.phase_record.phase_dof], start[1], start[0][:nsv])
            compset.fixed = start[2]
            for p in group:
                _audit(f'property_reference: {p} ({type(err).__name__}: {str(err)[:80]})')
            _log.debug("%s phase properties evaluated per condition (%r)", kind, err)
            continue
        for p, vals, needed in zip(group, values, convert):
            if needed:
                vals = Q_(vals, p.implementation_units).to(p.display_units).magnitude
            _audit(f'property_compiled: {p}')
            out[p] = vals
    return out


# Below this many conditions the per-condition evaluation is as fast as the
# vectorized one's fixed setup cost.
_MIN_VECTORIZED_CONDITIONS = int(os.environ.get('PYCGPU_VECTORIZED_MIN_CONDITIONS', 50))


def workspace_properties(wks, props, display_units=True):
    """Evaluate the properties in `props` that the compiled backend supports,
    for every condition of `wks` at once.

    Returns {prop: array of shape (n_conditions,) + prop.shape}, in display
    units (implementation units with ``display_units=False``); properties
    that are absent are evaluated per condition by the caller.
    """
    from pycalphad.backend import _current
    from pycalphad.property_framework.units import Q_
    if not getattr(wks, '_solved_by_backend', False) or _current()[0] == 'default':
        return {}
    from pycalphad.gpu import grid_properties
    out = _metaproperty_chain_properties(wks, props, display_units)
    grid_state = None
    for prop in props:
        if _metaproperty_of(prop) is not None:
            continue  # handled above (or left to the per-condition path)
        if grid_properties.supported(prop) and wks.eq.GM.size >= _MIN_VECTORIZED_CONDITIONS:
            # (no de-duplication: equal properties with different display
            # units overwrite each other in order, as in Workspace.get_dict)
            # state variables, chemical potentials, X, NP, Y and Model
            # properties: vectorized over the grid, bit-identical to the
            # per-condition evaluation
            try:
                if grid_state is None:
                    grid_state = grid_properties.GridState(wks)
                values = grid_properties.evaluate(grid_state, prop)
                if display_units and str(prop.implementation_units) != str(prop.display_units):
                    values = Q_(values, prop.implementation_units).to(prop.display_units).magnitude
            except Exception as err:
                _audit(f'property_reference: {prop} ({type(err).__name__}: {str(err)[:80]})')
                _log.debug("%s evaluated per condition (%r)", prop, err)
                continue
            _audit(f'property_vectorized: {prop}')
            out[prop] = values.reshape((-1,) + prop.shape)
            continue
        if prop in out:
            continue
        reason = _batched_jansson_supported(wks, prop)
        if reason is not None:
            _audit(f'property_reference: {prop} ({reason})')
            continue
        try:
            from pycalphad.gpu.jansson_derivatives import workspace_jansson_derivative
            values = workspace_jansson_derivative(wks, prop.numerator.model_attr_name,
                                                  prop.denominator)['values']
            values = np.asarray(values, dtype=np.float64).reshape((-1,) + prop.shape)
            if values.shape[0] != wks.eq.GM.size:
                raise ValueError('condition grid mismatch')
            if display_units and str(prop.implementation_units) != str(prop.display_units):
                values = Q_(values, prop.implementation_units).to(prop.display_units).magnitude
        except Exception as err:
            # e.g. a property the generated property modules do not provide,
            # or units that need per-phase molar masses
            _audit(f'property_reference: {prop} ({type(err).__name__}: {str(err)[:80]})')
            _log.debug("%s evaluated per condition (%r)", prop, err)
            continue
        _audit(f'property_compiled: {prop}')
        out[prop] = values
    return out


def equilibrium_outputs(properties, outputs, wks):
    """Evaluate plain Model-property outputs (HM, SM, CPM, ...) at the
    converged states with the generated property functions, adding them to
    `properties`. Returns the outputs it could not serve (the caller computes
    those with the reference implementation)."""
    from pycalphad.gpu.gpu_equilibrium import _compute_equilibrium_output_properties
    from pycalphad.backend import AcceleratedCapabilityError
    from pycalphad.core.light_dataset import LightDataset
    from pycalphad.property_framework import as_property
    plain = [o for o in outputs if isinstance(o, str) and o.isidentifier()]
    rest = [o for o in outputs if o not in plain]
    if rest:
        # Jansson derivatives (e.g. 'HM.T') for the whole grid at once
        props = {o: as_property(o) for o in rest}
        batched = workspace_properties(wks, list(props.values()), display_units=False)
        conds_keys = [str(k) for k in properties.coords.keys()
                      if k not in ('vertex', 'component', 'internal_dof')]
        for o, prop in props.items():
            if prop in batched:
                values = batched[prop].reshape(properties.GM.shape)
                properties.merge(LightDataset({str(prop): (conds_keys, values)},
                                              coords=properties.coords),
                                 inplace=True, compat='equals')
        rest = [o for o, prop in props.items() if prop not in batched]
    done = []
    for out in plain:
        try:
            _compute_equilibrium_output_properties(properties, [out], wks)
            done.append(out)
        except (AcceleratedCapabilityError, ValueError, AttributeError) as err:
            _log.debug("output %s uses the reference implementation (%r)", out, err)
    return rest + [o for o in plain if o not in done]
