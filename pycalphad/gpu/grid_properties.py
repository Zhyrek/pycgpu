"""
Vectorized evaluation of Workspace properties over a whole condition grid.

Workspace.get evaluates properties one condition at a time: it builds the
composition sets of every condition and calls each property's
compute_property. For the common properties (state variables, chemical
potentials, mole fractions, phase fractions, site fractions and Model
properties) this module computes the same values for all conditions at once
from the solved arrays (Phase, NP, Y, MU). Phase-record evaluations use the
reference phase records' own 2-D functions (prop_2d, mass_obj_2d), which run
the same compiled code row by row as the per-condition path, so the results
are bit-identical to it; summation orders follow compute_property.

The semantics mirrored here (pycalphad/variables.py, computed_property.py):

- composition sets are the non-empty vertices of a condition, in vertex order;
- StateVariable-based properties (X, NP, Y) skip composition sets with zero
  amount (StateVariable.filtered) and count miscibility-gap copies ('#2')
  among the remaining ones;
- ModelComputedProperty counts copies among all composition sets of the
  phase, and system-level values are NP-weighted sums over all of them;
- every property is NaN for a condition without composition sets.
"""
import numpy as np

from pycalphad import variables as v
from pycalphad.property_framework.computed_property import ModelComputedProperty

__all__ = ['GridState', 'supported', 'evaluate']

_STATE_VARIABLE_TYPES = (v.TemperatureType, v.PressureType, v.SystemMolesType)


class GridState:
    """The solved Workspace arrays, flattened over conditions in the order of
    Workspace.enumerate_composition_sets (C order of eq.GM)."""

    def __init__(self, wks):
        eq = wks.eq
        self.wks = wks
        self.prf = wks.phase_record_factory
        self.n = int(np.asarray(eq.GM).size)
        phase = np.asarray(eq.Phase)
        self.n_vertex = phase.shape[-1]
        self.phase = phase.reshape(self.n, self.n_vertex).astype(str)
        self.np = np.asarray(eq.NP, dtype=np.float64).reshape(self.n, self.n_vertex)
        y = np.asarray(eq.Y, dtype=np.float64)
        self.y = y.reshape(self.n, self.n_vertex, y.shape[-1])
        mu = np.asarray(eq.MU, dtype=np.float64)
        self.mu = mu.reshape(self.n, mu.shape[-1])
        self.present = (self.phase != '') & (self.phase != '_FAKE_')
        self.has_compsets = self.present.any(axis=1)
        conds_keys = [str(k) for k in eq.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
        mesh = np.meshgrid(*[np.asarray(eq.coords[k], dtype=np.float64) for k in conds_keys], indexing='ij')
        self.condition_values = {k: m.reshape(-1) for k, m in zip(conds_keys, mesh)}
        state_variables = list(self.prf.values())[0].state_variables
        self.state_variables = [str(sv) for sv in state_variables]
        self.statevar_values = np.stack([self.condition_values[k] for k in self.state_variables], axis=1)
        self.phase_names = set(str(ph) for ph in self.prf.keys())
        self._dof = {}
        self._x = {}
        self._prop = {}

    # -- composition sets of one phase -------------------------------------
    def rows(self, phase_name):
        """(condition, vertex) index arrays of the composition sets of a phase."""
        return np.nonzero(self.present & (self.phase == phase_name))

    def dof(self, phase_name):
        if phase_name not in self._dof:
            cond, vert = self.rows(phase_name)
            pdof = self.prf[phase_name].phase_dof
            self._dof[phase_name] = np.ascontiguousarray(
                np.concatenate([self.statevar_values[cond], self.y[cond, vert, :pdof]], axis=1))
        return self._dof[phase_name]

    def mole_fractions(self, phase_name):
        """compset.X of every composition set of the phase: (rows, components)."""
        if phase_name not in self._x:
            pr = self.prf[phase_name]
            dof = self.dof(phase_name)
            x = np.zeros((dof.shape[0], len(pr.nonvacant_elements)), order='F')
            for el_idx in range(x.shape[1]):
                pr.mass_obj_2d(x[:, el_idx], dof, el_idx)
            self._x[phase_name] = x
        return self._x[phase_name]

    def model_property(self, phase_name, attr):
        key = (phase_name, attr)
        if key not in self._prop:
            dof = self.dof(phase_name)
            out = np.zeros(dof.shape[0])
            self.prf[phase_name].prop_2d(out, dof, attr.encode('utf-8'))
            self._prop[key] = out
        return self._prop[key]

    # -- helpers -------------------------------------------------------------
    def occurrence(self, phase_name, nonzero_only):
        """Per (condition, vertex): 1-based index of the composition set among
        those of the same phase in its condition (0 where not counted)."""
        mask = self.present & (self.phase == phase_name)
        if nonzero_only:
            mask &= self.np != 0
        return np.where(mask, np.cumsum(mask, axis=1), 0)

    def nan_result(self):
        return np.full(self.n, np.nan)


def _phase_and_multiplicity(prop):
    if prop.phase_name is None:
        return None, None
    tokens = prop.phase_name.split('#')
    return tokens[0], (int(tokens[1]) if len(tokens) > 1 else 1)


def supported(prop):
    """True for property types this module evaluates."""
    if type(prop) in _STATE_VARIABLE_TYPES:
        return True
    if type(prop) is v.ChemicalPotential:
        return True
    if type(prop) in (v.MoleFraction, v.PhaseFraction):
        return True
    if type(prop) is v.SiteFraction:
        return prop.phase_name is not None
    if type(prop) is ModelComputedProperty:
        return True
    return False


def _filtered_sum(contrib, present):
    """StateVariable.compute_property accumulation over composition sets in
    vertex order: `if isnan(result): result = 0; result += value`."""
    result = np.full(contrib.shape[0], np.nan)
    for col in range(contrib.shape[1]):
        m = present[:, col]
        result[m] = np.where(np.isnan(result[m]), 0.0, result[m]) + contrib[m, col]
    return result


def _np_sum(contrib, present):
    """np.sum over each condition's terms in vertex order (NaN propagates;
    NumPy sums fewer than 8 terms sequentially from 0)."""
    result = np.full(contrib.shape[0], np.nan)
    count = present.sum(axis=1)
    small = (count > 0) & (count < 8)
    acc = np.zeros(contrib.shape[0])
    for col in range(contrib.shape[1]):
        m = present[:, col] & small
        acc[m] = acc[m] + contrib[m, col]
    result[small] = acc[small]
    for c in np.nonzero(count >= 8)[0]:
        result[c] = np.sum(contrib[c, present[c]])
    return result


def evaluate(state, prop):
    """Values of `prop` for every condition (implementation units)."""
    n = state.n
    if type(prop) in _STATE_VARIABLE_TYPES:
        result = state.nan_result()
        idx = state.state_variables.index(str(prop))
        result[state.has_compsets] = state.statevar_values[state.has_compsets, idx]
        return result

    if type(prop) is v.ChemicalPotential:
        nonvacant = list(list(state.prf.values())[0].nonvacant_elements)
        values = np.zeros(n)
        for el, multiplicity in prop.species.constituents.items():
            values = values + multiplicity * state.mu[:, nonvacant.index(el)]
        result = state.nan_result()
        result[state.has_compsets] = values[state.has_compsets]
        return result

    phase_name, multiplicity = _phase_and_multiplicity(prop)
    phases = [phase_name] if phase_name is not None else [str(ph) for ph in np.unique(state.phase[state.present])]
    phases = [ph for ph in phases if ph in state.phase_names]

    if type(prop) is ModelComputedProperty:
        if phase_name is None:
            contrib = np.zeros((n, state.n_vertex))
            present = np.zeros((n, state.n_vertex), dtype=bool)
            for ph in phases:
                cond, vert = state.rows(ph)
                contrib[cond, vert] = state.np[cond, vert] * state.model_property(ph, prop.model_attr_name)
                present[cond, vert] = True
            return _np_sum(contrib, present)
        result = state.nan_result()
        if phases:
            cond, vert = state.rows(phase_name)
            selected = state.occurrence(phase_name, nonzero_only=False)[cond, vert] == multiplicity
            result[cond[selected]] = state.model_property(phase_name, prop.model_attr_name)[selected]
        return result

    # StateVariable-based (X, NP, Y): StateVariable.filtered composition sets
    contrib = np.zeros((n, state.n_vertex))
    present = np.zeros((n, state.n_vertex), dtype=bool)
    for ph in phases:
        cond, vert = state.rows(ph)
        keep = state.np[cond, vert] != 0
        if phase_name is not None:
            keep &= state.occurrence(ph, nonzero_only=True)[cond, vert] == multiplicity
        if type(prop) is v.PhaseFraction:
            values = state.np[cond, vert]
        elif type(prop) is v.MoleFraction:
            el_idx = list(state.prf[ph].nonvacant_elements).index(str(prop.species))
            x = state.mole_fractions(ph)[:, el_idx]
            values = state.np[cond, vert] * x if phase_name is None else x
        else:  # SiteFraction
            sitefrac_idx = list(state.prf[ph].variables).index(prop._self_without_suffix)
            values = state.dof(ph)[:, len(state.state_variables) + sitefrac_idx]
        contrib[cond[keep], vert[keep]] = values[keep]
        present[cond[keep], vert[keep]] = True
    return _filtered_sum(contrib, present)
