"""
Temporary fixes to the reference implementation, applied at import.

MONKEY PATCH: remove each fix here once the corresponding upstream pycalphad
change is merged and this fork is rebased onto it. The upstream sources are
left untouched so that the rebase stays conflict-free.

Jansson derivative phase-amount units
-------------------------------------
``state_variable_differential``, ``fixed_component_differential`` and
``chemical_potential_differential`` return phase-amount deltas in the
solver's convention (moles of formula units), but every consumer of
``JanssonDerivativeDeltas.delta_phase_amounts`` combines them with
per-mole-of-atoms quantities (``CompositionSet.NP`` and the Model
properties). Derivatives are therefore wrong whenever a stable phase has
more or less than one atom per formula unit, or a composition-dependent
number of them (vacancies mixing with atoms), e.g. ``HM.T`` of a two-phase
FCC_A1 + CU2MG (Laves) Cu-Mg equilibrium, or of B2 Al-Ni. The fix converts the
deltas to moles of atoms where they are produced:

    NP = phase_amt * M(y)        (M: moles of atoms per formula unit)
    dNP = d(phase_amt) * M + phase_amt * (dM/dy . dy)
"""
import functools

import numpy as np

_applied = False


def delta_phase_amounts_in_moles_of_atoms(state, delta_phase_amounts, delta_sitefracs):
    """
    Convert phase amount deltas from moles of formula units to moles of atoms.

    The ``*_differential`` functions solve for changes in ``SystemState.phase_amt`` (moles of formula units),
    but Jansson derivatives treat ``delta_phase_amounts`` as changes in ``CompositionSet.NP`` (moles of atoms).
    With ``M`` the moles of atoms per formula unit, ``NP = phase_amt * M``, so
    ``dNP = d(phase_amt) * M + phase_amt * dM``.
    """
    result = np.array(delta_phase_amounts, dtype=float)
    for idx, compset in enumerate(state.compsets):
        phase_record = compset.phase_record
        num_statevars = len(phase_record.state_variables)
        dof = np.ascontiguousarray(compset.dof, dtype=float)
        moles, moles_grad = np.zeros(1), np.zeros(num_statevars + phase_record.phase_dof)
        atoms_per_formula_unit, delta_atoms_per_formula_unit = 0.0, 0.0
        for comp_idx in range(len(state.chemical_potentials)):
            phase_record.formulamole_obj(moles, dof, comp_idx)
            phase_record.formulamole_grad(moles_grad, dof, comp_idx)
            atoms_per_formula_unit += moles[0]
            # Site ratios do not depend on state variables, so only the site fraction terms contribute
            delta_atoms_per_formula_unit += np.dot(moles_grad[num_statevars:], delta_sitefracs[idx])
        formula_units = compset.NP / atoms_per_formula_unit
        result[idx] = result[idx] * atoms_per_formula_unit + formula_units * delta_atoms_per_formula_unit
    return result


def _patch_jansson_deltas(cls):
    original = cls.jansson_deltas

    @functools.wraps(original)
    def jansson_deltas(self, spec, state):
        deltas = original(self, spec, state)
        deltas.delta_phase_amounts = delta_phase_amounts_in_moles_of_atoms(
            state, deltas.delta_phase_amounts, deltas.delta_sitefracs)
        return deltas

    jansson_deltas._pycalphad_upstream_patch = True
    cls.jansson_deltas = jansson_deltas


def apply():
    global _applied
    if _applied:
        return
    import pycalphad.variables as v
    # The three classes whose jansson_deltas call into the minimizer's
    # *_differential functions (StateVariable covers T and P).
    for cls in (v.StateVariable, v.MoleFraction, v.ChemicalPotential):
        if not getattr(cls.__dict__.get('jansson_deltas'), '_pycalphad_upstream_patch', False):
            _patch_jansson_deltas(cls)
    _applied = True
