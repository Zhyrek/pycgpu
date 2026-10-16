"""
A pycalphad Workspace whose equilibria come from column generation, and a drop-in ``equilibrium`` built on it.

pycalphad's own Workspace finds equilibria by sampling every phase on a grid, taking the convex hull of the samples
as starting points and refining them (``Workspace.recompute``). CGWorkspace overrides only ``recompute``: the
returned dataset has the same layout, so everything downstream of it (``wks.eq``, ``wks.get(...)``, properties,
Jansson derivatives, reference states) is pycalphad's unchanged.
"""
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime
from typing import Optional

import numpy as np
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.light_dataset import LightDataset
from pycalphad.core.solver import Solver
from pycalphad.core.workspace import Workspace, TypedField
from pycalphad.property_framework import as_property

from pycalphad_cg.solve import solve_workspace, CGCapabilityError

__all__ = ['CGWorkspace', 'equilibrium']


class CGWorkspace(Workspace):
    """
    Workspace computing equilibria by column generation (see the package README).

    Additional attributes
    ---------------------
    cg_threads : int or None
        Worker threads for the conditions (default: the PYCALPHAD_CG_THREADS environment variable, else all cores).
    cg_fallback : bool
        If True (default), problems column generation does not handle (conditions or models it does not support,
        calc_opts other than pdens, a custom solver object or phase records) are solved by pycalphad's own
        ``Workspace.recompute``, with a warning; if False they raise CGCapabilityError.
    """
    cg_threads: Optional[int] = TypedField(lambda _: None)
    cg_fallback: bool = TypedField(lambda _: True)

    def recompute(self):
        self._validate_inputs()
        _check_degrees_of_freedom(self.conditions, self.phase_record_factory)
        try:
            self._check_supported()
            self.phase_record_factory.update_parameters(self.parameters.unwrap())
            return solve_workspace(self, threads=self.cg_threads)
        except CGCapabilityError as err:
            if not self.cg_fallback:
                raise
            warnings.warn(f'column generation does not handle this problem ({err}); using pycalphad\'s solver')
            return super().recompute()

    def _validate_inputs(self):
        """The input checks pycalphad makes in ``calculate`` (which column generation does not use), with the same
        errors."""
        from pycalphad.core.conditions import ConditionError
        from pycalphad.core.utils import filter_phases, unpack_species
        comps = sorted(unpack_species(self.database, self.components))
        possible = filter_phases(self.database, comps)
        if len(possible) == 0:
            raise ConditionError(f'There are no phases in the Database that can be active with components {comps}')
        if len(filter_phases(self.database, comps, self.phases)) == 0:
            raise ConditionError(f'None of the passed phases ({self.phases}) are active. List of possible phases: {possible}.')
        conds = {str(key): value for key, value in self.conditions.items()}
        if 'N' in conds and np.any(np.asarray(conds['N']) != 1):
            raise ConditionError(f'N!=1 is not yet supported, got N={conds["N"]}')
        missing = sorted({str(sv) for sv in self.phase_record_factory.state_variables} - set(conds) - {'N'})
        if missing:
            raise ConditionError(f'The following state variable(s) are required by the Model(s) for the active phases '
                                 f'but were not specified: {missing}.')

    def _check_supported(self):
        if self.calc_opts and set(self.calc_opts) - {'pdens'}:
            raise CGCapabilityError(f'calc_opts {sorted(set(self.calc_opts) - {"pdens"})}')
        if type(self.solver) is not Solver:
            raise CGCapabilityError('a custom solver object (pycalphad refines its own starting points with it)')
        if type(self.phase_record_factory) is not PhaseRecordFactory:
            raise CGCapabilityError('custom phase records')


def _check_degrees_of_freedom(conditions, phase_record_factory):
    """The check pycalphad makes in ``starting_point`` (which column generation does not use): the conditions on
    components must leave exactly one dependent component."""
    from pycalphad.property_framework.computed_property import LinearCombination
    elements = phase_record_factory.nonvacant_elements
    number_dof = len(elements) - 1
    for key in conditions.keys():
        if not (hasattr(key, 'species') or isinstance(key, LinearCombination)):
            continue
        if hasattr(key, 'species') and getattr(key, 'phase_name', None) is not None:
            continue   # phase-local conditions do not reduce the degrees of freedom
        number_dof -= 1
    if number_dof != 0:
        raise ValueError('Number of degrees of freedom is not zero')


def equilibrium(dbf, comps, phases, conditions, output=None, model=None, verbose=False, calc_opts=None,
                to_xarray=True, parameters=None, solver=None, phase_records=None, cg_threads=None, cg_fallback=True,
                **kwargs):
    """
    pycalphad.equilibrium with the equilibria computed by column generation (CGWorkspace); same arguments and
    result, plus ``cg_threads`` and ``cg_fallback`` (see CGWorkspace).
    """
    if output is None:
        output = set()
    elif (not isinstance(output, Iterable)) or isinstance(output, str):
        output = [output]
    wks = CGWorkspace(database=dbf, components=comps, phases=phases, conditions=conditions, models=model,
                      parameters=parameters, verbose=verbose, calc_opts=calc_opts, solver=solver,
                      phase_record_factory=phase_records, cg_threads=cg_threads, cg_fallback=cg_fallback)
    # (what follows is pycalphad.equilibrium's, on the CGWorkspace)
    properties = wks.eq
    conds_keys = [str(k) for k in properties.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
    output = sorted(set(output) - {'GM', 'MU'})
    for out in output:
        cprop = as_property(out)
        out = str(cprop)
        result_array = np.zeros(properties.GM.shape)
        for index, composition_sets in wks.enumerate_composition_sets():
            cur_conds = OrderedDict(zip(conds_keys, [np.asarray(properties.coords[b][a], dtype=np.float64)
                                                     for a, b in zip(index, conds_keys)]))
            result_array[index] = cprop.compute_property(composition_sets, cur_conds, properties.MU[index])
        result = LightDataset({out: (conds_keys, result_array)}, coords=properties.coords)
        properties.merge(result, inplace=True, compat='equals')
    if to_xarray:
        properties = wks.eq.get_dataset()
    properties.attrs['created'] = datetime.now().isoformat()
    if len(kwargs) > 0:
        warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
    return properties
