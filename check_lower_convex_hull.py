#!/usr/bin/env python
"""Check what lower_convex_hull returns."""

from pycalphad import Database, calculate, variables as v
from pycalphad.core.utils import filter_phases
from pycalphad.core.lower_convex_hull import lower_convex_hull
import numpy as np

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run calculate
calc_result = calculate(dbf, comps, phases, T=conditions[v.T], P=conditions[v.P], output='GM')

# Run lower_convex_hull
hull_result = lower_convex_hull(calc_result, comps)

# Check the compositions and site fractions
print('Lower convex hull results:')
stable_mask = hull_result.Phase != ''
stable_phases = hull_result.Phase.values[stable_mask]
stable_x_ti = hull_result.X.sel(component='TI').values[stable_mask]
stable_np = hull_result.NP.values[stable_mask]

print(f'Stable phases: {stable_phases}')
print(f'X(TI) values: {stable_x_ti}')
print(f'Phase amounts: {stable_np}')

# Check site fractions
for phase in set(stable_phases):
    if phase and phase != '':
        phase_mask = hull_result.Phase.values == phase
        phase_indices = np.where(phase_mask)[0]
        if len(phase_indices) > 0:
            idx = phase_indices[0]
            print(f'\nPhase {phase}:')
            print(f'  X(TI) = {hull_result.X.sel(component="TI").values[idx]:.15f}')
            if 'Y' in hull_result.data_vars:
                # Print all site fractions
                y_vars = [v for v in hull_result.data_vars if v.startswith('Y(')]
                for y_var in sorted(y_vars):
                    print(f'  {y_var} = {hull_result[y_var].values[idx]:.15f}')