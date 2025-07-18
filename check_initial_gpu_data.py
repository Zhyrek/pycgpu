#!/usr/bin/env python
"""Check what initial phase data is passed to GPU."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Patch gpu_equilibrium to print initial phase data
import pycalphad.gpu.gpu_equilibrium

original_prepare = pycalphad.gpu.gpu_equilibrium._prepare_gpu_data

def patched_prepare(wks_obj, unique_py_models, py_phase_name_to_unique_idx_map, dynamic_sizes, properties=None, grid=None):
    result = original_prepare(wks_obj, unique_py_models, py_phase_name_to_unique_idx_map, dynamic_sizes, properties, grid)
    
    if result[0] > 0:  # num_conditions > 0
        initial_phase_data = result[4]
        print("\n[INITIAL PHASE DATA] Being passed to GPU:")
        print(f"  num_phases: {initial_phase_data['num_phases'][0]}")
        print(f"  phase_indices: {initial_phase_data['phase_indices'][0, :]}")
        print(f"  phase_amounts: {initial_phase_data['phase_amounts'][0, :]}")
        print(f"  compositions: {initial_phase_data['compositions'][0, :, :]}")
        print(f"  site_fractions: {initial_phase_data['site_fractions'][0, :, :]}")
    
    return result

pycalphad.gpu.gpu_equilibrium._prepare_gpu_data = patched_prepare

# Run test
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("Running GPU equilibrium to check initial data...")
result = equilibrium(dbf, comps, phases, conditions, gpu=True)
print(f"\nFinal GPU X(TI) = {result.X.sel(component='TI').values.flatten()[0]:.8f}")