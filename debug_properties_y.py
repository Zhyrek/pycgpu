#!/usr/bin/env python
"""Debug what Y values are in properties."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np

# Patch gpu_equilibrium to print Y values
import pycalphad.gpu.gpu_equilibrium

original_prepare = pycalphad.gpu.gpu_equilibrium._prepare_gpu_data

def patched_prepare(wks_obj, unique_py_models, py_phase_name_to_unique_idx_map, dynamic_sizes, properties=None, grid=None):
    if properties is not None:
        print("\n[DEBUG PROPERTIES] Checking values in properties:")
        
        # Check X values
        if hasattr(properties, 'X'):
            print(f"\n  properties.X shape: {properties.X.shape if hasattr(properties.X, 'shape') else 'N/A'}")
            if hasattr(properties.X, '__getitem__'):
                print(f"  properties.X values:\n{properties.X}")
        
        # Check Y values
        if hasattr(properties, 'Y'):
            print(f"\n  properties.Y type: {type(properties.Y)}")
            print(f"  properties.Y shape: {properties.Y.shape if hasattr(properties.Y, 'shape') else 'N/A'}")
            if isinstance(properties.Y, np.ndarray):
                print(f"  properties.Y values:\n{properties.Y}")
        else:
            print(f"\n  properties has no Y attribute")
            
        # Check what other attributes exist
        print("\n[DEBUG] All properties attributes:")
        attrs = [attr for attr in dir(properties) if not attr.startswith('_')]
        print(f"  {attrs[:10]}...")  # First 10 attributes
    
    return original_prepare(wks_obj, unique_py_models, py_phase_name_to_unique_idx_map, dynamic_sizes, properties, grid)

pycalphad.gpu.gpu_equilibrium._prepare_gpu_data = patched_prepare

# Run test
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

print("Running GPU equilibrium to debug Y values...")
result = equilibrium(dbf, comps, phases, conditions, gpu=True)
print(f"\nFinal GPU X(TI) = {result.X.sel(component='TI').values.flatten()[0]:.8f}")