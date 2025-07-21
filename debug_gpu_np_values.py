#!/usr/bin/env python
"""Debug what NP values the GPU is actually returning."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing GPU NP values for X(TI)=0.1, T=600K")
print("="*80)

# Monkey-patch the GPU equilibrium function to print debug info
import pycalphad.gpu.gpu_equilibrium as gpu_eq

original_process = gpu_eq._process_gpu_results

def debug_process_gpu_results(results_cpu_flat, wks_obj, num_conditions_total, 
                              unique_py_models, py_phase_name_to_unique_idx_map,
                              original_properties=None, dynamic_sizes=None):
    """Wrapper to debug GPU results processing."""
    print("\n[DEBUG] Inside _process_gpu_results")
    print(f"results_cpu_flat type: {type(results_cpu_flat)}")
    print(f"results_cpu_flat shape: {results_cpu_flat.shape if hasattr(results_cpu_flat, 'shape') else 'N/A'}")
    
    if hasattr(results_cpu_flat, 'dtype'):
        print(f"results_cpu_flat dtype: {results_cpu_flat.dtype}")
        if results_cpu_flat.size > 0:
            print(f"\nFirst result fields:")
            for field in results_cpu_flat.dtype.names:
                value = results_cpu_flat[0][field]
                if isinstance(value, np.ndarray):
                    print(f"  {field}: {value[:4] if len(value) > 4 else value} (shape={value.shape})")
                else:
                    print(f"  {field}: {value}")
            
            # Special focus on NP values
            np_values = results_cpu_flat[0]['NP']
            print(f"\nDetailed NP values from GPU:")
            for i, val in enumerate(np_values):
                print(f"  NP[{i}] = {val:.15f}")
    
    # Call original function
    return original_process(results_cpu_flat, wks_obj, num_conditions_total,
                            unique_py_models, py_phase_name_to_unique_idx_map,
                            original_properties, dynamic_sizes)

# Apply monkey patch
gpu_eq._process_gpu_results = debug_process_gpu_results

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

print("\n\nFinal result analysis:")
print(f"GM: {float(result_gpu.GM.values)}")
print(f"Phase array shape: {result_gpu.Phase.values.shape}")
print(f"NP array shape: {result_gpu.NP.values.shape}")
print(f"Phase names: {result_gpu.Phase.values.flatten()}")
print(f"Phase amounts: {result_gpu.NP.values.flatten()}")

# Count active phases
active_phases = []
for i, (phase, amount) in enumerate(zip(result_gpu.Phase.values.flatten(), 
                                        result_gpu.NP.values.flatten())):
    if phase and amount > 1e-10:
        active_phases.append((phase, amount))

print(f"\nActive phases: {len(active_phases)}")
for phase, amount in active_phases:
    print(f"  {phase}: {amount}")