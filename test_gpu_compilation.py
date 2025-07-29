#!/usr/bin/env python
"""Test if GPU code compiles correctly."""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_equilibrium import _compile_and_run_equilibrium_cuda
import cupy as cp
import os

# Clear cache
os.environ['CUDA_CACHE_DISABLE'] = '1'
cp.clear_memo()

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test conditions
conditions = {
    v.T: np.array([500.0]),
    v.P: np.array([101325.0]),
    v.X('TI'): np.array([0.1]),
    v.N: np.array([1.0])
}

print("Testing GPU compilation...")
print("="*80)

try:
    # Create workspace
    wks = Workspace(db, components, phases, conditions, verbose=False)
    
    # Prepare inputs for GPU kernel
    state_var_vals = []
    state_var_shapes = []
    
    for var in wks.state_variables:
        if var in conditions:
            values = np.atleast_1d(conditions[var])
            state_var_vals.append(values)
            state_var_shapes.append(values.shape[0])
        else:
            state_var_vals.append(np.array([0.0]))
            state_var_shapes.append(1)
    
    print(f"State variables: {[str(v) for v in wks.state_variables]}")
    print(f"State var shapes: {state_var_shapes}")
    
    # Get phase info
    phase_names = list(wks.phase_record_factory.keys())
    phase_dof = [pr.phase_dof for pr in wks.phase_record_factory.values()]
    
    print(f"\nPhases: {phase_names}")
    print(f"Phase DOF: {phase_dof}")
    
    # Compile GPU kernel
    print("\nCompiling GPU kernel...")
    result = _compile_and_run_equilibrium_cuda(
        state_var_vals, 
        state_var_shapes, 
        wks, 
        verbose=True
    )
    
    print("\n✓ GPU kernel compiled successfully!")
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    
except Exception as e:
    print(f"\n✗ ERROR during compilation: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)