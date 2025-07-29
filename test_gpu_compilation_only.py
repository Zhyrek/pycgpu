#!/usr/bin/env python
"""Test just GPU kernel compilation for ALCU_ZETA"""

from pycalphad import Database, Model
from pycalphad.gpu.gpu_equilibrium import build_equilibrium_kernel
import numpy as np

db = Database('Al-Cu-Fe.tdb')

# Test ALCU_ZETA phase
phase_name = 'ALCU_ZETA'
print(f"\nTesting {phase_name} GPU compilation...")

try:
    mod = Model(db, ['AL', 'CU', 'FE'], phase_name)
    print(f"Model created, site fractions: {len(mod.site_fractions)}")
    
    # Try to build the kernel
    kernel = build_equilibrium_kernel(
        db=db,
        phases=[phase_name],
        components=['AL', 'CU', 'FE'],
        models={phase_name: mod},
        phase_records={},  # Will be built internally
        verbose=True
    )
    
    if kernel is not None:
        print(f"SUCCESS! Kernel compiled for {phase_name}")
    else:
        print(f"FAILED: Kernel is None for {phase_name}")
        
except Exception as e:
    print(f"FAILED with {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()