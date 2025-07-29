#!/usr/bin/env python
"""Test GPU vs CPU with multiple sublattice phases."""

from pycalphad import Database, equilibrium, variables as v
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Test just LIQUID (1 sublattice) and ALCU_ZETA (2 sublattices)
phases = ['LIQUID', 'ALCU_ZETA']

print(f"Testing GPU vs CPU with phases: {phases}")

# Check sublattice structure
for phase in phases:
    if phase in dbf.phases:
        p = dbf.phases[phase]
        print(f"{phase}: sublattices={p.sublattices}, constituents={p.constituents}")
print()

# Simple test condition
conditions = {v.T: 900, v.P: 101325, v.N: 1, v.X('CU'): 0.17, v.X('FE'): 0.0}

print(f"Test conditions: X(AL)=0.83, X(CU)=0.17, X(FE)=0.00, T=900K")

try:
    # CPU calculation
    print("\nRunning CPU...")
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 5}, verbose=False)
    cpu_gm = float(cpu_result.GM.values)
    
    # Get phase amounts
    cpu_phases = []
    for phase in phases:
        try:
            # Select by phase name
            phase_data = cpu_result.sel(phase=phase)
            np_val = float(phase_data.NP.values[0,0,0,0])
            if np_val > 1e-6:
                cpu_phases.append(f"{phase}({np_val:.3f})")
        except:
            pass
    
    print(f"CPU SUCCESS: GM={cpu_gm:.2f} J/mol")
    print(f"CPU phases: {', '.join(cpu_phases) if cpu_phases else 'NONE'}")
    
except Exception as e:
    print(f"CPU FAILED: {type(e).__name__}: {str(e)}")
    cpu_gm = None

try:
    # GPU calculation
    print("\nRunning GPU...")
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 5}, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values)
    
    # Get phase amounts
    gpu_phases = []
    for phase in phases:
        try:
            # Select by phase name
            phase_data = gpu_result.sel(phase=phase)
            np_val = float(phase_data.NP.values[0,0,0,0])
            if np_val > 1e-6:
                gpu_phases.append(f"{phase}({np_val:.3f})")
        except:
            pass
    
    print(f"GPU SUCCESS: GM={gpu_gm:.2f} J/mol")
    print(f"GPU phases: {', '.join(gpu_phases) if gpu_phases else 'NONE'}")
    
    # Compare
    if cpu_gm is not None:
        diff = abs(gpu_gm - cpu_gm)
        print(f"\nDifference: {diff:.6f} J/mol - {'PASS' if diff < 1.0 else 'FAIL'}")
    
except Exception as e:
    print(f"GPU FAILED: {type(e).__name__}: {str(e)}")