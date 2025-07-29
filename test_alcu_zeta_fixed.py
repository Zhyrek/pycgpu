#!/usr/bin/env python
"""Test GPU with ALCU_ZETA at conditions where it's stable - fixed version."""

from pycalphad import Database, equilibrium, variables as v
import warnings

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Test with LIQUID and ALCU_ZETA
phases = ['LIQUID', 'ALCU_ZETA']

print("Testing at conditions where ALCU_ZETA should be stable")
p = dbf.phases['ALCU_ZETA']
print(f"ALCU_ZETA: sublattices={p.sublattices}, constituents={p.constituents}")
print()

# Use the original test conditions that showed the error
conditions = {v.T: 1273.15, v.P: 101325, v.N: 1, v.X('CU'): 0.3, v.X('FE'): 0.4}

print(f"Conditions: X(AL)=0.3, X(CU)=0.3, X(FE)=0.4, T=1000°C")

# CPU test
print("\nCPU calculation...")
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 10}, verbose=False)
    cpu_gm = float(cpu_result.GM.values)
    print(f"CPU SUCCESS: GM={cpu_gm:.2f} J/mol")
    
    # Check which phases are stable
    print("CPU phases:")
    for idx in range(cpu_result.dims['vertex']):
        np_val = float(cpu_result.NP.values[0,0,0,0,0,idx])
        if np_val > 1e-6:
            phase_name = str(cpu_result.Phase.values[0,0,0,0,0,idx])
            if phase_name and phase_name != '':
                print(f"  {phase_name}: NP={np_val:.4f}")
    
except Exception as e:
    print(f"CPU FAILED: {type(e).__name__}: {str(e)}")
    cpu_gm = None

# GPU test
print("\nGPU calculation...")
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 10}, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values)
    print(f"GPU SUCCESS: GM={gpu_gm:.2f} J/mol")
    
    # Check which phases are stable
    print("GPU phases:")
    for idx in range(gpu_result.dims['vertex']):
        np_val = float(gpu_result.NP.values[0,0,0,0,0,idx])
        if np_val > 1e-6:
            phase_name = str(gpu_result.Phase.values[0,0,0,0,0,idx])
            if phase_name and phase_name != '':
                print(f"  {phase_name}: NP={np_val:.4f}")
    
    # Compare results
    if cpu_gm is not None:
        diff = abs(cpu_gm - gpu_gm)
        print(f"\nGM Difference: {diff:.6f} J/mol - {'PASS' if diff < 1.0 else 'FAIL'}")
        
except Exception as e:
    print(f"GPU FAILED: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()