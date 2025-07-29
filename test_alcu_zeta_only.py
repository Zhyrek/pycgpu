#!/usr/bin/env python
"""Test GPU with only ALCU_ZETA phase to isolate multi-sublattice handling."""

from pycalphad import Database, equilibrium, variables as v
import warnings

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Only ALCU_ZETA phase
phases = ['ALCU_ZETA']

print("Testing GPU with only ALCU_ZETA phase (2 sublattices)")
p = dbf.phases['ALCU_ZETA']
print(f"Sublattices: {p.sublattices}")
print(f"Constituents: {p.constituents}")
print()

# Composition in ALCU_ZETA stability region
conditions = {v.T: 700, v.P: 101325, v.N: 1, v.X('CU'): 0.45, v.X('FE'): 0.0}

print(f"Conditions: X(AL)=0.55, X(CU)=0.45, X(FE)=0.00, T=700K")

# CPU test
print("\nCPU calculation...")
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 10}, verbose=False)
    cpu_gm = float(cpu_result.GM.values)
    print(f"CPU SUCCESS: GM={cpu_gm:.2f} J/mol")
    
    # Get site fractions
    Y = cpu_result.Y.sel(vertex=0).values
    print(f"Site fractions: Y(ALCU_ZETA,0,AL)={Y[0,0,0,0,0]:.4f}")
    print(f"               Y(ALCU_ZETA,1,CU)={Y[0,0,0,0,1]:.4f}")
    
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
    
    # Get site fractions
    Y = gpu_result.Y.sel(vertex=0).values
    print(f"Site fractions: Y(ALCU_ZETA,0,AL)={Y[0,0,0,0,0]:.4f}")
    print(f"               Y(ALCU_ZETA,1,CU)={Y[0,0,0,0,1]:.4f}")
    
    # Compare
    if cpu_gm is not None:
        diff = abs(gpu_gm - cpu_gm)
        print(f"\nDifference: {diff:.6f} J/mol - {'PASS' if diff < 1.0 else 'FAIL'}")
    
except Exception as e:
    print(f"GPU FAILED: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()