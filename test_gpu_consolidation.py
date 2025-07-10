#!/usr/bin/env python3
"""Test GPU phase consolidation"""

from pycalphad import Database, equilibrium, calculate, variables as v
import numpy as np

# Create a simple TDB with known behavior
tdb_content = """
ELEMENT NB   FCC_A1   92.906     5220.000     36.270    !
ELEMENT TI   HCP_A3   47.867     4824.000     30.760    !
ELEMENT VA   VACUUM   0.0        0.0          0.0       !

TYPE_DEFINITION % SEQ *!

PHASE BCC_A2 %  2 1.0 3.0 !
CONSTITUENT BCC_A2 :NB,TI:VA: !

PARAMETER G(BCC_A2,NB:VA;0)   298.15  -8519.353+142.048*T-26.4711*T*LN(T)
   +0.203475E-3*T**2-0.35012E-6*T**3+93399*T**(-1);  6000.00  N  !
PARAMETER G(BCC_A2,TI:VA;0)   298.15  -8059.921+133.616*T-23.9933*T*LN(T)
   -0.0047743*T**2+0.106271E-6*T**3+72636*T**(-1);  6000.00  N  !
PARAMETER G(BCC_A2,NB,TI:VA;0) 298.15 +20000-5*T; 6000.00 N !
PARAMETER G(BCC_A2,NB,TI:VA;1) 298.15 +10000; 6000.00 N !
"""

# Write TDB file
with open('test_nbti.tdb', 'w') as f:
    f.write(tdb_content)

# Load database
db = Database('test_nbti.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions that should trigger consolidation
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Testing Phase Consolidation ===\n")

# First run calculate to see initial states
print("Initial calculate with two starting points:")
calc_result = calculate(db, comps, 'BCC_A2', T=300, P=101325, N=1,
                       points={'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]})

print(f"Number of points: {calc_result.X.shape[-1]}")
for i in range(calc_result.X.shape[-1]):
    print(f"  Point {i}: X(NB)={calc_result.X.sel(component='NB').values.flat[i]:.6f}, "
          f"X(TI)={calc_result.X.sel(component='TI').values.flat[i]:.6f}, "
          f"GM={calc_result.GM.values.flat[i]:.1f} J/mol")

# Run CPU equilibrium
print("\n--- CPU Equilibrium ---")
eq_cpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    verbose=True)

print(f"CPU GM: {eq_cpu.GM.values[0]:.1f} J/mol")
print(f"CPU phases: {eq_cpu.Phase.values}")
print(f"CPU NP: {eq_cpu.NP.values}")

# Run GPU equilibrium  
print("\n--- GPU Equilibrium ---")
eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True, verbose=True)

print(f"GPU GM: {eq_gpu.GM.values[0]:.1f} J/mol")
print(f"GPU phases: {eq_gpu.Phase.values}")
print(f"GPU NP: {eq_gpu.NP.values}")

print(f"\nDifference: {abs(eq_gpu.GM.values[0] - eq_cpu.GM.values[0]):.1f} J/mol")