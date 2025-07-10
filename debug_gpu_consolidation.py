#!/usr/bin/env python3
"""Debug why GPU phase consolidation isn't happening"""

from pycalphad import calculate, equilibrium, Database
import numpy as np

# Load database and set up system
dbf = Database('tests/databases/nbalti_print.tdb')

# Define conditions exactly as in the issue
conditions = {
    'T': 300.0,
    'P': 101325.0,
    'X(TI)': 0.4,
    'N': 1.0
}

print("=== GPU Phase Consolidation Debug ===\n")

# Run calculate to get initial states
calc_result = calculate(dbf, ['NB', 'TI'], 'BCC_A2', 
                        T=conditions['T'], P=conditions['P'], 
                        N=1.0, 
                        points={'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]})

print("Initial calculate result phases:")
for i in range(calc_result.X.shape[-1]):
    print(f"  Phase {i}: X(NB)={calc_result.X.sel(component='NB').values.flat[i]:.6f}, "
          f"X(TI)={calc_result.X.sel(component='TI').values.flat[i]:.6f}, "
          f"GM={calc_result.GM.values.flat[i]:.1f}")

# Now run equilibrium with GPU
print("\nRunning GPU equilibrium...")
try:
    eq_gpu = equilibrium(dbf, ['NB', 'TI'], ['BCC_A2'], conditions, 
                        calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                        gpu=True, verbose=True)
    
    print(f"\nGPU Result: GM = {eq_gpu.GM.values[0]:.1f} J/mol")
    print(f"GPU phases: {eq_gpu.Phase.values}")
    print(f"GPU NP: {eq_gpu.NP.values}")
    
except Exception as e:
    print(f"GPU equilibrium failed: {e}")

# Compare with CPU
print("\nRunning CPU equilibrium...")
eq_cpu = equilibrium(dbf, ['NB', 'TI'], ['BCC_A2'], conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    verbose=True)

print(f"\nCPU Result: GM = {eq_cpu.GM.values[0]:.1f} J/mol")
print(f"CPU phases: {eq_cpu.Phase.values}")
print(f"CPU NP: {eq_cpu.NP.values}")

print(f"\nDifference: {abs(eq_gpu.GM.values[0] - eq_cpu.GM.values[0]):.1f} J/mol")