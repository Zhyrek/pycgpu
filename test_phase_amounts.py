import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Simple test to check phase amounts in CPU vs GPU
dbf = Database("""
ELEMENT AL FCC_A1 0 0 0 !
ELEMENT CU FCC_A1 0 0 0 !
ELEMENT FE BCC_A2 0 0 0 !

PHASE LIQUID % 1 1.0 !
CONSTITUENT LIQUID :AL,CU,FE: !

PARAMETER G(LIQUID,AL;0) 298.15 -10000; 6000 N !
PARAMETER G(LIQUID,CU;0) 298.15 -12000; 6000 N !
PARAMETER G(LIQUID,FE;0) 298.15 -11000; 6000 N !
PARAMETER G(LIQUID,AL,CU;0) 298.15 -68000; 6000 N !
PARAMETER G(LIQUID,AL,FE;0) 298.15 -91700; 6000 N !
PARAMETER G(LIQUID,CU,FE;0) 298.15 -35500; 6000 N !
""")

# Test conditions
conditions = {v.X('CU'): 0.3, v.X('FE'): 0, v.T: 900, v.P: 101325}

# Run CPU version
print("Running CPU version...")
cpu_result = equilibrium(dbf, ['AL', 'CU', 'FE'], ['LIQUID'], conditions, calc_opts={'pdens': 50})
print(f"CPU GM: {cpu_result.GM.values.flatten()[0]:.2f} J/mol")
print(f"CPU Phase amounts: {cpu_result.NP.values}")
print(f"CPU X(CU): {cpu_result.X.sel(component='CU').values.flatten()[0]:.6f}")

# Run GPU version
print("\nRunning GPU version...")
gpu_result = equilibrium(dbf, ['AL', 'CU', 'FE'], ['LIQUID'], conditions, calc_opts={'pdens': 50}, gpu=True)
print(f"GPU GM: {gpu_result.GM.values.flatten()[0]:.2f} J/mol")
print(f"GPU Phase amounts: {gpu_result.NP.values}")
print(f"GPU X(CU): {gpu_result.X.sel(component='CU').values.flatten()[0]:.6f}")

print(f"\nDifference in GM: {abs(cpu_result.GM.values.flatten()[0] - gpu_result.GM.values.flatten()[0]):.2f} J/mol")