import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Simple test for single sublattice phase equilibrium
dbf = Database("""
ELEMENT AL FCC_A1 0 0 0 \!
ELEMENT CU FCC_A1 0 0 0 \!
ELEMENT FE BCC_A2 0 0 0 \!

PHASE LIQUID % 1 1.0 \!
CONSTITUENT LIQUID :AL,CU,FE: \!

PARAMETER G(LIQUID,AL;0) 298.15 -10000; 6000 N \!
PARAMETER G(LIQUID,CU;0) 298.15 -12000; 6000 N \!
PARAMETER G(LIQUID,FE;0) 298.15 -11000; 6000 N \!
PARAMETER G(LIQUID,AL,CU;0) 298.15 -68000; 6000 N \!
PARAMETER G(LIQUID,AL,FE;0) 298.15 -91700; 6000 N \!
PARAMETER G(LIQUID,CU,FE;0) 298.15 -35500; 6000 N \!
""")

# Test conditions - high Ti content
conditions = {v.X('CU'): 0.3, v.X('FE'): 0, v.T: 900, v.P: 101325}

# Run CPU version
cpu_result = equilibrium(dbf, ['AL', 'CU', 'FE'], ['LIQUID'], conditions, calc_opts={'pdens': 50})

# Run GPU version
gpu_result = equilibrium(dbf, ['AL', 'CU', 'FE'], ['LIQUID'], conditions, calc_opts={'pdens': 50}, gpu=True)

print("CPU X(CU):", cpu_result.X.sel(component='CU').values.item())
print("GPU X(CU):", gpu_result.X.sel(component='CU').values.item())
print("Difference:", abs(cpu_result.X.sel(component='CU').values.item() - gpu_result.X.sel(component='CU').values.item()))
EOF < /dev/null
