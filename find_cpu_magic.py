#!/usr/bin/env python
"""Find the exact point where CPU succeeds."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Look at previous output to understand the pattern
print("ANALYZING CPU SUCCESS PATTERN")
print("=" * 60)

print("\nFrom previous output, we see CPU:")
print("1. Starts with 2 phases")
print("2. Consolidates at iteration 1 when phases differ by 0.000014")
print("3. After consolidation has single phase with:")
print("   - Y(TI) = 0.9031471 (site fraction)")
print("   - X(TI) = 0.9031471 (mole fraction)")
print("4. RHS = 0.2056487 (this is key!)")
print("5. Converges immediately after consolidation")

print("\nBut GPU:")
print("1. Also consolidates to single phase")
print("2. Gets Y(TI) ≈ 0.903")
print("3. But RHS = 0.100 (DIFFERENT!)")
print("4. Gets stuck, can't converge")

print("\nThe key difference is the RHS value!")
print("CPU RHS = 0.206, GPU RHS = 0.100")

# Let's check the exact RHS calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# The RHS formula from code is:
# rhs = sum(phase_amt * c_G[component] / system_amt) + (target - current) * prefactor

print("\nRHS CALCULATION ANALYSIS:")
print("For single phase with X(TI) = 0.9031471:")
print("- Current X(TI) = 0.9031471")
print("- Target X(TI) = 0.9") 
print("- Error = 0.9031471 - 0.9 = 0.0031471")

print("\nCPU gets c_G ≈ [0.209, -0.209]")
print("RHS = phase_amt * c_G[1] + error")
print("    = 1.0 * (-0.209) + 0.0031471")
print("    = -0.209 + 0.0031471")
print("    = -0.2058529 (close to CPU's 0.2056487)")

print("\nBut this assumes c_G is negative for component 1...")
print("Wait, the sign matters!")

print("\nLet's check the actual formula from minimizer.pyx...")