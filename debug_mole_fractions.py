#!/usr/bin/env python
"""Debug mole fractions being used in GPU calculation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Single ternary condition
conditions = {
    v.T: 1200,
    v.P: 101325,
    v.X('CU'): 0.2,
    v.X('FE'): 0.3  # X(AL) = 0.5 implied
}

print("=" * 80)
print("DEBUG MOLE FRACTIONS IN GPU")
print("=" * 80)
print(f"Expected mole fractions: X(AL)=0.5, X(CU)=0.2, X(FE)=0.3, X(VA)=0.0")

# Run GPU with verbose and capture output
import sys
import io
from contextlib import redirect_stdout

captured = io.StringIO()
with redirect_stdout(captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

output = captured.getvalue()

# Look for mole fraction lines
print("\nMole fraction lines from GPU output:")
for line in output.split('\n'):
    if 'mole_fractions' in line.lower() and '[' in line:
        print(f"  {line.strip()}")
        # Check if it's the wrong value
        if '[0.800000' in line or '[0.8' in line:
            print("    ^^^ WRONG! Should be [0.5, 0.2, 0.3, 0.0]")
        elif '[0.500000' in line or '[0.5' in line:
            print("    ^^^ CORRECT!")
            
# Also check the debug output that shows what thread sees
print("\nThread mole fraction lines:")
for line in output.split('\n'):
    if 'Thread 0 mole fractions:' in line:
        print(f"  {line.strip()}")
        if 'X(NB)' in line:
            print("    ^^^ Wrong component names! Should be AL, CU, FE")