#!/usr/bin/env python
"""Simple test to check GPU final values."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Redirect output to file to avoid truncation
import subprocess
proc = subprocess.Popen([sys.executable, '-c', '''
import sys
sys.path.insert(0, ".")
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)
conditions = {v.X("TI"): 0.01, v.T: 1000, v.P: 101325}

result = equilibrium(dbf, comps, phases, conditions, gpu=True)
print(f"\\nGPU X(TI) = {result.X.sel(component='TI').values.flatten()[0]:.8f}")
'''], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

output, _ = proc.communicate()

# Find GPU FINAL VALUES section
lines = output.split('\n')
for i, line in enumerate(lines):
    if '[GPU FINAL VALUES]' in line:
        print("\nFound GPU FINAL VALUES:")
        for j in range(i, min(i+20, len(lines))):
            print(lines[j])
        break

# Also show the final result
if 'GPU X(TI)' in output:
    idx = output.find('GPU X(TI)')
    print("\n" + output[idx:idx+30])