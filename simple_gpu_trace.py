#!/usr/bin/env python
"""Simple trace to find 0.989890."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import subprocess

# Run GPU test and capture output
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

proc = subprocess.Popen([sys.executable, '-c', f'''
import sys
sys.path.insert(0, "{os.getcwd()}")
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = filter_phases(dbf, comps)
conditions = {{v.X("TI"): 0.01, v.T: 1000, v.P: 101325}}

result = equilibrium(dbf, comps, phases, conditions, gpu=True)
'''], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

output, _ = proc.communicate()

# Search for 0.989890
lines = output.split('\n')
for i, line in enumerate(lines):
    if '0.989890' in line or '0.98989' in line:
        print(f"\nFound 0.989890 at line {i}:")
        print("Context:")
        for j in range(max(0, i-5), min(len(lines), i+6)):
            print(f"  {j}: {lines[j]}")
        print()