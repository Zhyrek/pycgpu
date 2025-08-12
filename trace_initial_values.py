#!/usr/bin/env python
"""Trace initial values passed to solvers."""

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
print("TRACE INITIAL VALUES FOR SOLVERS")
print("=" * 80)

# First do CPU with minimal output
from pycalphad import calculate
from pycalphad.core.utils import unpack_condition
from pycalphad.core.starting_point import starting_point

# Get starting point
from collections import OrderedDict
unitless_conds = OrderedDict()
for cond, value in conditions.items():
    unitless_conds[cond] = np.atleast_1d(value)
unitless_conds[v.N] = np.array([1.0])

# Calculate grid
grid_calc = calculate(dbf, comps, phases, T=conditions[v.T], P=conditions[v.P], pdens=50, model=None)
print(f"Grid GM range: [{grid_calc.GM.values.min():.2f}, {grid_calc.GM.values.max():.2f}]")

# Get starting point
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.workspace import Workspace

wks = Workspace(dbf, comps, phases, conditions, calc_opts={'pdens': 50})
state_variables = [v.N, v.P, v.T]

# CPU starting point
prop = starting_point(unitless_conds, state_variables, wks.phase_record_factory, grid_calc)
print("\nCPU Starting Point:")
print(f"  GM: {float(prop.GM.values):.2f}")
print(f"  MU: {prop.MU.values.flatten()}")
print(f"  NP: {prop.NP.values.flatten()}")
print(f"  X shape: {prop.X.shape}")
print(f"  Phase array: {prop.Phase.values.flatten()}")

# Extract compositions for active phases
x_values = prop.X.values[0,0,0,0,0,:,:]
np_values = prop.NP.values[0,0,0,0,0,:]
phase_values = prop.Phase.values[0,0,0,0,0,:]

print("\nPhase-wise compositions from starting point:")
for i in range(len(np_values)):
    if np_values[i] > 1e-10:
        print(f"  Phase {i} ({phase_values[i]}): NP={np_values[i]:.6f}")
        print(f"    X(AL)={x_values[i,0]:.6f}, X(CU)={x_values[i,1]:.6f}, X(FE)={x_values[i,2]:.6f}")

# Now check what gets passed to GPU
print("\n" + "=" * 80)
print("GPU INITIAL VALUES")
print("=" * 80)

# Run GPU equilibrium with minimal verbose to capture initial values
import sys
import io
from contextlib import redirect_stdout

captured = io.StringIO()
with redirect_stdout(captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

output = captured.getvalue()

# Extract key initial values from GPU output
print("Extracted GPU initial values:")
for line in output.split('\n'):
    if 'initial_chemical_potentials' in line and 'Thread 0' in line:
        print(f"  {line.strip()}")
    elif 'prescribed_mole_fraction_rhs' in line and 'Thread 0' in line:
        print(f"  {line.strip()}")
    elif 'Phase 0 input data' in line:
        print(f"  {line.strip()}")
    elif 'phase_amount =' in line and 'Phase' in line:
        print(f"  {line.strip()}")
    elif 'initial_data->site_fractions' in line:
        print(f"  {line.strip()}")
    elif 'DOF for energy calc' in line and 'Phase 0' in line:
        print(f"  {line.strip()}")
        
# Final results comparison
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

result_cpu = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 50})
cpu_gm = float(result_cpu.GM.values)
gpu_gm = float(result_gpu.GM.values)

print(f"CPU GM: {cpu_gm:.2f} J/mol")
print(f"GPU GM: {gpu_gm:.2f} J/mol") 
print(f"Difference: {abs(cpu_gm - gpu_gm):.2f} J/mol")

# Check mole fractions being used
print("\nMole fraction conditions:")
print(f"  X(CU) = {conditions[v.X('CU')]}")
print(f"  X(FE) = {conditions[v.X('FE')]}")
print(f"  X(AL) = {1.0 - conditions[v.X('CU')] - conditions[v.X('FE')]}")

# Look for how mole fractions are calculated
print("\nGPU mole fraction values from output:")
for line in output.split('\n'):
    if 'mole_fractions:' in line and ('[0.800000' in line or '[0.500000' in line):
        print(f"  {line.strip()}")