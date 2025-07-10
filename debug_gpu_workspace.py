#!/usr/bin/env python3
"""Debug GPU workspace state"""
import os
os.environ['PYCALPHAD_DEBUG'] = '0'

from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache

# Clear cached GPU modules
clear_gpu_cache()

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Testing GPU Workspace ===")

# Manually create a workspace to inspect
from pycalphad.core.workspace import Workspace
wks = Workspace(db, ['NB', 'TI'], phases, conditions)

print(f"\nWorkspace phase_record_factory: {wks.phase_record_factory}")
if wks.phase_record_factory:
    print(f"State variables: {wks.phase_record_factory.state_variables}")

# Test GPU code generation
print("\n=== Testing GPU Code Generation ===")
from pycalphad.gpu.gpu_codegen import notebook_get_all_sym_names_for_model

model = wks.models['BCC_A2']
var_names = notebook_get_all_sym_names_for_model(model, wks)
print(f"Variable names for BCC_A2: {var_names}")

# Check the actual GPU equilibrium call
print("\n=== Running GPU Equilibrium (will show variable mapping) ===")
try:
    gpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})
    print("GPU equilibrium completed")
except Exception as e:
    print(f"GPU equilibrium failed: {e}")