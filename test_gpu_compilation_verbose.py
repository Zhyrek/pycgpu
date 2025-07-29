#!/usr/bin/env python
"""Test GPU compilation with verbose output to see actual nvcc errors."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import sys

# Redirect stderr to see compilation errors
import io
import contextlib

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Simple test conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing GPU compilation with minimal phases...")
print("="*80)

# Test with increasing number of phases to find which ones cause issues
phase_list = list(db.phases.keys())

# Start with single phases
for phase in phase_list[:10]:
    print(f"\nTesting phase: {phase}")
    
    # Capture stderr to see compilation errors
    stderr_capture = io.StringIO()
    
    try:
        with contextlib.redirect_stderr(stderr_capture):
            result = equilibrium(db, components, [phase], conditions, 
                               calc_opts={'pdens': 10}, 
                               gpu=True, 
                               verbose=False)
        print(f"  ✓ SUCCESS: {phase} compiled successfully")
        
    except Exception as e:
        stderr_output = stderr_capture.getvalue()
        print(f"  ✗ FAILED: {phase}")
        print(f"    Error type: {type(e).__name__}")
        print(f"    Error message: {str(e)[:200]}")
        
        # Look for nvcc errors in stderr
        if 'nvcc' in stderr_output or 'error' in stderr_output.lower():
            print(f"    Compilation errors found:")
            # Extract just the error lines
            error_lines = [line for line in stderr_output.split('\n') 
                          if 'error' in line.lower() or 'nvcc' in line]
            for line in error_lines[:5]:
                print(f"      {line}")

# Test combinations of phases
print("\n" + "="*80)
print("Testing phase combinations...")

test_combinations = [
    ['LIQUID'],
    ['LIQUID', 'FCC_A1'],
    ['LIQUID', 'FCC_A1', 'BCC_A2'],
    ['LIQUID', 'AL2FE'],
    ['LIQUID', 'ALCU_ZETA'],
]

for phases in test_combinations:
    print(f"\nTesting phases: {phases}")
    
    stderr_capture = io.StringIO()
    
    try:
        with contextlib.redirect_stderr(stderr_capture):
            result = equilibrium(db, components, phases, conditions, 
                               calc_opts={'pdens': 10}, 
                               gpu=True, 
                               verbose=False)
        print(f"  ✓ SUCCESS: Combination compiled successfully")
        
    except Exception as e:
        stderr_output = stderr_capture.getvalue()
        print(f"  ✗ FAILED")
        print(f"    Error: {str(e)[:200]}")
        
        # Save stderr output for detailed analysis
        if stderr_output and 'nvcc' in str(e):
            filename = f"nvcc_errors_{'_'.join(phases)}.txt"
            with open(filename, 'w') as f:
                f.write(f"Phases: {phases}\n")
                f.write(f"Error: {str(e)}\n")
                f.write("="*80 + "\n")
                f.write("STDERR OUTPUT:\n")
                f.write(stderr_output)
            print(f"    Saved detailed errors to: {filename}")

print("\n" + "="*80)
print("Summary: Check generated error files for nvcc compilation details")