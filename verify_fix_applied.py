#!/usr/bin/env python
"""Verify that fix_all_zero_piecewise_from_logs is being applied in GPU code generation."""

from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _nb_formulahess_from_model

# Load database and create model
dbf = Database('NbTi.tdb')
mod = Model(dbf, ['NB', 'TI'], 'LIQUID')

# Create a minimal workspace
wks = Workspace(dbf, ['NB', 'TI'], ['LIQUID'], {})

# Generate the Hessian C code
print("Generating Hessian C code...")
hess_code = _nb_formulahess_from_model(mod, 0, wks)

# Count patterns
import re

# Count all-zero patterns
all_zero_simple = len(re.findall(r'\(\(1e-15 < x\[\d+\]\) \? \(0\) : \(0\)\)', hess_code))
all_zero_mult = len(re.findall(r'1\.0\*\(\(1e-15 < x\[\d+\]\) \? \(0\) : \(0\)\)', hess_code))

# Count fixed patterns
fixed_simple = len(re.findall(r'\(\(1e-15 < x\[\d+\]\) \? \(pow\(x\[\d+\], \(-1\)\)\) : \(0\)\)', hess_code))
fixed_mult = len(re.findall(r'1\.0\*\(\(1e-15 < x\[\d+\]\) \? \(pow\(x\[\d+\], \(-1\)\)\) : \(0\)\)', hess_code))

print(f"\nPattern counts in generated Hessian:")
print(f"All-zero simple patterns: {all_zero_simple}")
print(f"All-zero multiplication patterns: {all_zero_mult}")
print(f"Fixed simple patterns: {fixed_simple}")
print(f"Fixed multiplication patterns: {fixed_mult}")

# Show a sample
if all_zero_mult > 0:
    match = re.search(r'(1\.0\*\(\(1e-15 < x\[\d+\]\) \? \(0\) : \(0\)\))', hess_code)
    if match:
        # Find surrounding context
        pos = match.start()
        start = max(0, pos - 50)
        end = min(len(hess_code), pos + 100)
        print(f"\nSample of unfixed pattern at position {pos}:")
        print(hess_code[start:end])