#!/usr/bin/env python
"""Analyze why CPU compiles Hessian expressions quickly while GPU is slow."""

import time
from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
import pycalphad.variables as v
import numpy as np

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Analyzing CPU vs GPU compilation differences")
print("="*60)

# Test LIQUID phase
phase = 'LIQUID'
print(f"\nAnalyzing {phase} phase...")

# 1. CPU approach - using SymEngine/compiled functions
print("\n1. CPU Compilation Process:")
start = time.time()

# Create model
model = Model(db, components, phase)
print(f"   - Model creation: {time.time() - start:.3f}s")

# Create phase record (this compiles the expressions)
start = time.time()
phase_rec = PhaseRecordFactory(db, components, v.GibbsMixingStateFunctions(), 
                               model=model, parameters={'GHSERAL': 298.15})
cpu_compile_time = time.time() - start
print(f"   - Phase record compilation: {cpu_compile_time:.3f}s")

# Test evaluation speed
start = time.time()
dof = np.array([1.0, 101325.0, 1273.15, 0.3, 0.3, 0.4])  # N, P, T, Y(AL), Y(CU), Y(FE)
for i in range(1000):
    energy = phase_rec.obj(dof)
eval_time = time.time() - start
print(f"   - 1000 evaluations: {eval_time:.3f}s ({eval_time/1000*1e6:.1f} μs per eval)")

# 2. GPU approach - analyze what happens
print("\n2. GPU Compilation Process:")
print("   - GPU uses nvcc to compile C++ code")
print("   - This involves:")
print("     a) Parsing massive C++ expressions (16k+ chars per line)")
print("     b) Building AST (Abstract Syntax Tree)")
print("     c) Optimizing for GPU architecture")
print("     d) Generating PTX (GPU assembly)")
print("     e) JIT compiling to GPU machine code")

# Show expression complexity
print("\n3. Expression Complexity Analysis:")
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models

wks = Workspace(db, components, [phase], conditions, verbose=False)
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, _, _, _ = result

lines = all_device_functions.split('\n')
hessian_lines = []
in_hess = False
for line in lines:
    if 'formulahess' in line and '__device__' in line:
        in_hess = True
    elif in_hess and line.strip() == '}':
        break
    elif in_hess:
        hessian_lines.append(line)

# Find longest line
max_len = max(len(line) for line in hessian_lines if line.strip())
print(f"   - Hessian function: {len(hessian_lines)} lines")
print(f"   - Longest line: {max_len:,} characters")

# Count operations in longest line
longest = max(hessian_lines, key=len)
print(f"\n4. Operations in longest Hessian line:")
print(f"   - Additions: {longest.count('+')}")
print(f"   - Multiplications: {longest.count('*')}")
print(f"   - pow() calls: {longest.count('pow(')}")
print(f"   - log() calls: {longest.count('log(')}")
print(f"   - Parentheses: {longest.count('(')}")

print("\n5. Key Differences:")
print("   CPU (SymEngine):")
print("   - Pre-compiled symbolic expressions")
print("   - Expression trees optimized at symbolic level")
print("   - Uses LLVM JIT for fast native code")
print("   - Can evaluate expressions incrementally")
print("   - Caches compiled functions")
print("")
print("   GPU (nvcc):")
print("   - Compiles raw C++ text expressions")
print("   - Must parse entire 16k+ char lines")
print("   - Complex optimization for GPU architecture")
print("   - No incremental compilation")
print("   - Generates different code for each GPU arch")

print("\n6. Why the difference matters:")
print("   - CPU: ~0.001s compilation + fast evaluation")
print("   - GPU: ~60s compilation but parallel evaluation")
print("   - GPU wins when evaluating millions of points")
print("   - CPU wins for small calculations")

# Test with simpler expression
print("\n7. Compilation time vs expression size:")
print("   (Based on our tests)")
print("   - ALCU_PRIME (8.4k chars): ~38s GPU compilation")
print("   - LIQUID (16.4k chars): ~66s GPU compilation")
print("   - BCC_B2 (122k chars): Fails (too complex)")
print(f"   - Ratio: {66/38:.2f}x time for {16.4/8.4:.2f}x chars")
print("   - Suggests non-linear scaling with expression size")