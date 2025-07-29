#!/usr/bin/env python
"""Test improved SymEngine method on gradient and Hessian expressions."""

from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
import pycalphad.variables as v
import symengine as se
from symengine import cse, diff
from symengine.lib.symengine_wrapper import ccode
import time

# Load database and create workspace for LIQUID phase
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing Improved Method on Gradient and Hessian Expressions")
print("="*80)

# Create workspace and model for LIQUID
wks = Workspace(db, components, ['LIQUID'], conditions, verbose=False)
model = Model(db, components, 'LIQUID')

# Get the free energy expression
free_energy = model.ast
print(f"Free energy expression length: {len(str(free_energy))} chars")

# Get the state variables for differentiation
# Extract the actual variables present in the free energy expression
free_vars = free_energy.free_symbols
actual_vars = list(free_vars)
print(f"Variables found in free energy expression: {[str(v) for v in actual_vars]}")

print(f"State variables found in expression: {len(actual_vars)}")
for i, var in enumerate(actual_vars):
    print(f"  {i}: {var}")

print("\n" + "="*80)
print("EXTRACTING GRADIENT EXPRESSIONS")
print("="*80)

# Compute gradient (first derivatives)
print("Computing gradient...")
start_time = time.time()
gradient_exprs = []
for var in actual_vars:
    grad_expr = diff(free_energy, var)
    gradient_exprs.append(grad_expr)
gradient_time = time.time() - start_time

print(f"Gradient computation time: {gradient_time:.6f}s")
print(f"Number of gradient expressions: {len(gradient_exprs)}")

# Analyze gradient complexity
grad_lengths = [len(str(expr)) for expr in gradient_exprs]
print(f"Gradient expression lengths: {grad_lengths}")
print(f"Longest gradient: {max(grad_lengths)} chars")
print(f"Total gradient chars: {sum(grad_lengths)} chars")

print("\n" + "="*80)
print("EXTRACTING HESSIAN EXPRESSIONS") 
print("="*80)

# Compute Hessian (second derivatives) - this is the expensive part!
print("Computing Hessian (this may take a moment)...")
start_time = time.time()
hessian_exprs = []
for i, var1 in enumerate(actual_vars):
    hessian_row = []
    for j, var2 in enumerate(actual_vars):
        if i <= j:  # Only compute upper triangle (symmetric)
            hess_expr = diff(free_energy, var1, var2)
            hessian_row.append(hess_expr)
        else:
            hessian_row.append(None)  # Use symmetry
    hessian_exprs.append(hessian_row)
hessian_time = time.time() - start_time

print(f"Hessian computation time: {hessian_time:.6f}s")
print(f"Hessian matrix size: {len(actual_vars)}x{len(actual_vars)}")

# Analyze Hessian complexity
hess_lengths = []
for i, row in enumerate(hessian_exprs):
    for j, expr in enumerate(row):
        if expr is not None:
            hess_lengths.append(len(str(expr)))

print(f"Number of unique Hessian elements: {len(hess_lengths)}")
print(f"Hessian expression lengths: min={min(hess_lengths)}, max={max(hess_lengths)}, avg={sum(hess_lengths)/len(hess_lengths):.0f}")
print(f"Longest Hessian element: {max(hess_lengths)} chars")
print(f"Total Hessian chars: {sum(hess_lengths)} chars")

print("\n" + "="*80)
print("TESTING IMPROVED METHOD ON GRADIENT")
print("="*80)

# Test improved method on the longest gradient expression
longest_grad_idx = grad_lengths.index(max(grad_lengths))
longest_grad = gradient_exprs[longest_grad_idx]
longest_grad_var = actual_vars[longest_grad_idx]

print(f"Testing on longest gradient: d/d{longest_grad_var} ({max(grad_lengths)} chars)")

# Apply CSE to gradient
start_time = time.time()
grad_replacements, grad_reduced = cse([longest_grad])
grad_cse_time = time.time() - start_time

print(f"Gradient CSE time: {grad_cse_time:.6f}s")
print(f"Gradient subexpressions found: {len(grad_replacements)}")
print(f"Gradient reduced length: {len(str(grad_reduced[0]))} chars")

# Generate C code for gradient
start_time = time.time()
grad_c_code = f"__device__ double grad_{longest_grad_var}_liquid(double T, double P, double N, double* site_fractions) {{\n"
for symbol, subexpr in grad_replacements:
    grad_c_code += f"    double {ccode(symbol)} = {ccode(subexpr)};\n"
grad_c_code += f"    return {ccode(grad_reduced[0])};\n"
grad_c_code += "}"
grad_codegen_time = time.time() - start_time

print(f"Gradient C code generation time: {grad_codegen_time:.6f}s")
print(f"Generated function length: {len(grad_c_code)} chars")

print("\n" + "="*80)
print("TESTING IMPROVED METHOD ON HESSIAN")
print("="*80)

# Test improved method on the longest Hessian expression
longest_hess = None
longest_hess_length = 0
longest_hess_indices = (0, 0)

for i, row in enumerate(hessian_exprs):
    for j, expr in enumerate(row):
        if expr is not None:
            expr_len = len(str(expr))
            if expr_len > longest_hess_length:
                longest_hess_length = expr_len
                longest_hess = expr
                longest_hess_indices = (i, j)

var1 = actual_vars[longest_hess_indices[0]]
var2 = actual_vars[longest_hess_indices[1]]
print(f"Testing on longest Hessian: d²/d{var1}d{var2} ({longest_hess_length} chars)")

# Apply CSE to Hessian
start_time = time.time()
hess_replacements, hess_reduced = cse([longest_hess])
hess_cse_time = time.time() - start_time

print(f"Hessian CSE time: {hess_cse_time:.6f}s")
print(f"Hessian subexpressions found: {len(hess_replacements)}")
print(f"Hessian reduced length: {len(str(hess_reduced[0]))} chars")

# Generate C code for Hessian
start_time = time.time()
hess_c_code = f"__device__ double hess_{var1}_{var2}_liquid(double T, double P, double N, double* site_fractions) {{\n"
for symbol, subexpr in hess_replacements:
    hess_c_code += f"    double {ccode(symbol)} = {ccode(subexpr)};\n"
hess_c_code += f"    return {ccode(hess_reduced[0])};\n"
hess_c_code += "}"
hess_codegen_time = time.time() - start_time

print(f"Hessian C code generation time: {hess_codegen_time:.6f}s")
print(f"Generated function length: {len(hess_c_code)} chars")

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save results to file
with open('gradient_hessian_codegen_test.txt', 'w') as f:
    f.write("Gradient and Hessian Code Generation Test Results\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"DATABASE: Al-Cu-Fe.tdb\n")
    f.write(f"PHASE: LIQUID\n")
    f.write(f"Free energy expression: {len(str(free_energy))} chars\n\n")
    
    f.write("GRADIENT ANALYSIS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Number of gradient expressions: {len(gradient_exprs)}\n")
    f.write(f"Gradient lengths: {grad_lengths}\n")
    f.write(f"Longest gradient: {max(grad_lengths)} chars\n")
    f.write(f"Total gradient complexity: {sum(grad_lengths)} chars\n\n")
    
    f.write("HESSIAN ANALYSIS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Hessian matrix size: {len(actual_vars)}x{len(actual_vars)}\n")
    f.write(f"Unique Hessian elements: {len(hess_lengths)}\n")
    f.write(f"Hessian lengths: min={min(hess_lengths)}, max={max(hess_lengths)}, avg={sum(hess_lengths)/len(hess_lengths):.0f}\n")
    f.write(f"Longest Hessian element: {max(hess_lengths)} chars\n")
    f.write(f"Total Hessian complexity: {sum(hess_lengths)} chars\n\n")
    
    f.write("IMPROVED METHOD RESULTS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Gradient CSE: {len(grad_replacements)} subexpressions, reduced to {len(str(grad_reduced[0]))} chars\n")
    f.write(f"Hessian CSE: {len(hess_replacements)} subexpressions, reduced to {len(str(hess_reduced[0]))} chars\n\n")
    
    f.write("LONGEST GRADIENT C CODE:\n")
    f.write("-" * 40 + "\n")
    f.write(grad_c_code + "\n\n")
    
    f.write("LONGEST HESSIAN C CODE:\n")
    f.write("-" * 40 + "\n")
    f.write(hess_c_code + "\n\n")
    
    f.write("PERFORMANCE COMPARISON:\n")
    f.write("-" * 40 + "\n")
    f.write("If we had used the current string method:\n")
    f.write(f"• Gradient: One {max(grad_lengths)}-char expression\n")
    f.write(f"• Hessian: One {max(hess_lengths)}-char expression\n")
    f.write("• Would likely cause nvcc compilation issues\n\n")
    
    f.write("With improved CSE method:\n")
    f.write(f"• Gradient: {len(grad_replacements)} subexpressions + {len(str(grad_reduced[0]))}-char main expr\n")
    f.write(f"• Hessian: {len(hess_replacements)} subexpressions + {len(str(hess_reduced[0]))}-char main expr\n")
    f.write("• Much more manageable for nvcc compilation\n")
    f.write("• Better GPU optimization potential\n")

print("Results saved to: gradient_hessian_codegen_test.txt")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Free energy complexity:")
print(f"  Original: {len(str(free_energy))} chars")
print(f"  Gradient: {sum(grad_lengths)} total chars, max {max(grad_lengths)} chars")
print(f"  Hessian: {sum(hess_lengths)} total chars, max {max(hess_lengths)} chars")

print("\nImproved method effectiveness:")
print(f"  Gradient CSE: {len(grad_replacements)} subexpressions")
print(f"  Hessian CSE: {len(hess_replacements)} subexpressions")
print("  Both should compile much faster on GPU!")

print(f"\nComputation times:")
print(f"  Gradient computation: {gradient_time:.3f}s")
print(f"  Hessian computation: {hessian_time:.3f}s")
print(f"  Gradient CSE: {grad_cse_time:.3f}s") 
print(f"  Hessian CSE: {hess_cse_time:.3f}s")