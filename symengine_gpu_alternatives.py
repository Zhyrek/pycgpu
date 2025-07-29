#!/usr/bin/env python
"""Demonstrate better ways to convert SymEngine to GPU code."""

import symengine as se
from symengine import symbols, sin, cos, exp, log, Piecewise
from symengine.lib.symengine_wrapper import ccode
import time

print("SymEngine to GPU Code Generation Alternatives")
print("="*60)

# Create a sample complex expression (similar to what we see in thermodynamics)
print("\n1. SAMPLE EXPRESSION CREATION:")
print("-" * 40)

x, y, T = symbols('x y T')
# Create a thermodynamic-like expression
expr = (x*log(x) + y*log(y) + (1-x-y)*log(1-x-y) + 
        x*y*(1000/T) + x*(1-x-y)*(2000/T) + 
        Piecewise((x**2 * 500/T, x > 0.1), (0, True)) +
        sin(T/1000) * exp(-x) * cos(y))

print(f"Expression type: {type(expr)}")
print(f"Expression length as string: {len(str(expr))} chars")

print("\n2. CURRENT APPROACH (String-based):")
print("-" * 40)
start = time.time()
current_str = str(expr)
print(f"Time to convert to string: {time.time() - start:.6f}s")
print(f"String length: {len(current_str)}")
print(f"First 200 chars: {current_str[:200]}...")

print("\n3. BETTER APPROACH 1: SymEngine ccode()")
print("-" * 40)
start = time.time()
c_code = ccode(expr)
print(f"Time to generate C code: {time.time() - start:.6f}s")
print(f"C code length: {len(c_code)}")
print(f"First 200 chars: {c_code[:200]}...")

print("\n4. BETTER APPROACH 2: Common Subexpression Elimination")
print("-" * 40)
from symengine import cse

# Apply CSE before code generation
start = time.time()
replacements, reduced = cse([expr])
cse_time = time.time() - start

print(f"Time for CSE: {cse_time:.6f}s")
print(f"Number of subexpressions: {len(replacements)}")
print(f"Reduced expression length: {len(str(reduced[0]))}")

# Generate C code for subexpressions
c_subexprs = []
for i, (symbol, subexpr) in enumerate(replacements):
    c_subexprs.append(f"double {ccode(symbol)} = {ccode(subexpr)};")

c_main = f"return {ccode(reduced[0])};"

print(f"\nGenerated C function with CSE:")
print("__device__ double formula_optimized(double x, double y, double T) {")
for subexpr in c_subexprs[:3]:  # Show first 3
    print(f"    {subexpr}")
if len(c_subexprs) > 3:
    print(f"    // ... {len(c_subexprs)-3} more subexpressions")
print(f"    {c_main}")
print("}")

print("\n5. BETTER APPROACH 3: LLVM Backend")
print("-" * 40)
from symengine import lambdify
import numpy as np

# Compile with LLVM backend (what CPU code uses)
start = time.time()
compiled_func = lambdify([x, y, T], [expr], backend='llvm', cse=True)
llvm_time = time.time() - start

print(f"Time to compile with LLVM: {llvm_time:.6f}s")
print(f"Function type: {type(compiled_func)}")

# Test evaluation speed
test_vals = np.array([[0.3, 0.4, 1273.15]])
start = time.time()
result = compiled_func(test_vals.T)
eval_time = time.time() - start
print(f"Evaluation time: {eval_time:.6f}s")
print(f"Result: {result[0][0]:.6f}")

print("\n6. PROBLEMS WITH CURRENT GPU APPROACH:")
print("-" * 40)
print("Current pycalphad GPU generation:")
print("• Converts SymEngine → string → 15+ regex fixes → C code")
print("• No common subexpression elimination")
print("• Manual fixes for Piecewise, parentheses, etc.")
print("• Results in 16k+ character single lines")
print("• nvcc struggles to parse such complex expressions")

print("\n7. PROPOSED IMPROVEMENTS:")
print("-" * 40)
print("Option 1 - Direct ccode() generation:")
print("• Replace str(expr) with ccode(expr)")
print("• Eliminate most regex processing")
print("• Cleaner, more standard C syntax")

print("\nOption 2 - CSE + hierarchical functions:")
print("• Use SymEngine CSE to break expressions")
print("• Generate multiple small device functions")
print("• Main function calls sub-functions")

print("\nOption 3 - LLVM → NVPTX pipeline:")
print("• Use SymEngine LLVM backend")
print("• Target NVPTX directly (if supported)")
print("• Preserve all optimizations")

print("\nOption 4 - Template-based generation:")
print("• Generate C++ expression templates")
print("• Compile-time optimization")
print("• Better type safety")

print("\n8. IMPLEMENTATION PRIORITIES:")
print("-" * 40)
print("1. Replace str() with ccode() in gpu_codegen.py")
print("2. Add CSE processing before code generation")  
print("3. Generate hierarchical device functions")
print("4. Test compilation speed improvements")

print(f"\n9. EXPECTED IMPROVEMENTS:")
print("-" * 40)
print("• Compilation time: 60s → ~5-10s (estimated)")
print("• Code readability: Much better")
print("• Debugging: Easier with smaller functions")
print("• Maintainability: Less regex processing")
print("• Reliability: Fewer manual syntax fixes")