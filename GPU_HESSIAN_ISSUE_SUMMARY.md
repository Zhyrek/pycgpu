# GPU Hessian Issue Summary

## Problem Description
The GPU equilibrium solver produces incorrect results compared to the CPU:
- CPU: GM = -49763 J/mol (converges to 60% NB, 40% TI)
- GPU: GM = -110429 J/mol (converges to ~16% NB, 84% TI)

## Root Cause Analysis

### 1. Hessian Values Differ by ~2.6x
- CPU hessian at [3,3]: ~1.36e+04
- GPU hessian at [3,3]: ~3.50e+04
- Ratio: GPU/CPU ≈ 2.6

### 2. Both CPU and GPU compute hessian of G (not GM)
- Confirmed by checking model expressions and code
- G = (Y_NB + Y_TI) * g(...) where g(...) is the core energy expression
- GM = g(...) without the site fraction sum factor

### 3. Cascade Effects
The larger GPU hessian values lead to:
- Smaller inverted matrix (full_e_matrix) values
- Different c_G values with opposite signs
- Different mole fraction constraint RHS values
- Solver converges to a different solution

## Investigation Summary

### What Was Checked:
1. **DOF format**: Fixed - both now use workspace DOF [N, P, T, Y1, Y2]
2. **Variable indexing**: Fixed - site fractions correctly mapped
3. **Mass Jacobian**: Fixed - now matches CPU with [0,1] for TI component
4. **Gradient indexing**: Fixed - using workspace indices directly
5. **Site fraction normalization**: Verified - sums to 1.0 as expected
6. **Symbolic expressions**: Both differentiate model.G

### Remaining Mystery:
Why does the GPU generate a hessian that's 2.6x larger when both should be computing d²G/dx²?

## Potential Solutions to Investigate

### 1. Check Symbolic Differentiation
The GPU uses its own symbolic differentiation code. There might be differences in:
- How Piecewise expressions are handled
- How the (Y_NB + Y_TI) factor is differentiated
- Simplification of expressions

### 2. Verify Generated Code
Compare the actual generated C code between CPU and GPU for a simple test case.

### 3. Alternative Approach
Instead of fixing the hessian generation, consider:
- Using the CPU's phase record factory to generate functions for GPU
- Implementing a compatibility layer to ensure identical symbolic expressions

## Recommendation
The issue appears to be in the symbolic differentiation or code generation phase. The GPU's `_nb_formulahess_from_model` function might be handling the differentiation differently than the CPU's `build_functions` approach.

A detailed comparison of the symbolic expressions before code generation would be the next logical step.