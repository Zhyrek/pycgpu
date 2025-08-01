# FCC Normalization Issue Analysis

## Problem
When testing Au-Bi system with LIQUID + FCC_A1 phases:
- CPU finds: FCC_A1 = 89.1%, LIQUID = 10.9%
- GPU finds: FCC_A1 = 93.9%, LIQUID = 6.1%
- Energy difference: 44.67 J/mol (GPU lower)

## Root Cause
The GPU is incorrectly normalizing energies for multi-sublattice phases:
- FCC_A1 has 2 sublattices with site ratios (1.0, 1.0), total = 2.0
- LIQUID has 1 sublattice with site ratio (1.0), total = 1.0
- GPU appears to be dividing FCC_A1 energy by 2, making it more stable

## Key Finding
User stated: "I think the issue was the CPU does NOT normalize the energy value as given to the equilibrium matrix, but the GPU does."

## Where to Look
1. The compilation errors when trying to fix `write_row_fixed_mole_fraction` suggest the issue is in dynamically generated code
2. Both CPU and GPU use `per_formula_unit=True` for moles calculation
3. The normalization must be happening in the GPU's energy calculation or equilibrium matrix construction

## Attempted Fix
Tried to remove normalization_factor from `write_row_fixed_mole_fraction` in minimizer.h, but this caused compilation errors in generated CUDA code.

## Next Steps
Need to find where the GPU code is normalizing by site ratios when it shouldn't be. Possible locations:
1. In the generated energy functions themselves
2. In how phase amounts are handled in the equilibrium matrix
3. In the GPU-specific code generation that creates the energy functions