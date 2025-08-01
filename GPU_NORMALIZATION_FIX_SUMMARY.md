# GPU Normalization Fix Summary

## Original Issue
The user reported: "the CPU does NOT normalize the energy value as given to the equilibrium matrix, but the GPU does"

## Root Causes Found

### 1. Energy normalization in equilibrium matrix (Fixed in minimizer.h)
- GPU was dividing by `moles_normalization` (sum of site ratios) in `write_row_fixed_mole_fraction`
- CPU doesn't do this normalization
- **Fix**: Set `normalization_factor = 1.0` instead of `moles_normalization`

### 2. Final GM calculation using G instead of GM (Fixed in gpu_codegen.py)
- GPU was using G (per formula unit) from `formulaobj` for final GM calculation
- CPU expects GM (per mole of atoms)
- For multi-sublattice phases: GM = G / (sum of site ratios)
- **Fix**: Divide by `moles_per_formula_unit` when calculating final GM

## Test Results

### AU2BI_C15 + LIQUID (no vacancy, site ratios 3.0 + 1.0)
- Before fix: GM difference = 37282.49 J/mol
- After fix: GM difference = 0.00 J/mol ✓
- Phase amounts match perfectly

### FCC_A1 + LIQUID (with vacancy, site ratios 2.0 + 1.0)
- Before fix: GM difference = 44.67 J/mol, very different phase amounts
- After fix: GM difference = 44.67 J/mol, still different phase amounts
- GPU finds more FCC_A1 than CPU in all cases

## Remaining Issue
The GPU equilibrium solver finds different phase amounts when vacancy is present. This is not a normalization issue but likely related to:
1. How vacancy contributes to mass balance
2. Numerical differences in the equilibrium solver
3. Possible differences in how constraints are handled

## Code Changes
1. `/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h`:
   - Changed `normalization_factor` from `moles_normalization_cs` to `1.0`

2. `/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py`:
   - Added conversion from G to GM in final energy calculation
   - Divides by sum of phase compositions (moles per formula unit)