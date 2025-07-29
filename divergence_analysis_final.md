# GPU/CPU Divergence Analysis: Al-Cu-Fe with ALCU_ZETA Phase

## Executive Summary

The divergence between CPU and GPU calculations for the Al-Cu-Fe system occurs at **iteration 0** of the equilibrium solver. The GPU removes one of the LIQUID phases while the CPU keeps both LIQUID and ALCU_ZETA phases stable.

## Root Cause

The divergence is caused by **poor matrix conditioning** (condition number 4.16e+19) when phases have very different site ratio sums:
- LIQUID phases: site ratio sum = 1.0
- ALCU_ZETA phase: site ratio sum = 20.0

## Detailed Findings

### 1. No Hardcoded Sublattice Assumptions
The GPU code correctly handles multiple sublattices:
- ✓ Site ratio sum correctly calculated as 20.0 for ALCU_ZETA
- ✓ Formulamole correctly returns [9.0 AL, 11.0 CU, 0.0 FE]
- ✓ Phase DOF calculations are correct

### 2. The Divergence Point

At iteration 0, the equilibrium matrix has this structure:

```
Row 0: Phase 0 (LIQUID)     - coefficients [0.768, 0.043, ...]
Row 1: Phase 1 (ALCU_ZETA)  - coefficients [9.0, 11.0, ...]  ← 10x larger
Row 2: Phase 2 (LIQUID)     - coefficients [0.667, 0.0, ...]
Row 3: X_CU constraint
Row 4: X_FE constraint  
Row 5: System amount        - coefficients [1.0, 20.0, 1.0]   ← 20x scaling!
```

The solution produces:
- Δφ₀ = +15.9 (LIQUID 1 increases)
- Δφ₁ = +0.12 (ALCU_ZETA slightly increases)
- Δφ₂ = -8.75 (LIQUID 2 removed!)

### 3. Why GPU Removes Phase 2

The large coefficient (20.0) in the system amount constraint causes:
1. ALCU_ZETA to be weighted 20x more than LIQUID phases
2. The linear solver to produce extreme solutions
3. Small LIQUID phases to be eliminated to satisfy constraints

### 4. CPU vs GPU Difference

The CPU likely handles this better through:
- Different linear algebra routines
- Implicit scaling or normalization
- Different numerical tolerances

## Recommended Fix

The GPU code should normalize phase amounts in the equilibrium system:

1. **Use mole fractions (NP) instead of phase amounts** in the system constraint
2. **Or divide by site_ratio_sum** when constructing the matrix:
   ```
   Row 5: [1.0/1.0, 1.0/20.0, 1.0/1.0] = [1.0, 0.05, 1.0]
   ```
3. **Pre-scale matrix rows** to improve conditioning

This would prevent the extreme solutions that incorrectly remove phases when site ratio sums differ significantly.