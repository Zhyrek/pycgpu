# Al-Cu-Fe Multi-Sublattice Divergence Analysis

## Summary of Findings

The CPU and GPU codes diverge when handling the ALCU_ZETA phase with multiple sublattices (9.0, 11.0). The key findings are:

### 1. **Primary Divergence: Different Phase Stability**
- **CPU Result**: Finds 2 stable phases
  - LIQUID: NP=0.5515
  - ALCU_ZETA: NP=0.4485
  - Total GM: -54914.53 J/mol
  
- **GPU Result**: Finds only 1 stable phase
  - ALCU_ZETA: NP=1.0000
  - Total GM: -54907.59 J/mol
  - Difference: 6.94 J/mol

### 2. **Site Ratio Handling is Correct**
The GPU correctly handles the site ratio sum of 20.0 for ALCU_ZETA:
- In the equilibrium matrix Row 5 (system amount constraint), the GPU uses coefficient 20.0 for ALCU_ZETA
- The formulamole calculation returns [9.0 AL, 11.0 CU, 0.0 FE] correctly
- This matches the expected behavior for a phase with sublattices (9.0, 11.0)

### 3. **The Issue: Phase Removal During Iteration**
The GPU removes the LIQUID phase during the equilibrium iteration, while the CPU keeps both phases:
- GPU starts with 3 phases (LIQUID, ALCU_ZETA, LIQUID) from the starting point
- During iteration, the GPU consolidates or removes the LIQUID phases
- The CPU maintains both LIQUID and ALCU_ZETA phases throughout

### 4. **Root Cause Hypothesis**
The divergence likely stems from:
1. Different numerical tolerances or phase removal criteria between CPU and GPU
2. The GPU's handling of the large site ratio sum (20.0) may affect phase stability calculations
3. Possible differences in how the equilibrium matrix is solved when phases have very different site ratio sums (1.0 for LIQUID vs 20.0 for ALCU_ZETA)

### 5. **No Hardcoded Sublattice=1 Assumptions Found**
The GPU code appears to correctly handle multiple sublattices:
- Site ratio sum is correctly calculated as 20.0
- Phase DOF calculations show correct values
- Formulamole calculations are correct

## Next Steps
1. Investigate why the GPU removes the LIQUID phase during iteration
2. Check phase removal criteria and thresholds
3. Compare the equilibrium solution convergence between CPU and GPU
4. Verify gradient and Hessian calculations for multi-sublattice phases