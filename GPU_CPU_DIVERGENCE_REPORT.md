# GPU vs CPU Divergence Analysis Report: Al-Cu-Fe System at X(CU)=0.50

## Executive Summary

This report documents a critical finding where GPU calculations appear to produce more physically correct results than CPU calculations for the Al-Cu-Fe ternary system at specific compositions. The divergence occurs at exactly X(CU)=0.50, where a numerical singularity causes the CPU solver to converge to an unphysical local minimum, while the GPU solver finds the correct global minimum.

## 1. Problem Overview

### 1.1 System Under Study
- **Database**: Al-Cu-Fe.tdb
- **Components**: AL, CU, FE, VA
- **Phases**: LIQUID, FCC_A1, BCC_A2, BCC_B2 (+ others)
- **Problematic Condition**: X(AL)=0.2, X(CU)=0.5, X(FE)=0.3, T=900K

### 1.2 Observed Divergence
- **CPU Gibbs Energy**: -54277.1 J/mol
- **GPU Gibbs Energy**: -54457.5 J/mol
- **Difference**: 180.4 J/mol (GPU finds lower energy state)

### 1.3 Pass Rate Statistics
- Overall pass rate: 91.5% (97/106 conditions)
- 9 failing conditions, all involving specific compositions
- Failures concentrated near X(CU)≈0.5

## 2. Root Cause Analysis

### 2.1 The Singular Matrix Problem

At iteration 1 of the equilibrium solver, the equilibrium matrix becomes nearly singular:

```
Matrix condition number: 4.825e+06 (poorly conditioned)
Determinant: 2.023e-12 (nearly zero)
```

The matrix has near-duplicate rows (rows 1 and 2 differ by only ~1.9e-05), corresponding to phases FCC_A1 and BCC_A2 having nearly identical compositions at this specific bulk composition.

### 2.2 Solver Response to Singularity

When the LU decomposition encounters this singular matrix:

**Iteration 1 Matrix Solution:**
- Input RHS: [-5.60e+04, -5.25e+04, -5.19e+04, -2.11e-01, 9.58e-02, -3.69e-14]
- Output solution: [-2.44e+08, 4.41e+07, 8.20e+07, -9.22e+02, 7.05e+08, -7.05e+08]

The chemical potentials jump from reasonable values (~-97,000) to enormous values (~-244,000,000), indicating complete numerical failure. Both CPU and GPU produce these wrong values initially, but they recover differently.

### 2.3 The BCC_B2 Phase Connection

The singularity only occurs when BCC_B2 phase is included in the phase list:
- **Without BCC_B2**: Perfect CPU/GPU agreement (0.0 J/mol difference)
- **With BCC_B2**: 180.4 J/mol divergence

BCC_B2 has a complex structure:
- 3 sublattices with site fractions (0.5, 0.5, 3.0)
- Even when not stable, its presence affects the matrix conditioning

## 3. Evidence That GPU Is Correct

### 3.1 Thermodynamic Evidence

The GPU finds a **lower Gibbs energy** (-54457.5 vs -54277.1 J/mol), indicating it found the true global minimum while CPU got trapped in a local minimum. In equilibrium calculations, lower Gibbs energy at constant T, P, and composition represents the more stable state.

### 3.2 Physical Continuity Evidence

**Copper content in LIQUID phase across bulk compositions:**

| Bulk X(CU) | CPU LIQUID X(CU) | GPU LIQUID X(CU) | 
|------------|------------------|------------------|
| 0.49       | 0.050            | 0.050            |
| **0.50**   | **0.000**        | **0.051**        |
| 0.51       | 0.053            | 0.053            |

The CPU shows an **unphysical discontinuity**: copper in LIQUID jumps from 5% → 0% → 5.3%, violating the principle of continuous phase composition changes. The GPU shows smooth variation: 5.0% → 5.1% → 5.3%, which is thermodynamically consistent.

### 3.3 Composition Range Analysis

Testing X(CU) from 0.48 to 0.52 reveals:
- **CPU/GPU agree** for X(CU) < 0.499 and X(CU) > 0.501
- **Divergence occurs** only in the narrow range 0.499 ≤ X(CU) ≤ 0.501
- This narrow "singularity window" is characteristic of numerical instability, not physical behavior

### 3.4 Phase Assemblage Consistency

Both CPU and GPU find the same stable phases (LIQUID + FCC_A1), but with different compositions:

**CPU at X(CU)=0.50:**
- LIQUID (40.6%): X(AL)=0.261, **X(CU)=0.000**, X(FE)=0.739
- FCC_A1 (59.4%): X(AL)=0.158, X(CU)=0.842, X(FE)=0.000

**GPU at X(CU)=0.50:**
- LIQUID (43.7%): X(AL)=0.265, **X(CU)=0.051**, X(FE)=0.684
- FCC_A1 (56.3%): X(AL)=0.150, X(CU)=0.849, X(FE)=0.001

The GPU's phase compositions are more consistent with neighboring conditions.

## 4. Technical Details

### 4.1 Numerical Precision Differences

The divergence appears to stem from different handling of the singular matrix:
- Both CPU (LAPACK) and GPU (custom LU solver) initially compute wrong solutions
- CPU converges to a local minimum with X(CU)=0 in LIQUID
- GPU escapes to the global minimum with X(CU)=0.051 in LIQUID

### 4.2 Convergence Paths

**CPU Convergence:**
- Iterations: 5
- Final residual: 6.195e-12
- Converges to discontinuous solution

**GPU Convergence:**
- Iterations: 100
- Final residual: 1.218e-10
- Converges to continuous solution

### 4.3 Matrix Analysis at Singularity

The problematic matrix at iteration 1:
```python
Matrix rows 1 and 2 (nearly identical):
Row 1: [0.1529026, 0.8470974, 2.67e-13, 0, 0, 0] | RHS: -52461.91
Row 2: [0.1529007, 0.8470993, 2.67e-13, 0, 0, 0] | RHS: -51915.23
```

These correspond to phases with nearly identical compositions, causing rank deficiency.

## 5. Broader Implications

### 5.1 Failure Pattern

All 9 failing conditions share similar characteristics:
- Involve specific composition ratios
- Occur when phase compositions become similar
- BCC_B2 presence triggers the instability

### 5.2 Performance Comparison

Despite the numerical challenges:
- **CPU time**: 60.1 seconds
- **GPU time**: 25.4 seconds
- **Speedup**: 2.4x

The GPU is both faster AND more accurate in this case.

## 6. Conclusions

### 6.1 Primary Finding

**The GPU solver produces the physically correct result** at X(CU)=0.50, while the CPU solver converges to an unphysical local minimum. This is demonstrated by:

1. **Thermodynamic correctness**: GPU finds lower Gibbs energy (global minimum)
2. **Physical continuity**: GPU maintains smooth composition variations
3. **Consistency**: GPU results align with neighboring conditions

### 6.2 Root Cause

The divergence is caused by a numerical singularity in the equilibrium matrix when:
- Bulk composition is exactly X(CU)=0.50 (±0.001)
- BCC_B2 phase is included in the calculation
- Multiple phases have similar compositions

### 6.3 Recommendations

1. **Investigate CPU solver robustness**: The CPU solver's convergence to an unphysical state with zero copper in LIQUID suggests it may need improved singular matrix handling.

2. **Consider GPU results as reference**: For conditions near X(CU)=0.50 in the Al-Cu-Fe system, GPU results appear more reliable.

3. **Add singularity detection**: Both solvers could benefit from detecting near-singular matrices and applying regularization techniques.

4. **Document known issues**: Users should be aware that certain composition points may trigger numerical instabilities, particularly when complex phases like BCC_B2 are present.

## 7. Reproducibility

All findings can be reproduced using the test scripts in `/mnt/c/users/scott/Documents/pycalphad/important_tests/`:

- `test_alcufe_8phases_comprehensive_multi.py` - Overall pass rate analysis
- `test_solver_trace.py` - Singularity window identification  
- `test_phase_compositions.py` - Detailed phase composition comparison
- `check_matrix_condition.py` - Matrix conditioning analysis
- `test_bcc_b2_issue.py` - BCC_B2 specific testing

## 8. Summary

This investigation reveals a case where the GPU implementation of pycalphad's equilibrium solver produces more physically reasonable results than the CPU implementation. The issue stems from numerical handling of a singular matrix condition, where the GPU's approach (possibly due to different floating-point precision or algorithmic details) allows it to find the true global minimum while the CPU gets trapped in an unphysical local minimum.

**The key insight**: At exactly X(CU)=0.50, the CPU incorrectly predicts zero copper in the LIQUID phase (discontinuous behavior), while the GPU correctly predicts 5.1% copper (continuous behavior), making this a rare case where the GPU divergence represents the correct physical solution.

---

*Report generated: 2024*  
*System: Al-Cu-Fe ternary system*  
*Software: pycalphad with GPU acceleration*