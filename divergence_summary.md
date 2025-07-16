# CPU vs GPU Divergence Analysis

## Summary of Findings

The CPU and GPU calculations diverge significantly for several test conditions, with errors exceeding the 1e-6 threshold.

## Divergent Conditions Found

| Condition | CPU GM (J/mol) | GPU GM (J/mol) | Difference | Status |
|-----------|----------------|----------------|------------|---------|
| T=1000K, X(TI)=0.001 | -49430.51 | -49459.08 | 28.6 | DIVERGED |
| T=1000K, X(TI)=0.010 | -49667.37 | -49673.04 | 5.67 | DIVERGED |
| T=1000K, X(TI)=0.100 | -50390.69 | -50382.30 | 8.39 | DIVERGED |
| T=1000K, X(TI)=0.900 | -46221.57 | -46182.86 | 38.7 | DIVERGED |
| T=1000K, X(TI)=0.990 | -44560.19 | -44437.14 | 123 | DIVERGED |
| T=1000K, X(TI)=0.999 | -44229.52 | -44437.14 | 208 | DIVERGED |
| T=300K, X(TI)=0.500 | -7787.48 | -7780.67 | 6.82 | DIVERGED |

## Detailed Analysis of T=1000K, X(TI)=0.01 Case

### CPU Result
- Final GM: -49667.37 J/mol
- Stable phases: 1 (BCC_A2 only)
- Site fractions: Y(NB)=0.990000, Y(TI)=0.010000
- Converged to single-phase solution

### GPU Result
- Final GM: -49673.04 J/mol  
- Stable phases: 1 (BCC_A2 only)
- Site fractions: Y(NB)=0.990109, Y(TI)=0.009891
- Also converged to single-phase solution

### Key Differences Observed

1. **Site Fraction Differences**: The GPU calculates slightly different equilibrium site fractions:
   - CPU: Y(TI) = 0.010000 (exactly matches input)
   - GPU: Y(TI) = 0.009891 (slightly different)

2. **Convergence Path**: 
   - CPU consolidates two BCC_A2 phases after iteration 1
   - GPU appears to handle the immiscibility gap warning differently

3. **Phase Removal Logic**: The warning "GPU found duplicate phase type (immiscibility gap)" suggests the GPU may handle multiple instances of the same phase differently than CPU.

## Root Causes of Divergence

Based on the analysis, the divergences appear to be caused by:

1. **Phase Consolidation Logic**: The CPU and GPU have different approaches to handling multiple composition sets of the same phase (immiscibility gaps).

2. **Numerical Precision in Matrix Operations**: Small differences in matrix inversion or linear algebra operations accumulate.

3. **Starting Point Differences**: Even with the same initial data, the handling of duplicate phases leads to different iteration paths.

4. **Convergence Criteria**: The exact site fraction Y(TI)=0.010000 in CPU vs Y(TI)=0.009891 in GPU suggests different convergence behavior.

## Recommendations

1. **Investigate Phase Consolidation**: The GPU warning about duplicate phase types needs to be addressed. The consolidation logic should match CPU behavior exactly.

2. **Check Matrix Solver Precision**: Compare the SVD implementation on GPU with the CPU's linear algebra routines.

3. **Verify Site Fraction Optimization**: The GPU should converge to the same site fractions as CPU when given identical conditions.

4. **Add Duplicate Phase Handling**: Implement proper handling of immiscibility gaps in the GPU code to match CPU behavior.