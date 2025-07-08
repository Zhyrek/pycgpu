# CPU vs GPU Solver Iteration 0 Comparison

## Key Differences Found

### 1. Chemical Potentials
- **CPU**: [-51884.59456529, -46673.1890193]
- **GPU**: [-98530.97531978, -46673.1890193]
- **Issue**: First chemical potential is drastically different (almost 2x)

### 2. Site Fractions Storage
- **CPU Phase 0**: [0.50000752, 0.49999248]
- **GPU Phase 0**: [1.0e-14, 0.50847458]
- **Issue**: GPU is storing site fractions incorrectly

### 3. State Variables Storage
- **CPU**: [1.0, 101325.0, 1000.0] (N, P, T)
- **GPU Phase 0**: [1000.0, 0.50847458, 0.49152542] (T, Y(NB), Y(TI))
- **Issue**: GPU is mixing state variables with site fractions

### 4. Phase Compositions
- **GPU Phase 1**: [0.0, 0.0] with phase_comp_sum = 0.0
- **CPU Phase 1**: Valid compositions with phase_comp_sum = 1.0
- **Issue**: GPU phase compositions are not being calculated correctly

### 5. Phase Amount Changes
- **CPU**: 5.738e-10
- **GPU**: 1.736e-10
- **Issue**: Different convergence rates

### 6. Largest Y Change
- **CPU**: 8.467e-03
- **GPU**: 1.0e-14
- **Issue**: GPU is not updating site fractions properly

## Root Causes Identified

1. **DOF Array Format Mismatch**: The GPU is using Model format (T, Y1, Y2...) while the CPU solver expects Workspace format (N, P, T, Y1, Y2...)

2. **Chemical Potential Calculation Error**: The first chemical potential is being calculated incorrectly, likely due to the DOF format issue affecting the energy calculations.

3. **Phase Composition Calculation**: Phase 1 has zero compositions, suggesting the formulamole functions are not working correctly for that phase.

## Next Steps

1. Fix the DOF array format to match CPU expectations
2. Ensure chemical potentials are calculated correctly
3. Debug why phase compositions are zero for phase 1
4. Verify site fraction updates are happening correctly