# CPU vs GPU Matrix Structure Difference

## Key Discovery
The CPU and GPU use different equilibrium matrix structures!

### CPU Matrix Structure (2 phases):
```
Variables: [μ_NB, μ_TI, NP_0, NP_1]
Row 0 (Phase 0): Gibbs energy minimization
Row 1 (Phase 1): Gibbs energy minimization  
Row 2 (X(TI)=0.9): -3.4e-05, +3.4e-05, -1.7e-03, +7.5e-03
Row 3 (N=1): System amount constraint
```

The phase amounts (NP) are explicit variables with larger coefficients!

### GPU Matrix Structure (2 phases):
```
Variables: [μ_NB, μ_TI, NP_0, NP_1]
Row 0 (Phase 0): Gibbs energy minimization
Row 1 (Phase 1): Gibbs energy minimization
Row 2 (X(TI)=0.9): -3.4e-05, +3.4e-05, -1.7e-03, +7.5e-03  
Row 3 (N=1): System amount constraint
```

Wait, they look similar for 2 phases!

### After Consolidation to 1 Phase:

**CPU (3x3 matrix):**
```
Variables: [μ_NB, μ_TI, NP_0]
Row 0 (Phase 0): Energy minimization
Row 1 (X(TI)=0.9): -3.2e-05, +3.2e-05, 0.0
Row 2 (N=1): System amount
```

**GPU (3x3 matrix):**
```
Variables: [μ_NB, μ_TI, Y_NB]  ← Different!
Row 0 (Phase 0): Energy minimization
Row 1 (X(TI)=0.9): -3.2e-05, +3.2e-05, ???
Row 2 (N=1): System amount
```

## The Critical Difference

For single phase:
- **CPU**: Phase amount NP is still a variable (but fixed at 1.0)
- **GPU**: Site fraction Y(NB) is the variable instead of NP

This explains why:
1. CPU constraint row has 0.0 for the phase amount column
2. GPU constraint row needs to use c_component which is tiny
3. CPU can enforce constraints more directly
4. GPU struggles with the indirect constraint enforcement

## Why This Matters

The CPU's approach separates:
- Phase stability (through chemical potentials)
- Phase amounts (explicit variables)
- Site fractions (internal to each phase)

The GPU's approach mixes:
- Phase stability and site fractions together
- This coupling through c_component causes scaling issues

## Solution

The GPU needs to either:
1. Keep phase amounts as explicit variables like CPU
2. Improve the scaling of c_component
3. Use a different constraint enforcement strategy