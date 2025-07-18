# GPU Solver Convergence Issue - Root Cause Analysis

## Problem Summary
The GPU solver fails to converge for X(TI)=0.9 at T=600K, maintaining a mass residual of ~0.003 instead of <1e-12.

## Root Cause
The equilibrium matrix is poorly conditioned due to scaling issues:

1. **Large Hessian values**: The energy Hessian has values ~50,000 J/mol
2. **Tiny inverse values**: e_matrix = inverse(phase_matrix) has values ~3e-05
3. **Tiny c_component**: c_component = mass_jac * e_matrix gives ~3e-05
4. **Ineffective constraints**: The mole fraction constraint row gets coefficients ~3e-05, making it ineffective at enforcing X(TI)=0.9

## Detailed Analysis

### The Newton Update Equation
For constrained optimization, the Newton update solves:
```
[H    B^T] [Δy]   [-∇G]
[B    0  ] [Δλ] = [-g  ]
```

Where:
- H = Hessian of Lagrangian  
- B = Constraint Jacobian (dX/dY)
- g = Constraint residual (X - X_target)

### The GPU Implementation
The GPU code computes:
- delta_y = c_G + Σ(c_component * Δμ)
- Where c_G = e_matrix * gradient
- And c_component = mass_jac * e_matrix

### The Scaling Problem
With Hessian ~50,000:
- e_matrix ~1/50,000 = 2e-05
- c_component ~1.0 * 2e-05 = 2e-05
- Constraint row coefficient ~2e-05

This makes the constraint row:
```
2e-05 * Δμ_NB + 2e-05 * Δμ_TI + ... = -0.003
```

The small coefficients (2e-05) mean large changes in chemical potentials are needed to correct the 0.003 composition error.

### Why c_G and Chemical Potential Terms Cancel
- c_G ≈ +0.209 (from energy gradient)
- Chemical potential term ≈ -0.209
- Net delta_y ≈ 3e-07 (too small to correct the error)

The cancellation occurs because the system is near a KKT point for the energy, but not for the constraint.

## Solution Options

1. **Scale the Hessian**: Divide by a characteristic energy scale before inversion
2. **Direct constraint enforcement**: Add explicit constraint correction terms
3. **Improved matrix conditioning**: Use better numerical methods for the linear solve
4. **Separate constraint handling**: Solve constraints separately from energy minimization

## Verification
The CPU solver converges in 6 iterations because it likely:
- Uses different scaling
- Has better conditioned matrices
- Employs more robust numerical methods

The GPU implementation needs similar improvements to achieve convergence.