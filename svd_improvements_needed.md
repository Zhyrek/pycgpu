# SVD Solver Improvements Needed for GPU

## Current Issue
The GPU's custom SVD solver (based on Golub-Reinsch method) struggles with poorly conditioned matrices that arise in single-phase composition constraints, while CPU's LAPACK dgelsd handles them robustly.

## Key Differences Between GPU SVD and LAPACK dgelsd

### 1. **Rank Detection and Truncation**
**LAPACK dgelsd:**
- Uses rank-revealing QR decomposition with column pivoting
- Automatically detects effective rank based on machine precision
- Truncates solution space to well-conditioned subspace

**GPU SVD (current):**
- Basic tolerance check: `tolerance = DBL_EPSILON * D[0] * ncols`
- No rank truncation - tries to use all singular values

**Change needed:**
```c
// After computing singular values in Singular_Value_Decomposition_Solve
int effective_rank = 0;
double rcond = 1e-10;  // Relative condition number threshold
double s_max = D[0];   // Largest singular value

// Determine effective rank
for (i = 0; i < ncols; i++) {
    if (D[i] > rcond * s_max) {
        effective_rank++;
    } else {
        D[i] = 0.0;  // Zero out small singular values
    }
}
```

### 2. **Minimum Norm Solution**
**LAPACK dgelsd:**
- Computes minimum norm least-squares solution
- For underdetermined systems, chooses solution with smallest magnitude

**GPU SVD (current):**
- Direct back-substitution without norm minimization

**Change needed:**
```c
// In back-substitution loop
for (j = 0; j < ncols; j++) {
    s = 0.0;
    if (D[j] != 0.0) {  // Only use non-zero singular values
        for (i = 0; i < nrows; i++) {
            s += U[i * ncols + j] * B[i];
        }
        s /= D[j];
        
        // Apply damping for very small singular values
        double damping = D[j] / (D[j] + tolerance);
        s *= damping;
    }
    for (i = 0; i < ncols; i++) {
        X[i] += s * V[i * ncols + j];
    }
}
```

### 3. **Iterative Refinement**
**LAPACK dgelsd:**
- May use iterative refinement for improved accuracy

**GPU SVD (current):**
- Single-pass solution

**Change needed:**
```c
// After initial solution, add refinement step
__device__ void svd_iterative_refinement(
    double* A, double* X, double* B, double* R,
    int nrows, int ncols, double tolerance
) {
    // Compute residual R = B - A*X
    for (int i = 0; i < nrows; i++) {
        R[i] = B[i];
        for (int j = 0; j < ncols; j++) {
            R[i] -= A[i * ncols + j] * X[j];
        }
    }
    
    // Check if refinement needed
    double residual_norm = 0.0;
    for (int i = 0; i < nrows; i++) {
        residual_norm += R[i] * R[i];
    }
    
    if (sqrt(residual_norm) > tolerance) {
        // Solve A*dX = R for correction
        double dX[MAX_VARIABLES];
        Singular_Value_Decomposition_Solve(A, dX, R, nrows, ncols, tolerance);
        
        // Update solution
        for (int j = 0; j < ncols; j++) {
            X[j] += dX[j];
        }
    }
}
```

### 4. **Scaling and Preconditioning**
**LAPACK dgelsd:**
- May apply column/row scaling for better numerical properties

**GPU SVD (current):**
- No scaling

**Change needed:**
```c
// Before SVD, scale matrix for better conditioning
__device__ void scale_matrix(
    double* A, double* row_scale, double* col_scale,
    int nrows, int ncols
) {
    // Compute row and column norms
    for (int i = 0; i < nrows; i++) {
        row_scale[i] = 0.0;
        for (int j = 0; j < ncols; j++) {
            row_scale[i] = fmax(row_scale[i], fabs(A[i * ncols + j]));
        }
        if (row_scale[i] > 0) row_scale[i] = 1.0 / row_scale[i];
    }
    
    for (int j = 0; j < ncols; j++) {
        col_scale[j] = 0.0;
        for (int i = 0; i < nrows; i++) {
            col_scale[j] = fmax(col_scale[j], 
                               fabs(A[i * ncols + j] * row_scale[i]));
        }
        if (col_scale[j] > 0) col_scale[j] = 1.0 / col_scale[j];
    }
    
    // Apply scaling
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            A[i * ncols + j] *= row_scale[i] * col_scale[j];
        }
    }
}
```

### 5. **Enhanced Tolerance Handling**
**Change needed in solve_equilibrium function:**
```c
// In minimizer.h, modify the linear solve section
if (spec->prescribed_mole_fraction_rhs.size > 0) {
    // For constrained problems, use more robust tolerance
    double matrix_tolerance = 1e-10;  // More aggressive than default
    
    // Check matrix conditioning
    double max_diag = 0.0, min_diag = DBL_MAX;
    for (int i = 0; i < n_dof; i++) {
        double diag_val = fabs(state->equilibrium_matrix[i * state->equilibrium_matrix.ncols + i]);
        max_diag = fmax(max_diag, diag_val);
        min_diag = fmin(min_diag, diag_val);
    }
    
    double approx_cond = max_diag / (min_diag + 1e-16);
    if (approx_cond > 1e12) {
        // Use specialized solver for ill-conditioned system
        matrix_tolerance = 1e-8 * max_diag;
    }
    
    Singular_Value_Decomposition_Solve(
        state->equilibrium_matrix.data,
        state->delta_statevars.data,
        state->equilibrium_rhs.data,
        n_dof, n_dof, matrix_tolerance
    );
}
```

## Implementation Priority

1. **Immediate fix (minimal change):** Implement rank truncation and damping (#1 and #2)
2. **Medium-term:** Add scaling/preconditioning (#4)
3. **Long-term:** Add iterative refinement (#3)

## Testing the Fix

Create a test that specifically targets the problematic X(TI)=0.9, T=600K condition to verify the SVD improvements resolve the convergence issue.