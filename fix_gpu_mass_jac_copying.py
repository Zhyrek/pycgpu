#!/usr/bin/env python
"""Fix for GPU mass_jac copying with dependent site fractions."""

print("GPU MASS_JAC COPYING FIX")
print("=" * 60)

print("\nPROBLEM:")
print("- formulamole_grad outputs gradients for ALL site fractions (including dependent)")
print("- But workspace DOF only tracks INDEPENDENT site fractions")
print("- This causes index mismatch when copying gradients")

print("\nEXAMPLE:")
print("BCC_A2 phase with Y(NB) and Y(TI) where Y(TI) = 1 - Y(NB)")
print("- formulamole_grad output: 2 components × 5 DOF = 10 values")
print("  Row 0 (NB): [0, 0, 0, 1.0, 0] - gradient w.r.t [N, P, T, Y(NB), Y(TI)]")
print("  Row 1 (TI): [0, 0, 0, -1.0, 0] - gradient w.r.t [N, P, T, Y(NB), Y(TI)]")
print("- Workspace DOF: [N, P, T, Y(NB)] - only 4 values!")

print("\nCURRENT COPYING:")
print("- Uses pr->phase_dof = 2 as stride")
print("- Copies position 8 (row 1, col 3 with 5-column stride)")
print("- But workspace expects 4-column stride")

print("\nFIX:")
print("When copying site fraction gradients, need to:")
print("1. Skip columns for dependent site fractions")
print("2. Use correct column mapping")

print("\nPROPOSED CODE CHANGE in minimizer.h:")
print("""
// In the mass_jac copying section around line 886:
for (int model_col = 0; model_col < pr->num_statevars + pr->phase_dof; model_col++) {
    int workspace_col;
    if (model_col < pr->num_statevars) {
        // State variable columns - unchanged
        workspace_col = /* existing mapping */;
    } else {
        // Site fraction column
        int site_frac_idx = model_col - pr->num_statevars;
        
        // CRITICAL FIX: Skip dependent site fractions
        // For single sublattice, last site fraction is dependent
        if (pr->phase_dof == spec->num_components - 1 && 
            site_frac_idx == pr->phase_dof - 1) {
            // This is the dependent site fraction - skip it
            continue;
        }
        
        // Map to workspace column
        workspace_col = spec->num_statevars + site_frac_idx;
    }
    
    // Copy the gradient value
    csst->mass_jac[comp_idx * csst->mass_jac_cols + workspace_col] = 
        temp_mass_jac[nonvacant_idx * (pr->num_statevars + pr->phase_dof) + model_col];
}
""")

print("\nALTERNATIVE FIX:")
print("Modify the indexing when accessing temp_mass_jac to account")
print("for the stride difference between model DOF and workspace DOF.")