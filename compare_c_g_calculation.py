#!/usr/bin/env python
"""Compare CPU vs GPU c_G calculation to find the exact difference."""

print("COMPARING CPU vs GPU c_G CALCULATION")
print("=" * 60)

print("From the debug output:")
print("CPU iteration 1 (after consolidation): c_G = [0.20879587, -0.20879587]")
print("GPU iteration 1 (after consolidation): c_G = [0.20927135, -0.20927135]")
print(f"Difference: {0.20927135 - 0.20879587:.8f}")

print(f"\n" + "="*60)
print("ANALYZING c_G CALCULATION")

# From CPU debug output at iteration 1:
# [CPU c_G DEBUG] Phase 0 calculation:
#   gradient values: [-19614.94249925 -13154.5657734 ]
#   full_e_matrix diagonal: [3.231945690370953e-05, 3.231945690370958e-05]
#   c_G[0] = 0.20879586717290466
#   c_G[1] = -0.20879586717290466

cpu_gradients = [-19614.94249925, -13154.5657734]
cpu_e_matrix_diag = [3.231945690370953e-05, 3.231945690370958e-05]

print(f"CPU gradients: {cpu_gradients}")
print(f"CPU e_matrix diagonal: {cpu_e_matrix_diag}")

# c_G calculation: c_G[i] = -e_matrix[i,i] * gradient[i+3] (approximately)
cpu_c_g_0 = -cpu_e_matrix_diag[0] * cpu_gradients[0] + cpu_e_matrix_diag[1] * cpu_gradients[1]
cpu_c_g_1 = cpu_e_matrix_diag[0] * cpu_gradients[0] - cpu_e_matrix_diag[1] * cpu_gradients[1]

print(f"CPU c_G[0] calculated: {cpu_c_g_0:.10f}")
print(f"CPU c_G[1] calculated: {cpu_c_g_1:.10f}")

print(f"\nNow I need to find the GPU's corresponding values:")
print(f"- GPU gradients at iteration 1")
print(f"- GPU e_matrix diagonal at iteration 1")
print(f"- GPU c_G calculation details")

print(f"\nFrom GPU debug output, I can see:")
print(f"GPU c_G = [0.20927135, -0.20927135]")
print(f"But I need to find what gradients and e_matrix values led to this.")

print(f"\nThe difference could be in:")
print(f"1. Different gradient values (energy derivatives)")
print(f"2. Different e_matrix values (inverted Hessian)")
print(f"3. Different calculation method")

print(f"\nLet me search for GPU's gradient and e_matrix values at the critical iteration...")

# The key is that both CPU and GPU should have identical:
# - Site fractions: [0.096853, 0.903147] after consolidation
# - Energy gradients at these site fractions
# - Hessian matrix at these site fractions
# - Inverted Hessian (e_matrix)

print(f"\nIf site fractions are identical, then gradients should be identical.")
print(f"If gradients are identical, then Hessian should be identical.")
print(f"If Hessian is identical, then e_matrix should be identical.")
print(f"If e_matrix is identical, then c_G should be identical.")

print(f"\nSo the difference must be in one of these steps.")
print(f"I need to trace through the GPU calculation to find where it diverges.")