#!/usr/bin/env python
"""Analyze GM difference between CPU and GPU."""

# The values from the test output
cpu_gm = -24602.651695
gpu_gm = -24602.651605
diff = abs(cpu_gm - gpu_gm)

print(f"CPU GM: {cpu_gm:.9f}")
print(f"GPU GM: {gpu_gm:.9f}")
print(f"Difference: {diff:.9f}")
print(f"Relative difference: {diff/abs(cpu_gm)*100:.7f}%")

# Check if this could be due to different final states
print("\nPossible causes:")
print("1. Different final site fractions")
print("2. Different chemical potentials") 
print("3. Different numerical precision in calculations")
print("4. Different handling of removed phases")

# The chemical potential differences
mu_nb_cpu = -25134.195682
mu_nb_gpu = -25134.195732
mu_ti_cpu = -19818.755811  
mu_ti_gpu = -19818.755365

print(f"\nChemical potential differences:")
print(f"μ(NB): {abs(mu_nb_cpu - mu_nb_gpu):.9f}")
print(f"μ(TI): {abs(mu_ti_cpu - mu_ti_gpu):.9f}")

# These differences in chemical potentials suggest slightly different final equilibrium states