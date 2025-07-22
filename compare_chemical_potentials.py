#!/usr/bin/env python
"""Compare chemical potentials from iteration 0."""

# CPU values from trace
cpu_mu = [-25124.6722576, -19902.46889519]

# GPU values from trace (scientific notation)
gpu_mu = [-2.512467225782581e+04, -1.990246889287236e+04]

print("=== ITERATION 0 CHEMICAL POTENTIALS ===")
print("\nComponent 0 (NB):")
print(f"  CPU: {cpu_mu[0]:.12f} J/mol")
print(f"  GPU: {gpu_mu[0]:.12f} J/mol")
diff0 = abs(cpu_mu[0] - gpu_mu[0])
print(f"  Difference: {diff0:.12e} J/mol")
if diff0 > 0.000001:
    print(f"  *** EXCEEDS THRESHOLD OF 0.000001 J ***")

print("\nComponent 1 (TI):")
print(f"  CPU: {cpu_mu[1]:.12f} J/mol")
print(f"  GPU: {gpu_mu[1]:.12f} J/mol")
diff1 = abs(cpu_mu[1] - gpu_mu[1])
print(f"  Difference: {diff1:.12e} J/mol")
if diff1 > 0.000001:
    print(f"  *** EXCEEDS THRESHOLD OF 0.000001 J ***")

print("\n=== ANALYSIS ===")
if diff0 > 0.000001 or diff1 > 0.000001:
    print("FIRST DIVERGENCE FOUND: Chemical potentials diverge after iteration 0")
    print("This occurs BEFORE any phase removal or consolidation")
    print("The divergence is in the equilibrium solver itself")
else:
    print("Chemical potentials are within tolerance")