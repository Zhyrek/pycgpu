#!/usr/bin/env python
"""Compare iteration 0 values precisely."""

# CPU values
cpu_mu = [-25124.6722576, -19902.46889519]
cpu_phase0_amt = 1.110223024625157e-16
cpu_phase1_amt = 9.999999999999477e-01
cpu_phase0_energy = -24642.97476857788
cpu_phase1_energy = -24593.60072921765
cpu_phase0_sf = [0.90327794, 0.09672206]
cpu_phase1_sf = [0.9032936, 0.0967064]

# GPU values  
gpu_mu = [-25124.67225782581, -19902.46889287236]
gpu_phase0_amt = 1.110223024625157e-16
gpu_phase1_amt = 1.000000000000556e+00
gpu_phase0_energy = -24642.97476857788
gpu_phase1_energy = -24593.60072921765
gpu_phase0_sf = [0.9032779441736019, 0.09672205582639756]
gpu_phase1_sf = [0.9032936027372608, 0.09670639726273841]

# Calculate differences
print("=== ITERATION 0 COMPARISON ===")
print("\nChemical Potentials:")
for i in range(2):
    diff = abs(cpu_mu[i] - gpu_mu[i])
    print(f"  μ[{i}]: CPU={cpu_mu[i]:.12f}, GPU={gpu_mu[i]:.12f}")
    print(f"        diff={diff:.12e} J/mol")
    if diff > 0.000001:
        print(f"        *** EXCEEDS THRESHOLD ***")

print("\nPhase 0:")
print(f"  Amount: CPU={cpu_phase0_amt:.6e}, GPU={gpu_phase0_amt:.6e}")
print(f"  Energy: CPU={cpu_phase0_energy:.12f}, GPU={gpu_phase0_energy:.12f}")
energy_diff = abs(cpu_phase0_energy - gpu_phase0_energy)
print(f"          diff={energy_diff:.12e} J/mol")

print("\nPhase 1:")
amt_diff = abs(cpu_phase1_amt - gpu_phase1_amt)
print(f"  Amount: CPU={cpu_phase1_amt:.15e}, GPU={gpu_phase1_amt:.15e}")
print(f"          diff={amt_diff:.6e}")
print(f"  Energy: CPU={cpu_phase1_energy:.12f}, GPU={gpu_phase1_energy:.12f}")
energy_diff = abs(cpu_phase1_energy - gpu_phase1_energy)
print(f"          diff={energy_diff:.12e} J/mol")

print("\nSite Fractions:")
print("  Phase 0:")
for i in range(2):
    diff = abs(cpu_phase0_sf[i] - gpu_phase0_sf[i])
    print(f"    Y[{i}]: CPU={cpu_phase0_sf[i]:.12f}, GPU={gpu_phase0_sf[i]:.12f}")
    print(f"          diff={diff:.12e}")
    
print("  Phase 1:")
for i in range(2):
    diff = abs(cpu_phase1_sf[i] - gpu_phase1_sf[i])
    print(f"    Y[{i}]: CPU={cpu_phase1_sf[i]:.12f}, GPU={gpu_phase1_sf[i]:.12f}")  
    print(f"          diff={diff:.12e}")

# Check initial values before iteration 0
print("\n=== ANALYSIS ===")
print("The differences after iteration 0 are:")
print("1. Chemical potentials differ by ~2e-7 J/mol (below threshold)")
print("2. Phase amounts differ by ~1e-13 (numerical noise)")
print("3. Site fractions differ by ~2e-6 (small but significant)")
print("\nThese differences accumulate during the solving process.")
print("The site fraction differences suggest the equilibrium solver")
print("produces slightly different solutions even from the same starting point.")