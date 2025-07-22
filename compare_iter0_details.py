#!/usr/bin/env python
"""Compare iteration 0 details between CPU and GPU."""

# From the debug output
print("=== ITERATION 0 COMPARISON ===")
print("\nPhase amounts:")
print("  Phase 0: CPU=1.110223024625157e-16, GPU=1.110223024625157e-16")
print("           (identical)")
print("  Phase 1: CPU=9.999999999999477e-01, GPU=1.000000000000556e+00")
print("           diff=1.083e-13")

print("\nSite fractions:")
print("  Phase 0:")
print("    CPU: [0.90327794, 0.09672206]")
print("    GPU: [0.9032779441736019, 0.09672205582639756]")
print("    Y[0] diff: 4.17e-09")
print("    Y[1] diff: 4.17e-09")

print("  Phase 1:")
print("    CPU: [0.9032936, 0.0967064]")
print("    GPU: [0.9032936027372608, 0.09670639726273841]")
print("    Y[0] diff: 2.74e-09")
print("    Y[1] diff: 2.74e-09")

print("\nChemical potentials:")
print("  μ[0] (NB): CPU=-25124.6722576, GPU=-25124.67225782581")
print("             diff=2.26e-07 J/mol")
print("  μ[1] (TI): CPU=-19902.46889519, GPU=-19902.46889287236")
print("             diff=2.32e-06 J/mol *** EXCEEDS THRESHOLD ***")

print("\n=== ANALYSIS ===")
print("The differences after iteration 0 suggest:")
print("1. The equilibrium solver produces slightly different solutions")
print("2. Site fraction differences are ~2-4e-09 (very small)")
print("3. Phase amount differences are ~1e-13 (numerical noise)")
print("4. But chemical potential μ(TI) differs by 2.32e-06 J/mol")
print("\nThe chemical potential is calculated from the gradient of the")
print("Gibbs energy with respect to component amounts. Even tiny")
print("differences in site fractions can lead to larger differences")
print("in chemical potentials due to the gradient calculation.")