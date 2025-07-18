#!/usr/bin/env python
"""Systematically trace where CPU and GPU diverge."""

print("SYSTEMATIC COMPARISON: CPU vs GPU")
print("=" * 60)

print("From the debug output, let me extract the key values at each iteration:")

print("\n" + "="*60)
print("ITERATION 0 (Initial two-phase state)")

print("\nCPU iteration 0:")
print("- Number of active phases: 2")
print("- Phase 0: site_fractions=[0.101695, 0.898305], amount=0.815592")
print("- Phase 1: site_fractions=[0.092504, 0.907496], amount=0.184408")
print("- Overall X(TI): 0.9 (satisfied)")

print("\nGPU iteration 0:")
print("- Number of active phases: 2 initially, then consolidates to 1")
print("- Phase 0: site_fractions=[0.101695, 0.898305], amount=0.815592")
print("- Phase 1: site_fractions=[0.092504, 0.907496], amount=0.184408")
print("- After consolidation: site_fractions=[0.096853, 0.903147], amount=1.0")
print("- Overall X(TI): 0.903147 (NOT satisfied)")

print("\n" + "="*60)
print("KEY DIFFERENCE: CONSOLIDATION TIMING")

print("\nCPU: Keeps two phases through iteration 1, consolidates between iter 1->2")
print("GPU: Consolidates immediately during iteration 0")

print("\n" + "="*60)
print("ITERATION 1")

print("\nCPU iteration 1 (still two phases):")
print("- Phase 0: site_fractions=[0.096853, 0.903147], amount=1.0")
print("- Phase 1: site_fractions=[0.096867, 0.903133], amount=8.33e-17")
print("- c_G values: [0.20879587, -0.20879587]")
print("- Matrix: 4x4 (two phases)")

print("\nGPU iteration 1 (already single phase):")
print("- Phase 0: site_fractions=[0.096853, 0.903147], amount=1.0")
print("- c_G values: [0.20879587, -0.20879587] (IDENTICAL to CPU)")
print("- Matrix: 3x3 (single phase)")

print("\n" + "="*60)
print("ITERATION 2+")

print("\nCPU iteration 2 (after consolidation):")
print("- Single phase: site_fractions=[0.096853, 0.903147], amount=1.0")
print("- Then converges to exact X(TI)=0.900000")

print("\nGPU iteration 2+:")
print("- Single phase: site_fractions=[0.097040, 0.902960], amount=1.0")
print("- c_G values: [0.20927135, -0.20927135] (DIFFERENT from CPU)")
print("- Remains stuck at X(TI)=0.902960")

print("\n" + "="*60)
print("CONCLUSIONS")

print("\n1. CPU and GPU are IDENTICAL through iteration 1 in terms of:")
print("   - Site fractions: [0.096853, 0.903147]")
print("   - c_G values: [0.20879587, -0.20879587]")
print("   - Phase amounts: 1.0")

print("\n2. CPU and GPU DIVERGE starting at iteration 2:")
print("   - CPU: Maintains site fractions, satisfies constraint exactly")
print("   - GPU: Changes site fractions to [0.097040, 0.902960], wrong constraint")

print("\n3. The root cause is NOT in consolidation timing")
print("   The root cause is in the constraint equation solution after consolidation")

print("\n4. Both reach the same state after consolidation:")
print("   Single phase with site_fractions=[0.096853, 0.903147]")
print("   But then they solve the constraint equation differently")

print("\n" + "="*60)
print("THE ACTUAL ISSUE")

print("\nAfter consolidation, both CPU and GPU have:")
print("- Single BCC phase")
print("- Site fractions: [0.096853, 0.903147]") 
print("- X(TI) = 0.903147 (violates constraint X(TI)=0.9)")

print("\nCPU: Solves constraint equation to achieve exactly X(TI)=0.9")
print("GPU: Tries to adjust site fractions but gets wrong answer")

print("\nThe bug is in the GPU's constraint equation construction/solution")
print("after consolidation, not in the consolidation process itself.")