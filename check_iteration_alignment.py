#!/usr/bin/env python
"""Check if GPU iteration 0 = CPU iteration 1 or if they're truly different."""

print("CHECKING ITERATION ALIGNMENT: GPU vs CPU")
print("=" * 60)

print("From the debug outputs, let me extract key algorithmic states:")

print("\n" + "="*60)
print("INITIAL STATE (Starting compositions)")

print("\nCPU iteration 0:")
print("- Phase 0 gradients: [-19434.73793796, -13118.21765729]")
print("- Phase 1 gradients: [-19787.40464808, -13187.33540866]")
print("- c_G values: Phase 0=[0.22148923, -0.22148923], Phase 1=[0.19798451, -0.19798451]")
print("- Phase compositions: Different, not ready for consolidation")

print("\nGPU iteration 0 (initial):")
print("- Need to find GPU's initial gradients and c_G values")
print("- Need to compare with CPU iteration 0")

print("\n" + "="*60)
print("STATE AT CONSOLIDATION")

print("\nCPU iteration 1 (at consolidation):")
print("- Phase 0: composition [0.09685286, 0.9031471]")
print("- Phase 1: composition [0.09686725, 0.9031328]")
print("- Phase 0 gradients: [-19614.94249925, -13154.5657734]")
print("- Phase 1 gradients: [-19614.38915241, -13154.45755301]")
print("- c_G values: [0.20879587, -0.20879587] (very close between phases)")
print("- max_diff = 0.000014 → CONSOLIDATES")

print("\nGPU iteration 0 (at consolidation):")
print("- Phase 0: composition [0.096853, 0.903147]")
print("- Phase 1: composition [0.096867, 0.903133]")
print("- Need to find: GPU gradients at this point")
print("- Need to find: GPU c_G values at this point")
print("- max_diff = 0.000014 → CONSOLIDATES")

print("\n" + "="*60)
print("THE KEY TEST")

print("\nIf GPU iteration 0 = CPU iteration 1:")
print("- GPU and CPU should have IDENTICAL gradients at consolidation")
print("- GPU and CPU should have IDENTICAL c_G values at consolidation")
print("- They would be at the same algorithmic state, just labeled differently")

print("\nIf GPU iteration 0 ≠ CPU iteration 1:")
print("- GPU and CPU would have DIFFERENT gradients")
print("- GPU and CPU would have DIFFERENT c_G values")
print("- They would be at truly different algorithmic states")

print("\n" + "="*60)
print("WHAT TO CHECK")

print("\n1. Find GPU's gradients when it consolidates (iteration 0)")
print("2. Compare with CPU's gradients when it consolidates (iteration 1)")
print("3. Find GPU's c_G values when it consolidates")
print("4. Compare with CPU's c_G values when it consolidates")

print("\nIf all these values match exactly, then:")
print("→ GPU iteration 0 = CPU iteration 1 (labeling offset)")
print("→ No actual timing difference")
print("→ Consolidation happens at same algorithmic state")

print("\nIf these values differ, then:")
print("→ GPU iteration 0 ≠ CPU iteration 1 (true timing difference)")
print("→ Actual algorithmic difference")
print("→ Need to fix the evolution path")

print("\n" + "="*60)
print("EXTRACTING THE DATA...")

print("\nFrom previous analysis:")
print("CPU consolidates with gradients: [-19614.94, -13154.57] and [-19614.39, -13154.46]")
print("CPU consolidates with c_G: [0.20879587, -0.20879587]")

print("\nNeed to find:")
print("GPU gradients when it consolidates")
print("GPU c_G values when it consolidates")