#!/usr/bin/env python
"""Understand the real problem with GPU vs CPU."""

print("THE REAL PROBLEM")
print("=" * 60)

print("\nWhat's happening:")
print("1. Two phases start with different compositions:")
print("   - Phase 0: X(TI) = 0.898305 (81.6% of system)")
print("   - Phase 1: X(TI) = 0.907496 (18.4% of system)")
print("   - Overall: 0.8156*0.898305 + 0.1844*0.907496 = 0.900")

print("\n2. Phase 1 amount becomes very small (1e-10)")

print("\n3. GPU: Simply removes Phase 1")
print("   - Keeps Phase 0 with X(TI) = 0.898305")
print("   - This violates mass balance! Should be 0.900")

print("\n4. CPU: Somehow gets X(TI) = 0.903147")
print("   - This is close to what's needed")
print("   - Suggests CPU does something different")

print("\n" + "="*60)
print("HYPOTHESIS:")

print("\nThe CPU might be:")
print("1. Adjusting the remaining phase composition to satisfy mass balance")
print("2. Or consolidating phases even though they're different")
print("3. Or using a different initial guess that leads to consolidation")

print("\nThe GPU damping fix doesn't help because:")
print("- We're starting with wrong composition (0.898305)")
print("- No amount of damping will fix the wrong starting point")
print("- We need to fix how phases are handled when one is removed")