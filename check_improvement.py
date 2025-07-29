#!/usr/bin/env python
"""Check exact improvement from dgelsd implementation."""

# Previous maximum error
previous_error = 6e-4  # 0.0006 J/mol

# New maximum error from test
new_error = 5.11e-4  # From test output

# Calculate improvement
improvement = previous_error / new_error

print(f"Previous maximum error: {previous_error:.6f} J/mol")
print(f"New maximum error:      {new_error:.6f} J/mol")
print(f"Improvement factor:     {improvement:.2f}x")
print(f"Error reduction:        {(1 - new_error/previous_error)*100:.1f}%")

# Also check if this is indeed an improvement
if new_error < previous_error:
    print("\n✓ SUCCESS: dgelsd_device implementation improves accuracy!")
else:
    print("\n✗ No improvement detected")