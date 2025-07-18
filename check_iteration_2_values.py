#!/usr/bin/env python
"""Check the reported values at iteration 2."""

# Values at iteration 0 (before consolidation):
print("ITERATION 0 (before consolidation):")
print("c_G[0] = 2.214892280104281e-01")
print("c_component[0,0] = 3.506507e-05")
print("chemical_potentials = [-25739.210873, -19284.602909]")

# What we see in the debug output at iteration 2:
print("\nITERATION 2 (after consolidation):")
print("From debug output:")
print("  delta_y[0] = 2.865204599913607e-06")
print("  c_G[0] = 2.865204599913607e-06")
print("  Chemical potential contribution: cp[0]=-0.000e+00 cp[1]=-0.000e+00 (total: -0.000000000000000e+00)")

print("\nThe issue is clear: After consolidation, c_G becomes tiny (2.865e-06 instead of 0.221)")
print("And the chemical potential contribution becomes zero!")