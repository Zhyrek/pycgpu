#!/usr/bin/env python3
"""Check the types of pycalphad variables"""
import pycalphad.variables as v

print("=== Variable Type Check ===")
print(f"v.P type: {type(v.P)}")
print(f"v.P.__class__.__mro__: {v.P.__class__.__mro__}")
print(f"Is P an IndependentPotential? {isinstance(v.P, v.IndependentPotential)}")

print(f"\nv.T type: {type(v.T)}")
print(f"v.T.__class__.__mro__: {v.T.__class__.__mro__}")
print(f"Is T an IndependentPotential? {isinstance(v.T, v.IndependentPotential)}")

print(f"\nv.N type: {type(v.N)}")
print(f"v.N.__class__.__mro__: {v.N.__class__.__mro__}")
print(f"Is N a SystemMolesType? {isinstance(v.N, v.SystemMolesType)}")

# Check what StateVariable includes
print("\n=== StateVariable Check ===")
print(f"v.StateVariable: {v.StateVariable}")

# List all state variables
print("\n=== All Variables ===")
for attr_name in dir(v):
    attr = getattr(v, attr_name)
    if hasattr(attr, '__class__') and hasattr(attr.__class__, '__mro__'):
        if v.StateVariable in attr.__class__.__mro__:
            print(f"{attr_name}: {attr} - type: {type(attr)}")