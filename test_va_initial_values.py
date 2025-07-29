#!/usr/bin/env python
"""Check initial site fraction values for phases with VA sublattices."""

import numpy as np
from pycalphad import Database, calculate
import pycalphad.variables as v

# Test with Al-Cu-Fe 
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test condition
conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.5,
    v.X('CU'): 0.3,
}

print("Checking initial site fraction values from calculate():")
print("="*60)

# Calculate for FCC_A1 which has (AL,CU,FE):(VA) structure
print("\nFCC_A1 phase:")
calc_result = calculate(db, components, 'FCC_A1', output='GM', 
                       T=conditions[v.T], P=conditions[v.P], points={'FCC_A1': [[0.5, 0.3, 0.2, 1.0]]})

print(f"GM values shape: {calc_result.GM.values.shape}")
print(f"Sample GM value: {calc_result.GM.values[0,0,0,0]:.1f} J/mol")

# Check the points that were used
if 'points' in calc_result.coords:
    print(f"Number of points: {len(calc_result.coords['points'])}")
    
# Try to access Y coordinate to see site fraction structure
print(f"\nInternal DOF coordinates: {list(calc_result.coords.keys())}")

# Check if VA site fraction is properly set to 1.0
va_index = 3  # Y(FCC_A1,1,VA) should be the 4th site fraction
if abs(y_values[va_index] - 1.0) < 1e-10:
    print(f"\n✓ Y(FCC_A1,1,VA) correctly initialized to 1.0")
else:
    print(f"\n⚠️  Y(FCC_A1,1,VA) = {y_values[va_index]:.6f}, expected 1.0")

# Also check BCC_A2 which has (AL,CU,FE,VA):(VA) structure
print("\n" + "="*60)
print("BCC_A2 phase:")
calc_result2 = calculate(db, components, 'BCC_A2', output='Y',
                        T=conditions[v.T], P=conditions[v.P], points={'BCC_A2': [[0.5, 0.3, 0.15, 0.05, 1.0]]})

print(f"Site fraction variables: {calc_result2.coords['Y_BCC_A2'].values}")
y_values2 = calc_result2.Y.values[0,0,0,0,:]
print(f"Site fraction values:")
for i, val in enumerate(y_values2):
    print(f"  Y[{i}] = {val:.6f}")

# Check if second sublattice VA is 1.0
va_index2 = 4  # Y(BCC_A2,1,VA) should be the 5th site fraction
if abs(y_values2[va_index2] - 1.0) < 1e-10:
    print(f"\n✓ Y(BCC_A2,1,VA) correctly initialized to 1.0")
else:
    print(f"\n⚠️  Y(BCC_A2,1,VA) = {y_values2[va_index2]:.6f}, expected 1.0")

# Test with default points (no explicit initialization)
print("\n" + "="*60)
print("Testing with default points (no explicit Y values):")
calc_default = calculate(db, components, 'FCC_A1', output='Y',
                        T=conditions[v.T], P=conditions[v.P])
y_default = calc_default.Y.values[0,0,0,:,:]
print(f"Default Y values shape: {y_default.shape}")
print(f"Sample site fractions:")
for pt in range(min(3, y_default.shape[0])):
    print(f"  Point {pt}: {y_default[pt,:]}")
    
# Check how many points have Y(FCC_A1,1,VA) = 1.0
va_values = y_default[:, va_index]
correct_va = np.sum(np.abs(va_values - 1.0) < 1e-10)
print(f"\nPoints with Y(FCC_A1,1,VA) = 1.0: {correct_va}/{len(va_values)}")
if correct_va < len(va_values):
    print(f"⚠️  Not all points have correct VA initialization!")
    print(f"  VA values range: [{np.min(va_values):.6f}, {np.max(va_values):.6f}]")