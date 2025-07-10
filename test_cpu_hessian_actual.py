#!/usr/bin/env python3
"""Test what the CPU actually computes for the hessian"""

import numpy as np
import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory

# Load model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory
conditions = {v.T: 1000, v.P: 101325, v.N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})

# Get phase record
phase_rec = prf.get('BCC_A2')

print("=== Testing CPU Hessian at Multiple Points ===")

# Test at different site fraction values
test_cases = [
    ([1.0, 101325.0, 1000.0, 0.6, 0.4], "Y_NB=0.6, Y_TI=0.4"),
    ([1.0, 101325.0, 1000.0, 0.5, 0.5], "Y_NB=0.5, Y_TI=0.5"),
    ([1.0, 101325.0, 1000.0, 0.8, 0.2], "Y_NB=0.8, Y_TI=0.2"),
    ([1.0, 101325.0, 1000.0, 0.6, 0.3], "Y_NB=0.6, Y_TI=0.3, sum=0.9"),
]

R = 8.3145
T = 1000.0

for dof, desc in test_cases:
    dof_array = np.array(dof)
    hess = np.zeros((5, 5), order='C')
    phase_rec.formulahess(hess, dof_array)
    
    Y_NB = dof[3]
    Y_TI = dof[4]
    sum_Y = Y_NB + Y_TI
    
    print(f"\n{desc}:")
    print(f"  Sum = {sum_Y}")
    print(f"  hess[3,3] = {hess[3,3]:.2f}")
    print(f"  hess[4,4] = {hess[4,4]:.2f}")
    print(f"  Expected [3,3] = RT/Y_NB = {R*T/Y_NB:.2f}")
    print(f"  Expected [4,4] = RT/Y_TI = {R*T/Y_TI:.2f}")
    
    # Check if it's scaled by site fraction sum
    if abs(sum_Y - 1.0) > 0.01:
        print(f"  Scaled by sum? hess[3,3] * sum = {hess[3,3] * sum_Y:.2f}")
        print(f"  RT/Y_NB/sum = {R*T/Y_NB/sum_Y:.2f}")

print("\n=== Hypothesis Testing ===")

# Test if CPU is doing: RT/Y_i / (sum of site fractions)
# This would explain why it works when sum=1.0

# For the sum=0.9 case:
dof_09 = np.array([1.0, 101325.0, 1000.0, 0.6, 0.3])
hess_09 = np.zeros((5, 5), order='C')
phase_rec.formulahess(hess_09, dof_09)

Y_NB = 0.6
Y_TI = 0.3
sum_Y = 0.9

expected_raw = R * T / Y_NB
expected_scaled = expected_raw / sum_Y

print(f"\nFor sum=0.9 case:")
print(f"  CPU hess[3,3] = {hess_09[3,3]:.2f}")
print(f"  RT/Y_NB = {expected_raw:.2f}")
print(f"  RT/Y_NB/sum = {expected_scaled:.2f}")
print(f"  Which matches? {'Scaled' if abs(hess_09[3,3] - expected_scaled) < 1 else 'Raw'}")