#!/usr/bin/env python
"""
Direct comparison of ALCU_ZETA energy calculations between CPU and GPU
"""

import pycalphad as cp
import numpy as np

# Load database
db = cp.Database('Al-Cu-Fe.tdb')

# Get ALCU_ZETA phase
alcu_zeta = db.phases['ALCU_ZETA']

print("ALCU_ZETA phase info:")
print(f"  Site ratios: {alcu_zeta.sublattices}")
print(f"  Sum of site ratios: {sum(alcu_zeta.sublattices)}")

# Test conditions
T = 600  # K
P = 101325  # Pa

# Site fractions for pure Al in first sublattice, pure Cu in second
Y_ALCU_ZETA0AL = 1.0
Y_ALCU_ZETA1CU = 1.0
Y_ALCU_ZETA1FE = 0.0

print(f"\nTest conditions:")
print(f"  T = {T} K")
print(f"  P = {P} Pa")
print(f"  Y(ALCU_ZETA,0,AL) = {Y_ALCU_ZETA0AL}")
print(f"  Y(ALCU_ZETA,1,CU) = {Y_ALCU_ZETA1CU}")
print(f"  Y(ALCU_ZETA,1,FE) = {Y_ALCU_ZETA1FE}")

# Calculate phase composition
# First sublattice: 9 moles of Al
# Second sublattice: 11 moles of Cu
# Total: 9 Al + 11 Cu = 20 atoms per formula unit
X_AL = 9.0 / 20.0  # 0.45
X_CU = 11.0 / 20.0  # 0.55
X_FE = 0.0 / 20.0  # 0.0

print(f"\nPhase composition:")
print(f"  X(AL) = {X_AL}")
print(f"  X(CU) = {X_CU}")
print(f"  X(FE) = {X_FE}")

# Create a simple equilibrium calculation with just ALCU_ZETA
components = ['AL', 'CU', 'FE', 'VA']

# Set up conditions to force these exact site fractions
conditions = {
    cp.v.T: T,
    cp.v.P: P,
    cp.v.X('AL'): X_AL,
    cp.v.X('CU'): X_CU,
}

print("\n" + "="*60)
print("CPU Calculation:")
print("="*60)

# Calculate with single phase
eq_cpu = cp.equilibrium(db, components, ['ALCU_ZETA'], conditions, 
                        verbose=True, debug=True)

# Extract energy per formula unit
cpu_energy = float(eq_cpu.GM.values.flat[0])
print(f"\nCPU Energy (GM): {cpu_energy:.6f} J/mol")

# Check phase amounts
for i, phase in enumerate(eq_cpu.Phase.values.flat):
    if i < len(eq_cpu.NP.values.flat) and eq_cpu.NP.values.flat[i] > 1e-6:
        print(f"  {phase}: NP = {eq_cpu.NP.values.flat[i]:.6f}")
        
# Extract site fractions
print("\nCPU Site fractions:")
Y_values = eq_cpu.Y.sel(vertex=0).values
print(f"  Y values shape: {Y_values.shape}")
if Y_values.size >= 3:
    print(f"  Y(ALCU_ZETA,0,AL) = {Y_values.flat[0]:.6f}")
    print(f"  Y(ALCU_ZETA,1,CU) = {Y_values.flat[1]:.6f}")
    print(f"  Y(ALCU_ZETA,1,FE) = {Y_values.flat[2]:.6f}")

print("\n" + "="*60)
print("GPU Calculation:")
print("="*60)

# Calculate with GPU
eq_gpu = cp.equilibrium(db, components, ['ALCU_ZETA'], conditions, 
                        gpu=True, verbose=True, debug=True)

# Extract energy
gpu_energy = float(eq_gpu.GM.values.flat[0])
print(f"\nGPU Energy (GM): {gpu_energy:.6f} J/mol")

# Check phase amounts
for i, phase in enumerate(eq_gpu.Phase.values.flat):
    if i < len(eq_gpu.NP.values.flat) and eq_gpu.NP.values.flat[i] > 1e-6:
        print(f"  {phase}: NP = {eq_gpu.NP.values.flat[i]:.6f}")

# Extract site fractions
print("\nGPU Site fractions:")
Y_values = eq_gpu.Y.sel(vertex=0).values
print(f"  Y values shape: {Y_values.shape}")
if Y_values.size >= 3:
    print(f"  Y(ALCU_ZETA,0,AL) = {Y_values.flat[0]:.6f}")
    print(f"  Y(ALCU_ZETA,1,CU) = {Y_values.flat[1]:.6f}")
    print(f"  Y(ALCU_ZETA,1,FE) = {Y_values.flat[2]:.6f}")

print("\n" + "="*60)
print("COMPARISON:")
print("="*60)
print(f"CPU Energy: {cpu_energy:.6f} J/mol")
print(f"GPU Energy: {gpu_energy:.6f} J/mol")
print(f"Difference: {abs(cpu_energy - gpu_energy):.6f} J/mol")
print(f"Error ratio vs 806 J/mol: {abs(cpu_energy - gpu_energy)/806:.2f}x")
print(f"Error ratio vs 2825 J/mol: {abs(cpu_energy - gpu_energy)/2825:.2f}x")