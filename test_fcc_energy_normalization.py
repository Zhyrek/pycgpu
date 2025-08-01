#!/usr/bin/env python
"""
Test to examine energy normalization differences between CPU and GPU
for FCC_A1 phase with multiple sublattices
"""

from pycalphad import Database, calculate, equilibrium
import numpy as np
import os

# Clear cache
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']

print("Testing energy normalization for FCC_A1 phase")
print("=" * 60)

# Check phase structure
fcc_phase = db.phases['FCC_A1']
liquid_phase = db.phases['LIQUID']

print(f"\nFCC_A1 sublattices: {fcc_phase.sublattices}")
print(f"FCC_A1 site ratio sum: {sum(fcc_phase.sublattices)}")
print(f"\nLIQUID sublattices: {liquid_phase.sublattices}")
print(f"LIQUID site ratio sum: {sum(liquid_phase.sublattices)}")

# Test at a specific composition and temperature
T = 400
P = 101325

# Calculate energy for FCC_A1 at a specific composition
# Let's use pure Au (Y(FCC_A1,0,AU)=1, Y(FCC_A1,1,VA)=1)
print("\n" + "=" * 60)
print("\nCalculating FCC_A1 energy for pure Au with vacancy:")

# First, let's do a simple calculate to see the energy
calc_result = calculate(db, components, 'FCC_A1', 
                       P=P, T=T, 
                       points={'Y(FCC_A1,0,AU)': [1.0], 
                              'Y(FCC_A1,0,BI)': [0.0],
                              'Y(FCC_A1,1,VA)': [1.0]})

print(f"\nCPU calculate result for FCC_A1:")
print(f"  GM values: {calc_result.GM.values}")
print(f"  Shape: {calc_result.GM.shape}")

# Now test equilibrium with both CPU and GPU
print("\n" + "=" * 60)
print("\nTesting equilibrium calculation:")

conditions = {
    'T': T,
    'P': P,
    'X(BI)': 0.1
}

# CPU equilibrium
print("\nCPU equilibrium:")
eq_cpu = equilibrium(db, components, ['LIQUID', 'FCC_A1'], conditions, 
                    calc_opts={'pdens': 1000})
cpu_gm = eq_cpu.GM.values.flat[0]
print(f"  System GM: {cpu_gm:.6f}")

# Extract phase information
for phase in np.unique(eq_cpu.Phase.values):
    if phase != '':
        mask = eq_cpu.Phase.values == phase
        amount = eq_cpu.NP.values[mask][0]
        if amount > 1e-10:
            print(f"  {phase}: NP = {amount:.6f}")

# GPU equilibrium
print("\nGPU equilibrium:")
eq_gpu = equilibrium(db, components, ['LIQUID', 'FCC_A1'], conditions, 
                    calc_opts={'pdens': 1000}, gpu=True)
gpu_gm = eq_gpu.GM.values.flat[0]
print(f"  System GM: {gpu_gm:.6f}")

# Extract phase information
for phase in np.unique(eq_gpu.Phase.values):
    if phase != '':
        mask = eq_gpu.Phase.values == phase
        amount = eq_gpu.NP.values[mask][0]
        if amount > 1e-10:
            print(f"  {phase}: NP = {amount:.6f}")

print("\n" + "=" * 60)
print("\nAnalysis:")
print(f"GM difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
print(f"Site ratio sum for FCC_A1: {sum(fcc_phase.sublattices)}")
print(f"\nIf GPU normalizes by site ratio but CPU doesn't:")
print(f"  - GPU would see FCC_A1 energy as ~1/{sum(fcc_phase.sublattices)} of CPU value")
print(f"  - This would make FCC_A1 appear more stable in GPU calculation")
print(f"  - GPU would predict MORE FCC_A1 than CPU")

# Let's also directly check the phase record energy calculation
print("\n" + "=" * 60)
print("\nDirect phase energy check:")

# Get phase records
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad import variables as v
prf = PhaseRecordFactory(db, components, [v.N, v.P, v.T], model=None)

# Get the FCC_A1 phase record
fcc_record = prf['FCC_A1']
print(f"\nFCC_A1 phase record info:")
print(f"  variables: {fcc_record.variables}")
print(f"  sublattice dof: {fcc_record.sublattice_dof}")
print(f"  site ratios: {fcc_record.site_ratios if hasattr(fcc_record, 'site_ratios') else 'Not available'}")

# Test energy calculation
test_dof = np.array([1.0, P, T, 1.0, 0.0, 1.0])  # N, P, T, Y(AU), Y(BI), Y(VA)
energy = fcc_record.obj(test_dof)
grad = np.zeros(6)
fcc_record.grad(test_dof, grad)

print(f"\nDirect energy calculation for pure Au FCC_A1:")
print(f"  DOF: {test_dof}")
print(f"  Energy: {energy:.6f} J/mol")
print(f"  Energy gradient: {grad}")

# Compare with LIQUID
liquid_record = prf['LIQUID']
print(f"\nLIQUID phase record info:")
print(f"  variables: {liquid_record.variables}")
print(f"  sublattice dof: {liquid_record.sublattice_dof}")

# Test Au-rich liquid
test_dof_liquid = np.array([1.0, P, T, 0.9, 0.1])  # N, P, T, Y(AU), Y(BI)
energy_liquid = liquid_record.obj(test_dof_liquid)

print(f"\nDirect energy calculation for Au-rich LIQUID:")
print(f"  DOF: {test_dof_liquid}")
print(f"  Energy: {energy_liquid:.6f} J/mol")