#!/usr/bin/env python
"""
Debug test to examine equilibrium matrix differences between CPU and GPU
for FCC_A1 phase with multiple sublattices
"""

from pycalphad import Database, equilibrium
import numpy as np
import os

# Clear cache
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Test a single condition that failed
conditions = {
    'T': 400,
    'P': 101325,
    'X(BI)': 0.1
}

print("Testing single condition with LIQUID + FCC_A1")
print(f"Conditions: T={conditions['T']}K, P={conditions['P']}Pa, X(BI)={conditions['X(BI)']}")
print("=" * 60)

# First check the phase information
print("\nPhase information:")
for phase_name in phases:
    phase = db.phases[phase_name]
    print(f"\n{phase_name}:")
    print(f"  Sublattices: {phase.sublattices}")
    print(f"  Site ratio sum: {sum(phase.sublattices)}")
    print(f"  Constituents: {phase.constituents}")

print("\n" + "=" * 60)

# CPU calculation with verbose output
print("\nCPU Calculation:")
print("-" * 60)
result_cpu = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 2000}, verbose=True)

# Extract CPU results
cpu_phases = []
cpu_energies = {}
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        if amount > 1e-10:
            cpu_phases.append((phase, amount))
            # Try to get phase energy
            try:
                # Find the index for this phase
                idx = np.where(mask)[0][0]
                # Get GM for this phase - this should be the phase energy
                # We need to be careful here as GM might be system GM
                print(f"\n{phase}:")
                print(f"  Amount: {amount:.6f}")
                print(f"  mask shape: {mask.shape}")
                print(f"  idx: {idx}")
                
                # Get composition
                x_bi = result_cpu['X_BI'].values.flat[idx] if 'X_BI' in result_cpu else None
                if x_bi is not None:
                    print(f"  X(BI): {x_bi:.6f}")
                
                # For debugging, let's look at the raw data structure
                if hasattr(result_cpu, 'GM'):
                    gm_values = result_cpu.GM.values
                    print(f"  GM values shape: {gm_values.shape}")
                    print(f"  GM at idx: {gm_values.flat[idx] if idx < gm_values.size else 'N/A'}")
                    
            except Exception as e:
                print(f"  Error getting details: {e}")

print(f"\nCPU System GM: {result_cpu.GM.values[0]:.6f}")
print(f"CPU MU: {result_cpu.MU.values[0]}")

# GPU calculation with verbose output
print("\n" + "=" * 60)
print("\nGPU Calculation:")
print("-" * 60)
result_gpu = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 2000}, gpu=True, verbose=True)

# Extract GPU results
gpu_phases = []
gpu_energies = {}
for phase in np.unique(result_gpu.Phase.values):
    if phase != '':
        mask = result_gpu.Phase.values == phase
        amount = result_gpu.NP.values[mask][0]
        if amount > 1e-10:
            gpu_phases.append((phase, amount))
            try:
                idx = np.where(mask)[0][0]
                print(f"\n{phase}:")
                print(f"  Amount: {amount:.6f}")
                print(f"  mask shape: {mask.shape}")
                print(f"  idx: {idx}")
                
                x_bi = result_gpu['X_BI'].values.flat[idx] if 'X_BI' in result_gpu else None
                if x_bi is not None:
                    print(f"  X(BI): {x_bi:.6f}")
                    
                if hasattr(result_gpu, 'GM'):
                    gm_values = result_gpu.GM.values
                    print(f"  GM values shape: {gm_values.shape}")
                    print(f"  GM at idx: {gm_values.flat[idx] if idx < gm_values.size else 'N/A'}")
                    
            except Exception as e:
                print(f"  Error getting details: {e}")

print(f"\nGPU System GM: {result_gpu.GM.values[0]:.6f}")
print(f"GPU MU: {result_gpu.MU.values[0]}")

# Compare results
print("\n" + "=" * 60)
print("\nComparison:")
print("-" * 60)

print("\nPhase fractions:")
for (cpu_phase, cpu_amt), (gpu_phase, gpu_amt) in zip(cpu_phases, gpu_phases):
    if cpu_phase == gpu_phase:
        diff = abs(cpu_amt - gpu_amt)
        print(f"{cpu_phase}: CPU={cpu_amt:.6f}, GPU={gpu_amt:.6f}, diff={diff:.6f}")

print(f"\nSystem GM difference: {abs(result_cpu.GM.values[0] - result_gpu.GM.values[0]):.6f}")

print("\nChemical potentials:")
mu_cpu = result_cpu.MU.values[0]
mu_gpu = result_gpu.MU.values[0]
for i, comp in enumerate(['AU', 'BI']):
    if i < len(mu_cpu) and i < len(mu_gpu):
        diff = abs(mu_cpu[i] - mu_gpu[i])
        print(f"MU_{comp}: CPU={mu_cpu[i]:.6f}, GPU={mu_gpu[i]:.6f}, diff={diff:.6f}")

# Now let's specifically look for the energy normalization issue
print("\n" + "=" * 60)
print("\nNormalization Analysis:")
print("-" * 60)
print("\nFor FCC_A1 phase:")
print(f"  Site ratio sum: {sum(db.phases['FCC_A1'].sublattices)} (should be 2.0)")
print("  If GPU normalizes by site ratio sum and CPU doesn't:")
print("  - GPU energy should be ~0.5x the CPU energy per mole of atoms")
print("  - This would make GPU find more FCC_A1 stable (lower energy)")
print("\nFor LIQUID phase:")
print(f"  Site ratio sum: {sum(db.phases['LIQUID'].sublattices)} (should be 1.0)")
print("  - No normalization difference expected")

# Let's see if we can extract the actual phase energies from the calculations
print("\n" + "=" * 60)
print("\nAttempting to extract phase energies from calculate step...")

# Run calculate to get phase energies directly
from pycalphad import calculate

# Create a grid of compositions for each phase
print("\nCalculating phase energies at equilibrium compositions...")

# We need to figure out the equilibrium compositions
# This is tricky because we need the actual Y values

# For now, let's just note the pattern
print("\nObserved pattern:")
print("- GPU finds MORE FCC_A1 than CPU (0.939 vs 0.891)")
print("- GPU finds LESS LIQUID than CPU (0.061 vs 0.109)")
print("- This is consistent with GPU having lower FCC_A1 energy due to normalization")
print("- Energy difference: 44.67 J/mol (GPU lower)")