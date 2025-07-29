#!/usr/bin/env python
"""
Test CPU vs GPU with LIQUID and BCC phases to check vacancy handling
"""

import pycalphad as cp
import numpy as np

# Load database
db = cp.Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test with LIQUID (no vacancies) and BCC_B2 (has vacancies)
phases = ['LIQUID', 'BCC_B2']

# Test conditions
conditions = {
    cp.v.X('AL'): 0.7,
    cp.v.X('CU'): 0.2,
    cp.v.T: 1500,  # Higher temperature to ensure liquid is stable
    cp.v.P: 101325
}

print("Testing LIQUID and BCC_B2 phases")
print("=" * 60)
print(f"Phases: {phases}")
print(f"Conditions: X(AL)={conditions[cp.v.X('AL')]}, X(CU)={conditions[cp.v.X('CU')]}, T={conditions[cp.v.T]}K")
print(f"X(FE) = {1 - conditions[cp.v.X('AL')] - conditions[cp.v.X('CU')]}")

# Check phase info
print("\nPhase information:")
for phase_name in phases:
    phase = db.phases[phase_name]
    print(f"\n{phase_name}:")
    print(f"  Sublattices: {phase.sublattices}")
    print(f"  Constituents: {phase.constituents}")
    has_va = any('VA' in str(const) for subl in phase.constituents for const in subl)
    print(f"  Has vacancies: {has_va}")

# CPU calculation
print("\n" + "="*60)
print("CPU calculation...")
try:
    eq_cpu = cp.equilibrium(db, components, phases, conditions, verbose=False)
    cpu_gm = float(eq_cpu.GM.values.flat[0])
    print(f"  GM = {cpu_gm:.3f} J/mol")
    
    # Check which phases are stable
    cpu_phases = {}
    for i, phase in enumerate(eq_cpu.Phase.values.flat):
        if i < len(eq_cpu.NP.values.flat) and eq_cpu.NP.values.flat[i] > 1e-6:
            cpu_phases[phase] = eq_cpu.NP.values.flat[i]
            print(f"  {phase}: NP = {eq_cpu.NP.values.flat[i]:.6f}")
            
    # Get chemical potentials
    print("\n  Chemical potentials (CPU):")
    mu_values = eq_cpu.MU.values.flat
    if len(mu_values) >= 3:
        print(f"    MU(AL) = {mu_values[0]:.3f} J/mol")
        print(f"    MU(CU) = {mu_values[1]:.3f} J/mol")
        print(f"    MU(FE) = {mu_values[2]:.3f} J/mol")
                
except Exception as e:
    print(f"  Error: {e}")
    cpu_gm = None

# GPU calculation
print("\n" + "="*60)
print("GPU calculation...")
try:
    eq_gpu = cp.equilibrium(db, components, phases, conditions, gpu=True, verbose=False)
    gpu_gm = float(eq_gpu.GM.values.flat[0])
    print(f"  GM = {gpu_gm:.3f} J/mol")
    
    # Check which phases are stable
    gpu_phases = {}
    for i, phase in enumerate(eq_gpu.Phase.values.flat):
        if i < len(eq_gpu.NP.values.flat) and eq_gpu.NP.values.flat[i] > 1e-6:
            gpu_phases[phase] = eq_gpu.NP.values.flat[i]
            print(f"  {phase}: NP = {eq_gpu.NP.values.flat[i]:.6f}")
            
    # Get chemical potentials
    print("\n  Chemical potentials (GPU):")
    mu_values = eq_gpu.MU.values.flat
    if len(mu_values) >= 3:
        print(f"    MU(AL) = {mu_values[0]:.3f} J/mol")
        print(f"    MU(CU) = {mu_values[1]:.3f} J/mol")
        print(f"    MU(FE) = {mu_values[2]:.3f} J/mol")
                
    if cpu_gm is not None:
        error = abs(cpu_gm - gpu_gm)
        print(f"\nGM Error: {error:.3f} J/mol")
        
        # Compare chemical potentials
        cpu_mu = eq_cpu.MU.values.flat
        gpu_mu = eq_gpu.MU.values.flat
        if len(cpu_mu) >= 3 and len(gpu_mu) >= 3:
            print("\nChemical potential errors:")
            print(f"  MU(AL) error: {abs(cpu_mu[0] - gpu_mu[0]):.3f} J/mol")
            print(f"  MU(CU) error: {abs(cpu_mu[1] - gpu_mu[1]):.3f} J/mol")
            print(f"  MU(FE) error: {abs(cpu_mu[2] - gpu_mu[2]):.3f} J/mol")
        
        # Check phase stability differences
        print("\nPhase stability comparison:")
        all_phases = set(cpu_phases.keys()) | set(gpu_phases.keys())
        for phase in sorted(all_phases):
            cpu_np = cpu_phases.get(phase, 0.0)
            gpu_np = gpu_phases.get(phase, 0.0)
            if cpu_np > 1e-6 or gpu_np > 1e-6:
                print(f"  {phase}: CPU={cpu_np:.6f}, GPU={gpu_np:.6f}, diff={abs(cpu_np-gpu_np):.6f}")
            
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# Now test at lower temperature where BCC might be more stable
print("\n" + "="*60)
print("Testing at lower temperature (800K)")
print("="*60)

conditions_low_t = {
    cp.v.X('AL'): 0.7,
    cp.v.X('CU'): 0.2,
    cp.v.T: 800,
    cp.v.P: 101325
}

# CPU calculation at low T
print("\nCPU calculation (800K)...")
try:
    eq_cpu_low = cp.equilibrium(db, components, phases, conditions_low_t, verbose=False)
    cpu_gm_low = float(eq_cpu_low.GM.values.flat[0])
    print(f"  GM = {cpu_gm_low:.3f} J/mol")
    
    # Check which phases are stable
    for i, phase in enumerate(eq_cpu_low.Phase.values.flat):
        if i < len(eq_cpu_low.NP.values.flat) and eq_cpu_low.NP.values.flat[i] > 1e-6:
            print(f"  {phase}: NP = {eq_cpu_low.NP.values.flat[i]:.6f}")
except Exception as e:
    print(f"  Error: {e}")
    cpu_gm_low = None

# GPU calculation at low T
print("\nGPU calculation (800K)...")
try:
    eq_gpu_low = cp.equilibrium(db, components, phases, conditions_low_t, gpu=True, verbose=False)
    gpu_gm_low = float(eq_gpu_low.GM.values.flat[0])
    print(f"  GM = {gpu_gm_low:.3f} J/mol")
    
    # Check which phases are stable
    for i, phase in enumerate(eq_gpu_low.Phase.values.flat):
        if i < len(eq_gpu_low.NP.values.flat) and eq_gpu_low.NP.values.flat[i] > 1e-6:
            print(f"  {phase}: NP = {eq_gpu_low.NP.values.flat[i]:.6f}")
            
    if cpu_gm_low is not None:
        error_low = abs(cpu_gm_low - gpu_gm_low)
        print(f"\nGM Error at 800K: {error_low:.3f} J/mol")
except Exception as e:
    print(f"  Error: {e}")

# Test with just LIQUID phase (no vacancies)
print("\n" + "="*60)
print("Testing LIQUID phase only (no vacancies)")
print("="*60)

# CPU - LIQUID only
print("\nCPU calculation (LIQUID only)...")
try:
    eq_cpu_liq = cp.equilibrium(db, components, ['LIQUID'], conditions, verbose=False)
    cpu_gm_liq = float(eq_cpu_liq.GM.values.flat[0])
    print(f"  GM = {cpu_gm_liq:.3f} J/mol")
except Exception as e:
    print(f"  Error: {e}")
    cpu_gm_liq = None

# GPU - LIQUID only
print("\nGPU calculation (LIQUID only)...")
try:
    eq_gpu_liq = cp.equilibrium(db, components, ['LIQUID'], conditions, gpu=True, verbose=False)
    gpu_gm_liq = float(eq_gpu_liq.GM.values.flat[0])
    print(f"  GM = {gpu_gm_liq:.3f} J/mol")
    
    if cpu_gm_liq is not None:
        error_liq = abs(cpu_gm_liq - gpu_gm_liq)
        print(f"\nLIQUID-only error: {error_liq:.3f} J/mol")
        print(f"This tests a phase without vacancies")
except Exception as e:
    print(f"  Error: {e}")