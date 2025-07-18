#!/usr/bin/env python
"""Debug the problematic condition: X(TI)=0.5, T=700K with large CPU/GPU discrepancy."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# The problematic condition
conditions = {v.X('TI'): 0.5, v.T: 700, v.P: 101325}

print("=" * 80)
print("DEBUGGING PROBLEMATIC CONDITION: X(TI)=0.5, T=700K")
print("=" * 80)

print("Available phases:", [p.name for p in phases])

# First, run both CPU and GPU with full output to see what's happening
print("\n" + "="*60)
print("CPU CALCULATION WITH FULL OUTPUT")
print("="*60)

try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    
    print(f"\nCPU Results Summary:")
    print(f"  GM: {result_cpu.GM.values.flatten()[0]:.6f} J/mol")
    print(f"  MU: {result_cpu.MU.values.flatten()}")
    print(f"  Phase fractions: {result_cpu.Phase.values.flatten()}")
    print(f"  X compositions: {result_cpu.X.values}")
    print(f"  Y site fractions: {result_cpu.Y.values if hasattr(result_cpu, 'Y') else 'N/A'}")
    
    # Check which phases are stable
    stable_phases = []
    for phase_idx, phase_frac in enumerate(result_cpu.Phase.values.flatten()):
        if phase_frac > 1e-12:
            stable_phases.append((phase_idx, phases[phase_idx].name, phase_frac))
    
    print(f"  Stable phases (fraction > 1e-12):")
    for idx, name, frac in stable_phases:
        print(f"    Phase {idx} ({name}): {frac:.6f}")
        
except Exception as e:
    print(f"CPU calculation failed: {e}")
    result_cpu = None

print("\n" + "="*60)
print("GPU CALCULATION WITH FULL OUTPUT")
print("="*60)

try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    print(f"\nGPU Results Summary:")
    print(f"  GM: {result_gpu.GM.values.flatten()[0]:.6f} J/mol")
    print(f"  MU: {result_gpu.MU.values.flatten()}")
    print(f"  Phase fractions: {result_gpu.Phase.values.flatten()}")
    print(f"  X compositions: {result_gpu.X.values}")
    print(f"  Y site fractions: {result_gpu.Y.values if hasattr(result_gpu, 'Y') else 'N/A'}")
    
    # Check which phases are stable
    stable_phases_gpu = []
    for phase_idx, phase_frac in enumerate(result_gpu.Phase.values.flatten()):
        if phase_frac > 1e-12:
            stable_phases_gpu.append((phase_idx, phases[phase_idx].name, phase_frac))
    
    print(f"  Stable phases (fraction > 1e-12):")
    for idx, name, frac in stable_phases_gpu:
        print(f"    Phase {idx} ({name}): {frac:.6f}")
        
except Exception as e:
    print(f"GPU calculation failed: {e}")
    result_gpu = None

# Compare results if both succeeded
if result_cpu is not None and result_gpu is not None:
    print("\n" + "="*60)
    print("DETAILED COMPARISON")
    print("="*60)
    
    cpu_gm = result_cpu.GM.values.flatten()[0]
    gpu_gm = result_gpu.GM.values.flatten()[0]
    
    print(f"Free Energy Comparison:")
    print(f"  CPU GM: {cpu_gm:.6f} J/mol")
    print(f"  GPU GM: {gpu_gm:.6f} J/mol")
    print(f"  Absolute difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
    print(f"  Relative error: {abs(cpu_gm - gpu_gm)/abs(cpu_gm)*100:.4f}%")
    
    print(f"\nPhase Fractions Comparison:")
    cpu_phases = result_cpu.Phase.values.flatten()
    gpu_phases = result_gpu.Phase.values.flatten()
    
    for i, (cpu_frac, gpu_frac) in enumerate(zip(cpu_phases, gpu_phases)):
        phase_name = phases[i].name if i < len(phases) else f"Phase_{i}"
        print(f"  {phase_name}: CPU={cpu_frac:.6f}, GPU={gpu_frac:.6f}, diff={abs(cpu_frac-gpu_frac):.6f}")
    
    print(f"\nComposition Comparison:")
    cpu_x = result_cpu.X.values
    gpu_x = result_gpu.X.values
    
    if cpu_x.shape == gpu_x.shape:
        for comp_idx in range(cpu_x.shape[-1]):
            comp_name = comps[comp_idx] if comp_idx < len(comps) else f"Comp_{comp_idx}"
            cpu_comp = cpu_x.flatten()[comp_idx] if cpu_x.size > comp_idx else 0
            gpu_comp = gpu_x.flatten()[comp_idx] if gpu_x.size > comp_idx else 0
            print(f"  X({comp_name}): CPU={cpu_comp:.6f}, GPU={gpu_comp:.6f}, diff={abs(cpu_comp-gpu_comp):.6f}")
    
    # Check if different phases are stable
    cpu_stable = set(idx for idx, frac in enumerate(cpu_phases) if frac > 1e-12)
    gpu_stable = set(idx for idx, frac in enumerate(gpu_phases) if frac > 1e-12)
    
    if cpu_stable != gpu_stable:
        print(f"\n⚠️  DIFFERENT STABLE PHASES DETECTED!")
        print(f"  CPU stable phases: {[phases[i].name for i in cpu_stable]}")
        print(f"  GPU stable phases: {[phases[i].name for i in gpu_stable]}")
        print(f"  This explains the large free energy difference!")
    else:
        print(f"\n✓ Same phases are stable in both CPU and GPU calculations")
        
    # Check chemical potentials
    print(f"\nChemical Potential Comparison:")
    cpu_mu = result_cpu.MU.values.flatten()
    gpu_mu = result_gpu.MU.values.flatten()
    
    for i, (cpu_mu_val, gpu_mu_val) in enumerate(zip(cpu_mu, gpu_mu)):
        comp_name = comps[i] if i < len(comps) else f"Comp_{i}"
        print(f"  MU({comp_name}): CPU={cpu_mu_val:.3f}, GPU={gpu_mu_val:.3f}, diff={abs(cpu_mu_val-gpu_mu_val):.3f} J/mol")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)