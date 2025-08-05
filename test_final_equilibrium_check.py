#!/usr/bin/env python
"""Check final equilibrium results for 6-phase case."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_final_results():
    """Compare final equilibrium results."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("FINAL EQUILIBRIUM RESULTS - 6 PHASE CASE")
    print("="*70)
    
    # CPU calculation
    print("\n--- CPU CALCULATION ---")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = result_cpu.GM.values[0,0,0,0]
    cpu_phases = result_cpu.Phase.values[0,0,0,0]
    cpu_np = result_cpu.NP.values[0,0,0,0]
    cpu_mu = result_cpu.MU.values[0,0,0,0]
    
    print(f"GM: {cpu_gm:.6f}")
    print(f"MU: {cpu_mu}")
    print("Active phases:")
    cpu_active = []
    for phase, amount in zip(cpu_phases, cpu_np):
        if phase != '' and amount > 1e-8:
            print(f"  {phase}: {amount:.6f}")
            cpu_active.append((phase, amount))
    
    # GPU calculation - with less verbose output
    print("\n--- GPU CALCULATION ---")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values[0,0,0,0]
    gpu_phases = result_gpu.Phase.values[0,0,0,0]
    gpu_np = result_gpu.NP.values[0,0,0,0]
    gpu_mu = result_gpu.MU.values[0,0,0,0]
    
    print(f"GM: {gpu_gm:.6f}")
    print(f"MU: {gpu_mu}")
    print("Active phases:")
    gpu_active = []
    for phase, amount in zip(gpu_phases, gpu_np):
        if phase != '' and amount > 1e-8:
            print(f"  {phase}: {amount:.6f}")
            gpu_active.append((phase, amount))
    
    # Analysis
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    gm_diff = abs(gpu_gm - cpu_gm)
    print(f"GM difference: {gm_diff:.2e}")
    
    if gm_diff > 1.0:
        print("❌ Large GM difference - different equilibria found")
    else:
        print("✅ Similar GM values")
    
    # Phase comparison
    cpu_phase_set = set(p[0] for p in cpu_active)
    gpu_phase_set = set(p[0] for p in gpu_active)
    
    if cpu_phase_set != gpu_phase_set:
        print("\n❌ Different phase assemblages:")
        print(f"  CPU: {cpu_phase_set}")
        print(f"  GPU: {gpu_phase_set}")
        
        only_cpu = cpu_phase_set - gpu_phase_set
        only_gpu = gpu_phase_set - cpu_phase_set
        
        if only_cpu:
            print(f"  Only in CPU: {only_cpu}")
        if only_gpu:
            print(f"  Only in GPU: {only_gpu}")
            
        if 'HCP_A3' in only_gpu and 'AU2BI_C15' in only_cpu:
            print("\n🔍 KEY FINDING:")
            print("  GPU has HCP_A3 instead of AU2BI_C15")
            print("  Despite starting with same phases (FCC_A1 + AU2BI_C15),")
            print("  the GPU solver converged to a different equilibrium.")
            print("\n  Possible reasons:")
            print("  1. Different numerical behavior in solver iterations")
            print("  2. Different handling of phases with fractional sublattices")
            print("  3. Phase switching logic differences")
            print("  4. Numerical precision in phase stability checks")
    else:
        print("✅ Same phase assemblages")
    
    # Energy comparison
    print(f"\n--- Energy Analysis ---")
    print(f"CPU GM: {cpu_gm:.6f}")
    print(f"GPU GM: {gpu_gm:.6f}")
    
    if gpu_gm < cpu_gm - 1e-6:
        print("⚠️  GPU found lower energy state")
        print("   This could indicate CPU missed the global minimum")
    elif cpu_gm < gpu_gm - 1e-6:
        print("⚠️  CPU found lower energy state")
        print("   This could indicate GPU missed the global minimum")
    else:
        print("   Similar energies - likely same thermodynamic state")

if __name__ == "__main__":
    test_final_results()