#!/usr/bin/env python
"""Debug the phase equilibrium to understand the matrix differences."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

def debug_phase_equilibrium():
    """Debug which phases are stable and why GM differs."""
    
    print("DEBUGGING PHASE EQUILIBRIUM DIFFERENCES")
    print("="*50)
    
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    # Test with ALL major phases to see what's stable
    phases_to_test = ['LIQUID', 'FCC_A1', 'BCC_A2', 'L12']
    
    print("CPU equilibrium:")
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], phases_to_test, conditions, gpu=False)
    print(f"  GM: {float(eq_cpu.GM.values.flatten()[0]):.6f} J/mol")
    
    # Check which phases are stable
    phase_data_cpu = eq_cpu.Phase.values.flatten()
    phase_fractions_cpu = eq_cpu.NP.values.flatten()
    
    print("  Stable phases (CPU):")
    for i, (phase, fraction) in enumerate(zip(phase_data_cpu, phase_fractions_cpu)):
        if isinstance(fraction, (int, float)) and fraction > 1e-6:
            print(f"    {phase}: {fraction:.6f} mole fraction")
    
    print("\nGPU equilibrium:")
    eq_gpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], phases_to_test, conditions, gpu=True)
    print(f"  GM: {float(eq_gpu.GM.values.flatten()[0]):.6f} J/mol")
    
    # Check which phases are stable  
    phase_data_gpu = eq_gpu.Phase.values.flatten()
    phase_fractions_gpu = eq_gpu.NP.values.flatten()
    
    print("  Stable phases (GPU):")
    for i, (phase, fraction) in enumerate(zip(phase_data_gpu, phase_fractions_gpu)):
        if isinstance(fraction, (int, float)) and fraction > 1e-6:
            print(f"    {phase}: {fraction:.6f} mole fraction")
    
    # Compare GM values
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    gpu_gm = float(eq_gpu.GM.values.flatten()[0])
    diff = abs(cpu_gm - gpu_gm)
    
    print(f"\nDifference: {diff:.6f} J/mol")
    
    # Now test LIQUID phase only (which is what we were testing before)
    print(f"\n" + "="*50)
    print("LIQUID PHASE ONLY COMPARISON:")
    print("="*50)
    
    print("CPU (LIQUID only):")
    eq_cpu_liquid = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False)
    cpu_gm_liquid = float(eq_cpu_liquid.GM.values.flatten()[0])
    print(f"  GM: {cpu_gm_liquid:.6f} J/mol")
    
    print("GPU (LIQUID only):")
    eq_gpu_liquid = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=True)
    gpu_gm_liquid = float(eq_gpu_liquid.GM.values.flatten()[0])
    print(f"  GM: {gpu_gm_liquid:.6f} J/mol")
    
    liquid_diff = abs(cpu_gm_liquid - gpu_gm_liquid)
    print(f"  Difference: {liquid_diff:.6f} J/mol")
    
    print(f"\n" + "="*50)
    print("ANALYSIS:")
    print("="*50)
    
    if diff < liquid_diff:
        print("Multi-phase equilibrium has smaller error than single LIQUID phase.")
        print("This suggests the issue may be specific to single-phase calculations.")
    else:
        print("Single LIQUID phase has smaller or equal error to multi-phase.")
        print("This is the comparison we should focus on.")
    
    if liquid_diff < 50:
        print(f"✓ LIQUID-only difference ({liquid_diff:.1f} J/mol) is reasonable")
        print("  The matrix construction appears to be working correctly")
        print("  Small differences likely due to numerical precision")
    else:
        print(f"○ LIQUID-only difference ({liquid_diff:.1f} J/mol) suggests matrix issues")
        print("  CPU and GPU are constructing different equilibrium systems")
        print("  Need to investigate matrix construction more closely")
    
    return liquid_diff

if __name__ == "__main__":
    debug_phase_equilibrium()