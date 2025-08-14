#!/usr/bin/env python
"""Compare starting points between CPU and GPU for the failing AlCuFe case."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, calculate, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def get_cpu_starting_point(dbf, comps, phases, conditions):
    """Extract starting point calculation from CPU pathway."""
    
    # Calculate starting point - this is what provides initial values
    print("\nCPU CALCULATE STEP:")
    print("-" * 60)
    
    # Convert conditions to kwargs format
    calc_kwargs = {}
    for key, val in conditions.items():
        if hasattr(key, 'name'):
            calc_kwargs[key.name] = val
        else:
            calc_kwargs[str(key)] = val
    
    calc_result = calculate(dbf, comps, phases, output='GM', **calc_kwargs)
    
    # Extract GM values for each phase
    phase_gms = {}
    for phase in phases:
        try:
            gm_vals = calc_result.GM.sel(phase=phase).values.squeeze()
            if gm_vals.size > 0:
                phase_gms[phase] = float(np.min(gm_vals))
                print(f"  {phase:12s}: GM = {phase_gms[phase]:10.2f} J/mol")
        except:
            pass
    
    # Find minimum GM phases
    if phase_gms:
        min_phase = min(phase_gms, key=phase_gms.get)
        print(f"\nLowest GM phase: {min_phase} ({phase_gms[min_phase]:.2f} J/mol)")
    
    return calc_result, phase_gms

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC', 
              'ALCU_THETA', 'AL13FE4_D03', 'AL5FE2_D82']
    
    # Failing condition
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.40,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("STARTING POINT ANALYSIS FOR FAILING CONDITION")
    print("=" * 80)
    print(f"X(AL)={conditions[v.X('AL')]:.2f}, X(CU)={conditions[v.X('CU')]:.2f}, X(FE)=0.20, T={conditions[v.T]}K")
    
    # Get CPU starting point
    calc_result, phase_gms = get_cpu_starting_point(dbf, comps, phases, conditions)
    
    # Now run full equilibrium for both
    print("\n" + "=" * 80)
    print("FULL EQUILIBRIUM RESULTS")
    print("=" * 80)
    
    print("\nCPU Equilibrium:")
    cpu_eq = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = float(cpu_eq.GM.values.squeeze())
    print(f"  Final GM = {cpu_gm:.2f} J/mol")
    
    # Get active phases
    cpu_phases = {}
    for phase in phases:
        try:
            np_val = float(cpu_eq.NP.sel(phase=phase).values.squeeze())
            if np_val > 1e-6:
                cpu_phases[phase] = np_val
        except:
            pass
    print(f"  Active phases: {list(cpu_phases.keys())}")
    
    print("\nGPU Equilibrium:")
    gpu_eq = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = float(gpu_eq.GM.values.squeeze())
    print(f"  Final GM = {gpu_gm:.2f} J/mol")
    
    # Get active phases
    gpu_phases = {}
    for phase in phases:
        try:
            np_val = float(gpu_eq.NP.sel(phase=phase).values.squeeze())
            if np_val > 1e-6:
                gpu_phases[phase] = np_val
        except:
            pass
    print(f"  Active phases: {list(gpu_phases.keys())}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Compare starting vs final
    print("\nStarting point ranking (by GM):")
    sorted_phases = sorted(phase_gms.items(), key=lambda x: x[1])
    for i, (phase, gm) in enumerate(sorted_phases, 1):
        in_cpu = "✓" if phase in cpu_phases else " "
        in_gpu = "✓" if phase in gpu_phases else " "
        print(f"  {i}. {phase:12s}: {gm:10.2f} J/mol  [CPU:{in_cpu} GPU:{in_gpu}]")
    
    print(f"\nGM Difference: {gpu_gm - cpu_gm:.2f} J/mol")
    
    # Test nearby points to see pattern
    print("\n" + "=" * 80)
    print("TESTING SLIGHT VARIATIONS")
    print("=" * 80)
    
    # Test with slight perturbations
    for dal in [-0.001, 0, 0.001]:
        for dcu in [-0.001, 0, 0.001]:
            if dal == 0 and dcu == 0:
                continue
                
            test_conds = {
                v.X('AL'): 0.40 + dal,
                v.X('CU'): 0.40 + dcu,
                v.T: 600,
                v.P: 101325
            }
            
            cpu_test = equilibrium(dbf, comps, phases, test_conds, gpu=False, verbose=False)
            gpu_test = equilibrium(dbf, comps, phases, test_conds, gpu=True, verbose=False)
            
            cpu_test_gm = float(cpu_test.GM.values.squeeze())
            gpu_test_gm = float(gpu_test.GM.values.squeeze())
            diff = gpu_test_gm - cpu_test_gm
            
            status = "MATCH" if abs(diff) < 1.0 else "DIFF"
            print(f"  X(AL)={0.40+dal:.3f}, X(CU)={0.40+dcu:.3f}: "
                  f"Δ={diff:+6.2f} J/mol [{status}]")

if __name__ == "__main__":
    main()