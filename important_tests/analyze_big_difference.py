#!/usr/bin/env python
"""Detailed analysis of the 706 J/mol difference case."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def extract_phase_compositions(result, phases_list):
    """Extract detailed phase information including compositions."""
    phase_labels = result.Phase.values.flatten()
    np_vals = result.NP.values.flatten()
    x_vals = result.X.values
    
    # Get dimensions
    n_comps = 3  # AL, CU, FE
    
    active_phases = []
    for i, (phase, amount) in enumerate(zip(phase_labels, np_vals)):
        if not np.isnan(amount) and amount > 0.001:
            # Extract composition for this phase
            try:
                # X array is shaped (conditions, phases, components)
                # We need to extract for this phase index
                x_flat = x_vals.reshape(-1, n_comps)
                if i < len(x_flat):
                    comp_dict = {
                        'AL': x_flat[i][0],
                        'CU': x_flat[i][1], 
                        'FE': x_flat[i][2]
                    }
                else:
                    comp_dict = {'AL': np.nan, 'CU': np.nan, 'FE': np.nan}
            except:
                comp_dict = {'AL': np.nan, 'CU': np.nan, 'FE': np.nan}
            
            active_phases.append({
                'name': phase,
                'amount': amount,
                'composition': comp_dict
            })
    
    return active_phases

def test_composition_variation(dbf, comps, phases, x_al_center, x_cu_center, temp):
    """Test how results vary with composition."""
    
    print("\n" + "=" * 80)
    print(f"COMPOSITION VARIATION ANALYSIS")
    print(f"Center: X(AL)={x_al_center:.2f}, X(CU)={x_cu_center:.2f}, T={temp}K")
    print("=" * 80)
    
    # Test points in a grid around center
    test_points = []
    for dal in [-0.02, -0.01, 0, 0.01, 0.02]:
        for dcu in [-0.02, -0.01, 0, 0.01, 0.02]:
            x_al = x_al_center + dal
            x_cu = x_cu_center + dcu
            if x_al >= 0 and x_cu >= 0 and x_al + x_cu < 1.0:
                test_points.append((x_al, x_cu, dal, dcu))
    
    results_table = []
    
    for x_al, x_cu, dal, dcu in test_points:
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        try:
            # CPU calculation
            cpu_result = equilibrium(dbf, comps, phases, conditions,
                                    gpu=False, verbose=False)
            cpu_gm = cpu_result.GM.values.item()
            cpu_phases = extract_phase_compositions(cpu_result, phases)
            
            # GPU calculation
            gpu_result = equilibrium(dbf, comps, phases, conditions,
                                    gpu=True, verbose=False)
            gpu_gm = gpu_result.GM.values.item()
            gpu_phases = extract_phase_compositions(gpu_result, phases)
            
            diff = abs(gpu_gm - cpu_gm)
            
            results_table.append({
                'x_al': x_al,
                'x_cu': x_cu,
                'dal': dal,
                'dcu': dcu,
                'cpu_gm': cpu_gm,
                'gpu_gm': gpu_gm,
                'diff': diff,
                'cpu_phases': cpu_phases,
                'gpu_phases': gpu_phases
            })
            
        except Exception as e:
            print(f"Error at X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}: {e}")
    
    # Print grid of energy differences
    print("\nEnergy Difference Map (J/mol):")
    print("     ", end="")
    for dcu in [-0.02, -0.01, 0, 0.01, 0.02]:
        print(f"  CU{dcu:+.2f}", end="")
    print()
    
    for dal in [-0.02, -0.01, 0, 0.01, 0.02]:
        print(f"AL{dal:+.2f}", end="")
        for dcu in [-0.02, -0.01, 0, 0.01, 0.02]:
            # Find result for this point
            result = next((r for r in results_table 
                          if r['dal'] == dal and r['dcu'] == dcu), None)
            if result:
                diff = result['diff']
                if diff > 100:
                    print(f"  {diff:6.0f}*", end="")
                else:
                    print(f"  {diff:6.0f} ", end="")
            else:
                print("     -- ", end="")
        print()
    
    print("\n(* indicates > 100 J/mol difference)")
    
    # Find and analyze the center point
    center = next((r for r in results_table if r['dal'] == 0 and r['dcu'] == 0), None)
    if center:
        print("\n" + "=" * 80)
        print("CENTER POINT DETAILED ANALYSIS")
        print("=" * 80)
        print(f"X(AL)={center['x_al']:.2f}, X(CU)={center['x_cu']:.2f}, X(FE)={1-center['x_al']-center['x_cu']:.2f}")
        print(f"CPU GM: {center['cpu_gm']:.1f} J/mol")
        print(f"GPU GM: {center['gpu_gm']:.1f} J/mol")
        print(f"Difference: {center['diff']:.1f} J/mol")
        
        print("\n--- CPU PHASES ---")
        for phase in center['cpu_phases']:
            print(f"{phase['name']:12s}: {phase['amount']:.3f} ({phase['amount']*100:.1f}%)")
            if not any(np.isnan(v) for v in phase['composition'].values()):
                print(f"  Composition: X(AL)={phase['composition']['AL']:.3f}, "
                      f"X(CU)={phase['composition']['CU']:.3f}, "
                      f"X(FE)={phase['composition']['FE']:.3f}")
        
        print("\n--- GPU PHASES ---")
        for phase in center['gpu_phases']:
            print(f"{phase['name']:12s}: {phase['amount']:.3f} ({phase['amount']*100:.1f}%)")
            if not any(np.isnan(v) for v in phase['composition'].values()):
                print(f"  Composition: X(AL)={phase['composition']['AL']:.3f}, "
                      f"X(CU)={phase['composition']['CU']:.3f}, "
                      f"X(FE)={phase['composition']['FE']:.3f}")
    
    # Check for discontinuities
    print("\n" + "=" * 80)
    print("DISCONTINUITY ANALYSIS")
    print("=" * 80)
    
    # Check if phase assemblages change suddenly
    phase_changes = []
    for r in results_table:
        cpu_names = set(p['name'] for p in r['cpu_phases'])
        gpu_names = set(p['name'] for p in r['gpu_phases'])
        if cpu_names != gpu_names:
            phase_changes.append((r['x_al'], r['x_cu'], cpu_names, gpu_names))
    
    if phase_changes:
        print("Points where CPU and GPU have different phase assemblages:")
        for x_al, x_cu, cpu_set, gpu_set in phase_changes[:5]:
            print(f"  X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}:")
            print(f"    CPU: {', '.join(cpu_set)}")
            print(f"    GPU: {', '.join(gpu_set)}")
    else:
        print("CPU and GPU have same phase assemblages at all test points")

def main():
    """Analyze the big difference case in detail."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("DETAILED ANALYSIS OF 706 J/mol DIFFERENCE CASE")
    print("=" * 80)
    
    # The problematic condition
    x_al = 0.60
    x_cu = 0.10
    temp = 600
    
    # First, get the exact phases and compositions at this point
    conditions = {
        v.X('AL'): x_al,
        v.X('CU'): x_cu,
        v.T: temp,
        v.P: 101325
    }
    
    print("\nCalculating equilibrium at X(AL)=0.60, X(CU)=0.10, T=600K...")
    
    # CPU calculation
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=False, verbose=False)
    cpu_phases = extract_phase_compositions(cpu_result, phases)
    
    # GPU calculation  
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            gpu=True, verbose=False)
    gpu_phases = extract_phase_compositions(gpu_result, phases)
    
    print("\n" + "=" * 80)
    print("PHASES PRESENT")
    print("=" * 80)
    
    print("\nCPU RESULT:")
    print(f"GM = {cpu_result.GM.values.item():.1f} J/mol")
    print("Active phases:")
    for phase in cpu_phases:
        print(f"  {phase['name']:12s}: {phase['amount']:.3f} ({phase['amount']*100:.1f}%)")
        if not any(np.isnan(v) for v in phase['composition'].values()):
            print(f"    X(AL)={phase['composition']['AL']:.3f}, "
                  f"X(CU)={phase['composition']['CU']:.3f}, "
                  f"X(FE)={phase['composition']['FE']:.3f}")
    
    print("\nGPU RESULT:")
    print(f"GM = {gpu_result.GM.values.item():.1f} J/mol")
    print("Active phases:")
    for phase in gpu_phases:
        print(f"  {phase['name']:12s}: {phase['amount']:.3f} ({phase['amount']*100:.1f}%)")
        if not any(np.isnan(v) for v in phase['composition'].values()):
            print(f"    X(AL)={phase['composition']['AL']:.3f}, "
                  f"X(CU)={phase['composition']['CU']:.3f}, "
                  f"X(FE)={phase['composition']['FE']:.3f}")
    
    print(f"\nEnergy difference: {abs(cpu_result.GM.values.item() - gpu_result.GM.values.item()):.1f} J/mol")
    
    # Now test composition variation
    test_composition_variation(dbf, comps, phases, x_al, x_cu, temp)

if __name__ == "__main__":
    main()