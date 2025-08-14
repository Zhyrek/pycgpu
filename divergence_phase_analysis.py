#!/usr/bin/env python
"""Detailed phase and composition analysis for divergence region."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def analyze_equilibrium(result, label, phases):
    """Extract detailed phase and composition information from equilibrium result."""
    
    print(f"\n{label}")
    print("-" * 60)
    
    # Get GM
    gm = float(result.GM.values.squeeze())
    print(f"Total GM: {gm:.2f} J/mol")
    
    # Get overall composition
    try:
        x_overall = result.X.sel(component=['AL', 'CU', 'FE']).values.squeeze()
        if len(x_overall.shape) > 1:
            x_overall = x_overall[0]
        print(f"Overall composition: AL={x_overall[0]:.4f}, CU={x_overall[1]:.4f}, FE={x_overall[2]:.4f}")
    except:
        print("Could not extract overall composition")
    
    # Analyze each phase
    print("\nPhase Analysis:")
    active_phases = []
    
    for phase in phases:
        try:
            # Get phase fraction
            np_val = float(result.NP.sel(phase=phase).values.squeeze())
            
            if np_val > 1e-6:  # Phase is present
                active_phases.append(phase)
                print(f"\n  {phase}:")
                print(f"    Phase fraction: {np_val:.6f}")
                
                # Get phase composition
                try:
                    x_phase = result.X.sel(phase=phase, component=['AL', 'CU', 'FE']).values.squeeze()
                    if x_phase.size == 3:
                        print(f"    Composition: AL={x_phase[0]:.6f}, CU={x_phase[1]:.6f}, FE={x_phase[2]:.6f}")
                        # Check sum
                        total = x_phase[0] + x_phase[1] + x_phase[2]
                        if abs(total - 1.0) > 0.01:
                            print(f"    WARNING: Compositions sum to {total:.6f}")
                except Exception as e:
                    print(f"    Could not extract phase composition: {e}")
                
                # Get phase energy
                try:
                    gm_phase = float(result.GM.sel(phase=phase).values.squeeze())
                    print(f"    GM: {gm_phase:.2f} J/mol")
                except:
                    pass
                
                # Get site fractions if available
                try:
                    y_phase = result.Y.sel(phase=phase).values.squeeze()
                    if y_phase.size > 0:
                        print(f"    Site fractions shape: {y_phase.shape}")
                        # Print first few site fractions as example
                        if y_phase.ndim == 1:
                            print(f"    First site fractions: {y_phase[:min(4, len(y_phase))]}")
                except:
                    pass
                    
        except Exception as e:
            # Phase not in result or error accessing it
            pass
    
    if not active_phases:
        print("  No active phases detected (all phase fractions < 1e-6)")
    else:
        print(f"\nActive phases summary: {active_phases}")
    
    return active_phases

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['FCC_A1', 'BCC_A2', 'BCC_B2', 'L12_FCC']
    
    print("=" * 80)
    print("PHASE AND COMPOSITION ANALYSIS - DIVERGENCE REGION")
    print("=" * 80)
    print(f"Phase set: {phases}")
    print(f"Temperature: 600K, Pressure: 101325 Pa")
    
    # Test three key points
    test_points = [
        (0.390, 0.40, "BEFORE DIVERGENCE (both agree)"),
        (0.400, 0.40, "INSIDE DIVERGENCE (peak difference)"),
        (0.405, 0.40, "AFTER DIVERGENCE (both agree)")
    ]
    
    for x_al, x_cu, description in test_points:
        x_fe = 1.0 - x_al - x_cu
        
        print("\n" + "=" * 80)
        print(f"{description}")
        print(f"X(AL)={x_al:.3f}, X(CU)={x_cu:.3f}, X(FE)={x_fe:.3f}")
        print("=" * 80)
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: 600,
            v.P: 101325
        }
        
        # CPU calculation
        cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_phases = analyze_equilibrium(cpu_result, "CPU RESULT", phases)
        
        # GPU calculation  
        gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_phases = analyze_equilibrium(gpu_result, "GPU RESULT", phases)
        
        # Compare
        cpu_gm = float(cpu_result.GM.values.squeeze())
        gpu_gm = float(gpu_result.GM.values.squeeze())
        diff = gpu_gm - cpu_gm
        
        print("\n" + "-" * 60)
        print("COMPARISON:")
        print(f"  GM difference: {diff:.2f} J/mol")
        if cpu_phases != gpu_phases:
            print(f"  Phase assemblage DIFFERS:")
            print(f"    CPU phases: {cpu_phases}")
            print(f"    GPU phases: {gpu_phases}")
        else:
            print(f"  Same phase assemblage: {cpu_phases}")
    
    # Additional analysis - scan the transition region more finely
    print("\n" + "=" * 80)
    print("FINE SCAN OF TRANSITION REGION")
    print("=" * 80)
    
    print("\nX(AL)    CPU Phases                  GPU Phases                  Difference")
    print("-" * 80)
    
    for x_al in np.linspace(0.392, 0.404, 13):
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): 0.40,
            v.T: 600,
            v.P: 101325
        }
        
        cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        
        # Get active phases
        cpu_active = []
        gpu_active = []
        
        for phase in phases:
            try:
                cpu_np = float(cpu_result.NP.sel(phase=phase).values.squeeze())
                if cpu_np > 1e-6:
                    cpu_active.append(f"{phase}({cpu_np:.3f})")
            except:
                pass
                
            try:
                gpu_np = float(gpu_result.NP.sel(phase=phase).values.squeeze())
                if gpu_np > 1e-6:
                    gpu_active.append(f"{phase}({gpu_np:.3f})")
            except:
                pass
        
        cpu_str = ', '.join(cpu_active) if cpu_active else "None"
        gpu_str = ', '.join(gpu_active) if gpu_active else "None"
        
        diff_marker = "SAME" if cpu_active == gpu_active else "DIFF"
        
        print(f"{x_al:.3f}  {cpu_str:30s}  {gpu_str:30s}  {diff_marker}")

if __name__ == "__main__":
    main()