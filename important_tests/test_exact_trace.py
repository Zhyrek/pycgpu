#!/usr/bin/env python
"""Exact trace of the failing condition with correct phases."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Enable debugging
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

def main():
    """Trace exact failing condition."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2']  # EXACT phase list from failing test
    
    # The exact failing condition
    conditions = {
        v.X('AL'): 0.2,
        v.X('CU'): 0.5,
        v.T: 900,
        v.P: 101325
    }
    
    print("=" * 80)
    print("EXACT TRACE - CORRECT PHASE LIST")
    print("=" * 80)
    print(f"\nPhases: {phases}")
    print(f"Condition: X(AL)={conditions[v.X('AL')]}, X(CU)={conditions[v.X('CU')]}, T={conditions[v.T]}K")
    
    print("\n" + "-" * 40)
    print("CPU CALCULATION:")
    print("-" * 40)
    
    cpu_result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=False, verbose=False)
    
    cpu_gm = cpu_result.GM.values.item()
    cpu_np = cpu_result.NP.values.flatten()
    
    print(f"CPU GM: {cpu_gm:.6f} J/mol")
    print(f"CPU phase amounts:")
    for i, phase in enumerate(phases):
        if i < len(cpu_np) and not np.isnan(cpu_np[i]):
            print(f"  {phase}: {cpu_np[i]:.6f}")
    
    # Extract phase compositions
    print(f"\nCPU phase compositions (X values):")
    cpu_x = cpu_result.X.values
    print(f"  Shape: {cpu_x.shape}")
    # X shape is typically (points, phases, components)
    if len(cpu_x.shape) >= 2:
        for p_idx in range(min(len(phases), cpu_x.shape[0])):
            phase = phases[p_idx]
            print(f"  {phase}:")
            for c_idx, comp in enumerate(['AL', 'CU', 'FE']):
                if c_idx < cpu_x.shape[-1]:
                    val = cpu_x.flatten()[p_idx * cpu_x.shape[-1] + c_idx]
                    if not np.isnan(val):
                        print(f"    X({comp}): {val:.6f}")
    
    print("\n" + "-" * 40)
    print("GPU CALCULATION:")
    print("-" * 40)
    
    gpu_result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50},
                            gpu=True, verbose=True)
    
    gpu_gm = gpu_result.GM.values.item()
    gpu_np = gpu_result.NP.values.flatten()
    
    print(f"\nGPU GM: {gpu_gm:.6f} J/mol")
    print(f"GPU phase amounts:")
    for i, phase in enumerate(phases):
        if i < len(gpu_np) and not np.isnan(gpu_np[i]):
            print(f"  {phase}: {gpu_np[i]:.6f}")
    
    # Extract phase compositions
    print(f"\nGPU phase compositions (X values):")
    gpu_x = gpu_result.X.values
    print(f"  Shape: {gpu_x.shape}")
    if len(gpu_x.shape) >= 2:
        for p_idx in range(min(len(phases), gpu_x.shape[0])):
            phase = phases[p_idx]
            print(f"  {phase}:")
            for c_idx, comp in enumerate(['AL', 'CU', 'FE']):
                if c_idx < gpu_x.shape[-1]:
                    val = gpu_x.flatten()[p_idx * gpu_x.shape[-1] + c_idx]
                    if not np.isnan(val):
                        print(f"    X({comp}): {val:.6f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    
    gm_diff = abs(gpu_gm - cpu_gm)
    print(f"GM difference: {gm_diff:.6f} J/mol")
    
    print("\nPhase amount differences:")
    for i, phase in enumerate(phases):
        if i < len(cpu_np) and i < len(gpu_np):
            if not np.isnan(cpu_np[i]) and not np.isnan(gpu_np[i]):
                diff = abs(cpu_np[i] - gpu_np[i])
                print(f"  {phase}: {diff:.6f}")
    
    print("\nPhase composition differences:")
    if len(cpu_x.shape) >= 2 and len(gpu_x.shape) >= 2:
        for p_idx in range(min(len(phases), cpu_x.shape[0], gpu_x.shape[0])):
            phase = phases[p_idx]
            print(f"  {phase}:")
            for c_idx, comp in enumerate(['AL', 'CU', 'FE']):
                if c_idx < cpu_x.shape[-1] and c_idx < gpu_x.shape[-1]:
                    cpu_val = cpu_x.flatten()[p_idx * cpu_x.shape[-1] + c_idx]
                    gpu_val = gpu_x.flatten()[p_idx * gpu_x.shape[-1] + c_idx]
                    if not np.isnan(cpu_val) and not np.isnan(gpu_val):
                        diff = abs(cpu_val - gpu_val)
                        print(f"    X({comp}) diff: {diff:.6f}")
    
    # Check for near-duplicate compositions
    print("\n" + "-" * 40)
    print("CHECKING FOR DUPLICATE PHASE COMPOSITIONS:")
    print("-" * 40)
    
    for result_name, x_vals in [("CPU", cpu_x), ("GPU", gpu_x)]:
        print(f"\n{result_name}:")
        active_phases = []
        if len(x_vals.shape) >= 2:
            for p_idx in range(min(len(phases), x_vals.shape[0])):
                phase = phases[p_idx]
                comp_vector = []
                for c_idx in range(min(3, x_vals.shape[-1])):  # AL, CU, FE
                    val = x_vals.flatten()[p_idx * x_vals.shape[-1] + c_idx]
                    if not np.isnan(val):
                        comp_vector.append(val)
                if comp_vector and len(comp_vector) == 3:
                    active_phases.append((phase, comp_vector))
        
        # Check for duplicates
        for i, (phase1, comp1) in enumerate(active_phases):
            for j, (phase2, comp2) in enumerate(active_phases[i+1:], i+1):
                diff = np.linalg.norm(np.array(comp1) - np.array(comp2))
                if diff < 0.05:  # Within 5% composition
                    print(f"  ⚠ {phase1} and {phase2} have similar compositions!")
                    print(f"    {phase1}: AL={comp1[0]:.4f}, CU={comp1[1]:.4f}, FE={comp1[2]:.4f}")
                    print(f"    {phase2}: AL={comp2[0]:.4f}, CU={comp2[1]:.4f}, FE={comp2[2]:.4f}")
                    print(f"    Difference: {diff:.6f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()