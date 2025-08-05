#!/usr/bin/env python
"""Test driving force calculation to understand GPU issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, calculate, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_driving_forces():
    """Calculate driving forces for all phases."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("DRIVING FORCE ANALYSIS")
    print("="*70)
    
    # Calculate grid
    grid = calculate(dbf, comps, phases, output='GM', 
                    fake_points=True, to_xarray=False, 
                    conditions=conditions, pdens=60)
    
    # Chemical potentials from starting point
    mu = np.array([-19395.97354416, -23494.6194504])
    
    print(f"\nChemical potentials from starting point: {mu}")
    
    # Calculate driving forces
    X = grid.X if isinstance(grid.X, np.ndarray) else grid.X.values
    GM = grid.GM if isinstance(grid.GM, np.ndarray) else grid.GM.values
    Phase = grid.Phase if isinstance(grid.Phase, np.ndarray) else grid.Phase.values
    
    # Flatten arrays
    X_flat = X.reshape(-1, X.shape[-1])
    GM_flat = GM.flatten()
    Phase_flat = Phase.flatten()
    
    # Remove vacancy component (last component)
    X_nonvac = X_flat[:, :2]
    
    # Calculate driving forces
    driving_forces = np.dot(X_nonvac, mu) - GM_flat
    
    # Group by phase
    phase_df = {}
    for phase in phases:
        mask = Phase_flat == phase
        if np.any(mask):
            df_phase = driving_forces[mask]
            gm_phase = GM_flat[mask]
            # Filter out fake points
            valid_mask = gm_phase < 1e9
            if np.any(valid_mask):
                df_valid = df_phase[valid_mask]
                max_df = np.max(df_valid)
                phase_df[phase] = max_df
                print(f"\n{phase}:")
                print(f"  Number of grid points: {np.sum(valid_mask)}")
                print(f"  Max driving force: {max_df:.6f}")
                print(f"  Would be added: {'YES' if max_df >= -1000 else 'NO'}")
            
            # Find the point with max driving force
            if max_df >= -1000:
                idx = np.argmax(df_phase)
                all_indices = np.where(mask)[0]
                global_idx = all_indices[idx]
                print(f"  X at max DF: {X_nonvac[global_idx]}")
                print(f"  GM at max DF: {GM_flat[global_idx]:.6f}")
    
    # Show which phases would be added
    print("\n" + "="*50)
    print("PHASES THAT WOULD BE ADDED (DF >= -1000):")
    for phase, df in phase_df.items():
        if df >= -1000:
            print(f"  {phase}: DF = {df:.6f}")

if __name__ == "__main__":
    test_driving_forces()