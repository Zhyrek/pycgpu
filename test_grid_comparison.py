#!/usr/bin/env python
"""Compare grid inputs to starting_point between CPU and GPU paths."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, calculate, variables as v
from pycalphad.core.utils import unpack_condition, unpack_kwarg
import warnings
warnings.filterwarnings("ignore")

def compare_grids():
    """Compare grid calculations between CPU and GPU paths."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("GRID COMPARISON FOR STARTING_POINT")
    print("="*70)
    
    # Replicate CPU grid calculation
    str_conds = {str(k): v for k, v in conditions.items()}
    state_variables = [c for c in conditions.keys() if c in [v.T, v.P, v.N]]
    
    grid_opts = {'pdens': 60}
    statevar_strings = [str(x) for x in state_variables]
    grid_opts.update({key: value for key, value in str_conds.items() if key in statevar_strings})
    
    print("\nCalculating grid with same parameters as CPU/GPU...")
    print(f"  Components: {comps}")
    print(f"  Phases: {phases}")
    print(f"  Grid options: {grid_opts}")
    
    # Calculate grid
    grid = calculate(dbf, comps, phases, output='GM', 
                    fake_points=True, to_xarray=False, 
                    conditions=conditions, **grid_opts)
    
    print(f"\nGrid properties:")
    print(f"  GM shape: {grid.GM.shape}")
    print(f"  Phase shape: {grid.Phase.shape}")
    print(f"  Y shape: {grid.Y.shape}")
    
    # Analyze phase distribution in grid
    phase_counts = {}
    phase_data = grid.Phase if isinstance(grid.Phase, np.ndarray) else grid.Phase.values
    gm_data = grid.GM if isinstance(grid.GM, np.ndarray) else grid.GM.values
    y_data = grid.Y if isinstance(grid.Y, np.ndarray) else grid.Y.values
    
    for phase in np.unique(phase_data):
        if phase != '' and phase != '_FAKE_':
            count = np.sum(phase_data == phase)
            phase_counts[phase] = count
    
    print(f"\nGrid points per phase:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count} points")
    
    # Check HCP_A3 specifically
    print(f"\nHCP_A3 analysis:")
    hcp_mask = phase_data.flatten() == 'HCP_A3'
    if np.any(hcp_mask):
        hcp_gm = gm_data.flatten()[hcp_mask]
        hcp_y = y_data.reshape(-1, y_data.shape[-1])[hcp_mask]
        
        print(f"  HCP_A3 GM range: [{np.min(hcp_gm):.2f}, {np.max(hcp_gm):.2f}]")
        print(f"  Number of HCP_A3 points: {len(hcp_gm)}")
        
        # Find lowest energy HCP_A3 configuration
        min_idx = np.argmin(hcp_gm)
        print(f"  Lowest energy HCP_A3:")
        print(f"    GM = {hcp_gm[min_idx]:.2f}")
        print(f"    Y = {hcp_y[min_idx]}")
        
        # Check if any HCP_A3 points are competitive
        all_gm = gm_data.flatten()
        overall_min = np.min(all_gm[all_gm < 1e9])  # Exclude fake points
        print(f"\n  Overall minimum GM in grid: {overall_min:.2f}")
        print(f"  HCP_A3 min GM - overall min GM = {hcp_gm[min_idx] - overall_min:.2f}")
        
        # Find which phase has the overall minimum
        min_phase_idx = np.argmin(all_gm[all_gm < 1e9])
        min_phase = phase_data.flatten()[all_gm < 1e9][min_phase_idx]
        print(f"  Phase with overall minimum: {min_phase}")
    
    # Check AU2BI_C15 for comparison
    print(f"\nAU2BI_C15 analysis:")
    c15_mask = phase_data.flatten() == 'AU2BI_C15'
    if np.any(c15_mask):
        c15_gm = gm_data.flatten()[c15_mask]
        c15_y = y_data.reshape(-1, y_data.shape[-1])[c15_mask]
        
        print(f"  AU2BI_C15 GM range: [{np.min(c15_gm):.2f}, {np.max(c15_gm):.2f}]")
        print(f"  Number of AU2BI_C15 points: {len(c15_gm)}")
        
        # Find lowest energy AU2BI_C15 configuration
        min_idx = np.argmin(c15_gm)
        print(f"  Lowest energy AU2BI_C15:")
        print(f"    GM = {c15_gm[min_idx]:.2f}")
        print(f"    Y = {c15_y[min_idx]}")
    
    # Check FCC_A1 too
    print(f"\nFCC_A1 analysis:")
    fcc_mask = phase_data.flatten() == 'FCC_A1'
    if np.any(fcc_mask):
        fcc_gm = gm_data.flatten()[fcc_mask]
        fcc_y = y_data.reshape(-1, y_data.shape[-1])[fcc_mask]
        
        print(f"  FCC_A1 GM range: [{np.min(fcc_gm):.2f}, {np.max(fcc_gm):.2f}]")
        print(f"  Number of FCC_A1 points: {len(fcc_gm)}")
        
        # Find lowest energy FCC_A1 configuration  
        min_idx = np.argmin(fcc_gm)
        print(f"  Lowest energy FCC_A1:")
        print(f"    GM = {fcc_gm[min_idx]:.2f}")
        print(f"    Y = {fcc_y[min_idx]}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print("The grid input to starting_point() should be identical for CPU and GPU.")
    print("If the convex hull selects different phases, it must be due to:")
    print("  1. Different numerical precision in convex hull calculation")
    print("  2. Different tie-breaking behavior when phases have similar energies")
    print("  3. Issues with phase amount normalization affecting the hull")
    
    return grid

if __name__ == "__main__":
    grid = compare_grids()