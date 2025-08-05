#!/usr/bin/env python
"""Compare starting points between CPU and GPU for the HCP_A3 issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, calculate, variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.utils import unpack_condition
import warnings
warnings.filterwarnings("ignore")

def compare_starting_points():
    """Compare starting point calculations between different phase sets."""
    
    # Load database
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    # Test cases
    test_cases = [
        (['FCC_A1', 'AU2BI_C15', 'BCC_A2', 'HCP_A3'], "4-phase case"),
        (['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7'], "6-phase case"),
    ]
    
    for phases, case_name in test_cases:
        print(f"\n{'='*60}")
        print(f"{case_name}")
        print(f"Phases: {phases}")
        print(f"{'='*60}")
        
        # Calculate grid for starting point
        calc_conditions = {
            v.T: conditions[v.T],
            v.P: conditions[v.P],
            v.X('BI'): conditions[v.X('BI')]
        }
        grid = calculate(dbf, comps, phases, output='GM', 
                        pdens=60, **calc_conditions)
        
        print(f"\nGrid info:")
        print(f"  Shape: {grid.GM.shape}")
        print(f"  Phases in grid: {np.unique(grid.Phase.values)}")
        
        # Count grid points per phase
        phase_counts = {}
        for phase in np.unique(grid.Phase.values):
            if phase != '' and phase != '_FAKE_':
                count = np.sum(grid.Phase.values == phase)
                phase_counts[phase] = count
        
        print(f"\nGrid points per phase:")
        for phase, count in sorted(phase_counts.items()):
            print(f"  {phase}: {count} points")
            
            # Check energy range for HCP_A3 specifically
            if phase == 'HCP_A3':
                hcp_mask = grid.Phase.values.flatten() == 'HCP_A3'
                hcp_energies = grid.GM.values.flatten()[hcp_mask]
                print(f"    HCP_A3 energy range: [{np.min(hcp_energies):.2f}, {np.max(hcp_energies):.2f}]")
                
                # Check if HCP_A3 has any low-energy points
                low_energy_count = np.sum(hcp_energies < -19000)
                print(f"    HCP_A3 points with GM < -19000: {low_energy_count}")
        
        # Calculate starting point
        state_vars = [v.T, v.P, v.N]
        conds_dict = unpack_condition(conditions)
        phase_records = {name: None for name in phases}  # Simplified
        
        try:
            print(f"\nCalculating starting point...")
            result = starting_point(conds_dict, state_vars, phase_records, grid)
            
            print(f"\nStarting point result:")
            print(f"  GM: {result.GM.values}")
            print(f"  MU: {result.MU.values}")
            print(f"  Phases: {result.Phase.values}")
            print(f"  NP: {result.NP.values}")
            
            # Identify active phases
            active_phases = []
            for i, (phase, np_val) in enumerate(zip(result.Phase.values.flatten(), 
                                                   result.NP.values.flatten())):
                if phase != '' and np_val > 1e-8:
                    active_phases.append(f"{phase}({np_val:.3f})")
            
            print(f"  Active phases: {active_phases}")
            
            # Check if HCP_A3 is active
            if any('HCP_A3' in p for p in active_phases):
                print(f"\n⚠️  HCP_A3 is selected in starting point!")
                print(f"  This is likely causing the GPU issue")
            
        except Exception as e:
            print(f"Starting point error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    compare_starting_points()