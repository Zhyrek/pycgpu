#!/usr/bin/env python
"""
Debug script to understand internal DOF for Nb-Ti system.
"""

import numpy as np
from pycalphad import Database, equilibrium, calculate, variables as v
from pycalphad.core.workspace import Workspace
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("Nb-Ti Internal DOF Analysis")
    print("="*70)
    
    try:
        dbf_nbti = Database("NbTi.tdb")
        
        # Check phase model structure
        print("\nPhase structures in Nb-Ti database:")
        for phase_name in ['BCC_A2', 'HCP_A3']:
            if phase_name in dbf_nbti.phases:
                phase = dbf_nbti.phases[phase_name]
                print(f"\n{phase_name}:")
                print(f"  Sublattices: {len(phase.sublattices)}")
                print(f"  Constituents: {phase.constituents}")
                print(f"  Sublattice model: {phase.sublattices}")
                
        # Run equilibrium to see actual DOF
        conditions = {v.T: 1000, v.P: 101325, v.X('TI'): 0.5}
        
        # Single phase
        print("\n\nSingle phase equilibrium (BCC_A2 only):")
        result_single = equilibrium(dbf_nbti, ['NB', 'TI'], ['BCC_A2'], conditions, gpu=False)
        print(f"  Y shape: {result_single.Y.shape}")
        print(f"  Internal DOF dimension: {list(result_single.coords['internal_dof'].values)}")
        
        # Two phases
        print("\n\nTwo phase equilibrium (BCC_A2 + HCP_A3):")
        result_multi = equilibrium(dbf_nbti, ['NB', 'TI'], ['BCC_A2', 'HCP_A3'], conditions, gpu=False)
        print(f"  Y shape: {result_multi.Y.shape}")
        print(f"  Internal DOF dimension: {list(result_multi.coords['internal_dof'].values)}")
        
        # Check calculate to understand phase DOF
        print("\n\nCalculate grid for phases:")
        for phase_name in ['BCC_A2', 'HCP_A3']:
            grid = calculate(dbf_nbti, ['NB', 'TI'], phase_name, 
                           output='GM', T=1000, P=101325, pdens=50)
            print(f"\n{phase_name} grid:")
            print(f"  Y shape: {grid.Y.shape}")
            if hasattr(grid.Y, 'values'):
                print(f"  Y values sample: {grid.Y.values[0,0,0,0,:]}")
            
    except FileNotFoundError:
        print("NbTi.tdb not found")

if __name__ == "__main__":
    main()