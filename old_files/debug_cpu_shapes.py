#!/usr/bin/env python
"""
Debug script to understand the expected shapes from CPU equilibrium calculation.
"""

import numpy as np
from pycalphad import Database, equilibrium, calculate, variables as v
from pycalphad.core.workspace import Workspace
import warnings
warnings.filterwarnings('ignore')

def main():
    # Create simple Al-Ni database
    dbf = Database("""
    ELEMENT AL FCC_A1 26.98 69.95 28.30 !
    ELEMENT NI FCC_A1 58.69 67.40 29.87 !
    ELEMENT VA VACUUM 0.00 0.00 0.00 !
    
    PHASE FCC_A1 %  2 1 1 !
    CONSTITUENT FCC_A1 : AL,NI : VA : !
    
    PARAMETER G(FCC_A1,AL:VA;0) 298.15 -7976.15+137.0715*T-24.36720*T*LN(T)
        -0.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 700 Y
        -11276.24+223.0481*T-38.58443*T*LN(T)+0.018531982*T**2
        -5.764227E-06*T**3+74092*T**(-1); 933.6 Y
        -11277.68+188.6620*T-31.74819*T*LN(T)-1230.622E25*T**(-9); 2900 N !
    
    PARAMETER G(FCC_A1,NI:VA;0) 298.15 -5179.159+117.8540*T-22.09600*T*LN(T)
        -0.0048407*T**2; 1728 Y
        -27840.62+279.1350*T-43.10*T*LN(T)+1127.54E28*T**(-9); 3000 N !
    
    PARAMETER G(FCC_A1,AL,NI:VA;0) 298.15 -162407.75+16.212965*T; 6000 N !
    PARAMETER G(FCC_A1,AL,NI:VA;1) 298.15 +73417.798-34.914168*T; 6000 N !
    PARAMETER G(FCC_A1,AL,NI:VA;2) 298.15 +33471.014-9.8373558*T; 6000 N !
    """)
    
    # Test simple single point calculation
    conditions = {v.T: 1000, v.P: 101325, v.X('NI'): 0.5}
    
    print("="*70)
    print("CPU Equilibrium Shape Analysis")
    print("="*70)
    
    # Create workspace to understand structure
    wks = Workspace(database=dbf, components=['AL', 'NI', 'VA'], 
                    phases=['FCC_A1'], conditions=conditions)
    
    print("\nWorkspace info:")
    print(f"  Components: {wks.components}")
    print(f"  Number of components: {len(wks.components)}")
    print(f"  Phases: {wks.phases}")
    print(f"  Number of phases: {len(wks.phases)}")
    
    # Run calculate to understand grid structure
    print("\nCalculate() grid structure:")
    grid = calculate(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], 
                     output='GM', T=1000, P=101325, pdens=100)
    
    print(f"  Grid GM shape: {grid.GM.shape}")
    print(f"  Grid Phase shape: {grid.Phase.shape}")
    print(f"  Grid X shape: {grid.X.shape}")
    print(f"  Grid Y shape: {grid.Y.shape}")
    print(f"  Grid coords: {list(grid.coords.keys())}")
    
    # CPU equilibrium calculation
    print("\nRunning CPU equilibrium calculation...")
    cpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=False)
    
    print("\nCPU Equilibrium result structure:")
    print(f"  Components in result: {list(cpu_result.coords['component'].values)}")
    print(f"  Number of components: {len(cpu_result.coords['component'].values)}")
    print(f"  Vertex dimension: {list(cpu_result.coords['vertex'].values)}")
    print(f"  Number of vertices: {len(cpu_result.coords['vertex'].values)}")
    print(f"  Internal DOF dimension: {list(cpu_result.coords['internal_dof'].values)}")
    
    print("\nShape analysis:")
    attrs = ['GM', 'MU', 'NP', 'Phase', 'X', 'Y']
    for attr in attrs:
        if hasattr(cpu_result, attr):
            data = getattr(cpu_result, attr)
            print(f"  {attr}: shape = {data.shape}, dims = {data.dims if hasattr(data, 'dims') else 'N/A'}")
    
    # Check with multiple phases
    print("\n" + "="*70)
    print("Multi-phase system analysis (if NbTi.tdb exists)")
    print("="*70)
    
    try:
        dbf_nbti = Database("NbTi.tdb")
        
        # Create workspace for multi-phase
        wks_multi = Workspace(database=dbf_nbti, components=['NB', 'TI'], 
                             phases=['BCC_A2', 'HCP_A3'], 
                             conditions={v.T: 1000, v.P: 101325, v.X('TI'): 0.5})
        
        print(f"\nMulti-phase workspace:")
        print(f"  Components: {wks_multi.components}")
        print(f"  Phases: {wks_multi.phases}")
        
        # Run equilibrium
        multi_result = equilibrium(dbf_nbti, ['NB', 'TI'], ['BCC_A2', 'HCP_A3'],
                                  {v.T: 1000, v.P: 101325, v.X('TI'): 0.5}, gpu=False)
        
        print(f"\nMulti-phase equilibrium result:")
        print(f"  Vertex dimension: {list(multi_result.coords['vertex'].values)}")
        print(f"  Number of vertices: {len(multi_result.coords['vertex'].values)}")
        
        for attr in attrs:
            if hasattr(multi_result, attr):
                data = getattr(multi_result, attr)
                print(f"  {attr}: shape = {data.shape}")
                
    except FileNotFoundError:
        print("  NbTi.tdb not found - skipping multi-phase analysis")
    
    print("\n" + "="*70)
    print("Key findings:")
    print("="*70)
    print("1. The vertex dimension in CPU results corresponds to the maximum")
    print("   number of composition sets that can exist in equilibrium")
    print("2. For single-phase systems, this is typically phase_count + 2")
    print("3. The actual number of active phases is indicated by NP > 0")
    print("4. Empty phases have '' in the Phase array and 0 in NP array")

if __name__ == "__main__":
    main()