#!/usr/bin/env python
"""
Debug multi-dimensional indexing issue.
"""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.workspace import Workspace
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

def main():
    dbf = Database("""
    ELEMENT AL FCC_A1 26.98 69.95 28.30 !
    ELEMENT NI FCC_A1 58.69 67.40 29.87 !
    ELEMENT VA VACUUM 0.00 0.00 0.00 !
    PHASE FCC_A1 %  2 1 1 !
    CONSTITUENT FCC_A1 : AL,NI : VA : !
    PARAMETER G(FCC_A1,AL:VA;0) 298.15 -7976.15+137.0715*T-24.36720*T*LN(T); 700 Y !
    PARAMETER G(FCC_A1,NI:VA;0) 298.15 -5179.159+117.8540*T-22.09600*T*LN(T); 1728 Y !
    """)
    
    wks = Workspace(database=dbf, components=['AL', 'NI', 'VA'], 
                    phases=['FCC_A1'], 
                    conditions={v.T: [800, 1000, 1200], v.P: 101325, v.X('NI'): 0.5})
    
    print("Multi-dimensional Indexing Debug")
    print("="*50)
    
    # Replicate the exact GPU data extraction
    from pycalphad import calculate
    
    unitless_conds = OrderedDict((key, wks.conditions[key]) for key in wks.conditions.keys())
    state_variables = wks.phase_record_factory.state_variables
    
    grid = calculate(wks.database, wks.components, wks.phases, 
                    output='GM', T=[800,1000,1200], P=101325, pdens=60)
    
    properties = starting_point(unitless_conds, state_variables, wks.phase_record_factory, grid)
    
    print(f"Properties.NP full array:")
    print(f"  Shape: {properties.NP.shape}")
    print(f"  Values: {properties.NP}")
    
    print(f"\nTesting indexing for each condition:")
    gm_shape = properties.GM.shape
    num_conditions = np.prod(gm_shape)
    
    for cond_idx in range(num_conditions):
        multi_idx = np.unravel_index(cond_idx, gm_shape)
        print(f"\nCondition {cond_idx}: multi_idx = {multi_idx}")
        
        # Test different indexing approaches
        try:
            direct_index = properties.NP[multi_idx]
            print(f"  Direct indexing: {direct_index}")
        except Exception as e:
            print(f"  Direct indexing FAILED: {e}")
        
        try:
            if len(multi_idx) > 0:
                conditional_index = properties.NP[multi_idx]
            else:
                conditional_index = properties.NP
            print(f"  Conditional indexing: {conditional_index}")
        except Exception as e:
            print(f"  Conditional indexing FAILED: {e}")
        
        try:
            asarray_index = np.asarray(properties.NP[multi_idx])
            print(f"  np.asarray indexing: {asarray_index}")
        except Exception as e:
            print(f"  np.asarray indexing FAILED: {e}")

if __name__ == "__main__":
    main()