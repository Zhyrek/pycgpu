#!/usr/bin/env python
"""
Debug starting_point data structure for multi-condition cases.
"""

import numpy as np
from pycalphad import Database, calculate, variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.workspace import Workspace
from collections import OrderedDict
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
    
    print("Starting Point Data Structure Debug")
    print("="*50)
    
    # Create workspace
    wks = Workspace(database=dbf, components=['AL', 'NI', 'VA'], 
                    phases=['FCC_A1'], 
                    conditions={v.T: [800, 1000, 1200], v.P: 101325, v.X('NI'): 0.5})
    
    # Replicate GPU data preparation process
    unitless_conds = OrderedDict((key, wks.conditions[key]) for key in wks.conditions.keys())
    str_conds = OrderedDict((str(key), value) for key, value in unitless_conds.items())
    local_conds = {key: value for key, value in wks.conditions.items()
                   if getattr(key, 'phase_name', None) is not None}
    state_variables = wks.phase_record_factory.state_variables
    
    # Set up grid options
    grid_opts = wks.calc_opts.copy() if wks.calc_opts else {}
    statevar_strings = [str(x) for x in state_variables]
    grid_opts.update({key: value for key, value in str_conds.items() if key in statevar_strings})
    
    if 'pdens' not in grid_opts:
        grid_opts['pdens'] = 60
    
    # Call calculate()
    grid = calculate(wks.database, wks.components, wks.phases, 
                    model=wks.models.unwrap() if hasattr(wks.models, 'unwrap') else wks.models,
                    fake_points=True, phase_records=wks.phase_record_factory, 
                    output='GM', parameters=wks.parameters.unwrap() if hasattr(wks.parameters, 'unwrap') else wks.parameters,
                    to_xarray=False, conditions=local_conds, **grid_opts)
    
    print(f"\nGrid structure:")
    print(f"  GM shape: {grid.GM.shape}")
    print(f"  Phase shape: {grid.Phase.shape}")
    
    # Call starting_point()
    properties = starting_point(unitless_conds, state_variables, wks.phase_record_factory, grid)
    
    print(f"\nStarting point properties:")
    print(f"  GM shape: {properties.GM.shape}")
    print(f"  GM values: {np.array(properties.GM)}")
    print(f"  MU shape: {properties.MU.shape}")
    print(f"  MU values: {np.array(properties.MU)}")
    print(f"  NP shape: {properties.NP.shape}")
    print(f"  NP values: {np.array(properties.NP)}")
    print(f"  Phase shape: {properties.Phase.shape}")
    print(f"  Phase values: {np.array(properties.Phase)}")
    
    # Test multi-dimensional indexing
    print(f"\nMulti-dimensional indexing test:")
    gm_shape = properties.GM.shape
    print(f"  GM shape: {gm_shape}")
    
    total_conditions = np.prod(gm_shape)
    print(f"  Total conditions: {total_conditions}")
    
    for cond_idx in range(total_conditions):
        multi_idx = np.unravel_index(cond_idx, gm_shape)
        print(f"\n  Condition {cond_idx}: multi_idx = {multi_idx}")
        
        gm_val = properties.GM[multi_idx] if len(multi_idx) > 0 else properties.GM
        mu_val = properties.MU[multi_idx] if len(multi_idx) > 0 else properties.MU
        np_val = properties.NP[multi_idx] if len(multi_idx) > 0 else properties.NP
        
        print(f"    GM: {gm_val}")
        print(f"    MU: {mu_val}")
        print(f"    NP: {np_val}")

if __name__ == "__main__":
    main()