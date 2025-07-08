#!/usr/bin/env python
"""
Debug script to understand shape mismatches between CPU and GPU equilibrium results.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

def analyze_results(result, label):
    """Analyze the shape and content of equilibrium results."""
    print(f"\n{label} Results Analysis:")
    print("-" * 50)
    
    if result is None:
        print("Result is None!")
        return
        
    # Check all available attributes
    attrs = ['GM', 'MU', 'NP', 'Phase', 'X', 'Y']
    
    for attr in attrs:
        if hasattr(result, attr):
            data = getattr(result, attr)
            if hasattr(data, 'values'):
                array = data.values
            else:
                array = np.array(data)
                
            print(f"{attr}:")
            print(f"  Shape: {array.shape}")
            print(f"  Dtype: {array.dtype}")
            
            if attr == 'Phase':
                # Show unique phase names
                unique_phases = np.unique(array.flatten())
                print(f"  Unique phases: {unique_phases}")
            elif attr == 'NP':
                # Show non-zero phase amounts
                flat_np = array.flatten()
                nonzero_mask = flat_np > 1e-10
                nonzero_indices = np.where(nonzero_mask)[0]
                print(f"  Non-zero phases: {len(nonzero_indices)}")
                if len(nonzero_indices) > 0:
                    print(f"  Non-zero amounts: {flat_np[nonzero_mask]}")
            elif attr in ['GM', 'MU']:
                # Show first few values
                flat_data = array.flatten()
                print(f"  First 3 values: {flat_data[:3]}")
                
    # Check coordinates
    if hasattr(result, 'coords'):
        print("\nCoordinates:")
        for coord_name, coord_data in result.coords.items():
            if hasattr(coord_data, 'values'):
                coord_array = coord_data.values
            else:
                coord_array = np.array(coord_data)
            print(f"  {coord_name}: shape={coord_array.shape}, values={coord_array.flatten()[:5]}...")

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
    print("Shape Mismatch Debug - Single Point Calculation")
    print("="*70)
    
    # CPU calculation
    print("\nRunning CPU calculation...")
    cpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=False, verbose=False)
    analyze_results(cpu_result, "CPU")
    
    # GPU calculation
    print("\nRunning GPU calculation...")
    gpu_result = equilibrium(dbf, ['AL', 'NI', 'VA'], ['FCC_A1'], conditions, gpu=True, verbose=True)
    analyze_results(gpu_result, "GPU")
    
    # Compare dimensions
    print("\n" + "="*70)
    print("Dimension Comparison Summary:")
    print("="*70)
    
    attrs_to_compare = ['GM', 'MU', 'NP', 'Phase', 'X', 'Y']
    for attr in attrs_to_compare:
        cpu_shape = None
        gpu_shape = None
        
        if hasattr(cpu_result, attr):
            cpu_data = getattr(cpu_result, attr)
            cpu_shape = cpu_data.values.shape if hasattr(cpu_data, 'values') else np.array(cpu_data).shape
            
        if hasattr(gpu_result, attr):
            gpu_data = getattr(gpu_result, attr)
            gpu_shape = gpu_data.values.shape if hasattr(gpu_data, 'values') else np.array(gpu_data).shape
            
        if cpu_shape != gpu_shape:
            print(f"{attr}: CPU {cpu_shape} vs GPU {gpu_shape} - MISMATCH!")
        else:
            print(f"{attr}: {cpu_shape} - OK")

if __name__ == "__main__":
    main()