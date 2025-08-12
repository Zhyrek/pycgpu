#!/usr/bin/env python
"""Test current Al-Cu-Fe ternary system status after gradient fixes."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

def test_current_status():
    print('TESTING CURRENT AL-CU-FE TERNARY STATUS')
    print('='*50)

    try:
        tdb = Database('Al-Cu-Fe.tdb')
        
        # Test ternary condition
        conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
        
        # CPU calculation
        print('CPU calculation...')
        eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False)
        cpu_gm = float(eq_cpu.GM.values[0])
        print(f'CPU GM: {cpu_gm:.6f} J/mol')
        
        # GPU calculation  
        print('GPU calculation...')
        eq_gpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=True)
        gpu_gm = float(eq_gpu.GM.values[0])
        print(f'GPU GM: {gpu_gm:.6f} J/mol')
        
        # Compare
        diff = abs(gpu_gm - cpu_gm)
        print(f'Absolute difference: {diff:.6f} J/mol')
        
        if diff < 1e-6:
            print('SUCCESS: GPU matches CPU to numerical precision!')
        elif diff < 1.0:
            print('GOOD: GPU matches CPU within 1 J/mol')
        elif diff < 10.0:
            print('OK: GPU matches CPU within 10 J/mol')  
        else:
            print('PROBLEM: GPU/CPU difference is significant')
            
        return diff
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_current_status()