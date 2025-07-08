import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

print("CPU Reference Hessian Values (site fraction block):")
print("BCC_A2 at Y(NB)=0.773706, Y(TI)=0.226294:")
print("  H[2,2] = 1.088871887384e+06")
print("  H[2,3] = 1.304530000007e+04") 
print("  H[3,2] = 1.304530000007e+04")
print("  H[3,3] = 3.722885770281e+06")
print()
print("LIQUID at Y(NB)=0.666667, Y(TI)=0.333333:")
print("  H[2,2] = 1.263699436900e+06")
print("  H[2,3] = 7.406099999998e+03")
print("  H[3,2] = 7.406099999998e+03") 
print("  H[3,3] = 2.527402664903e+06")
print()
print("="*60)
print("Running GPU test - look for GPU Hessian debug output...")
print("="*60)

try:
    # Run with minimal points to get to Hessian calculation quickly
    eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 2}, gpu=True)
    print(f"\nGPU completed successfully!")
    print(f"GPU GM: {float(eq_gpu.GM.values[0]):.6f}")
except Exception as e:
    print(f"\nGPU failed: {e}")
    print("\nLook in the debug output above for:")
    print("  'GPU DEBUG: Hessian calculated for phase record 0' (BCC_A2)")  
    print("  'GPU DEBUG: Hessian calculated for phase record 1' (LIQUID)")
    print("  'Site fraction Hessian block:' values")
    print("  Compare these against the CPU reference values shown above")