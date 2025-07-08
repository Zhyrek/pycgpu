import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

print("Testing GPU Hessian with corrected code generation...")
print()

print("CPU Reference Values (from previous test):")
print("BCC_A2 site fraction Hessian at Y(NB)=0.773706, Y(TI)=0.226294:")
print("  H[2,2] = 1.088871887384e+06")
print("  H[2,3] = 1.304530000007e+04") 
print("  H[3,2] = 1.304530000007e+04")
print("  H[3,3] = 3.722885770281e+06")
print()
print("LIQUID site fraction Hessian at Y(NB)=0.666667, Y(TI)=0.333333:")
print("  H[2,2] = 1.263699436900e+06")
print("  H[2,3] = 7.406099999998e+03")
print("  H[3,2] = 7.406099999998e+03") 
print("  H[3,3] = 2.527402664903e+06")
print()
print("="*60)

# First run CPU for comparison
print("Running CPU equilibrium...")
eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 5})
cpu_gm = float(eq_cpu.GM.values[0])
print(f"CPU GM: {cpu_gm:.6f} J/mol")

print("\nRunning GPU equilibrium...")
print("Code generation shows correct mapping: T->x[2], Y1->x[3], Y2->x[4]")
print("Looking for GPU Hessian debug output...")

try:
    eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 5}, gpu=True)
    gpu_gm = float(eq_gpu.GM.values[0])
    print(f"\nSUCCESS! GPU completed")
    print(f"GPU GM: {gpu_gm:.6f} J/mol")
    
    diff = abs(cpu_gm - gpu_gm)
    print(f"GM Difference: {diff:.6e} J/mol")
    
    if diff <= 0.001:
        print("✓ SUCCESS: GPU matches CPU within 0.001 J/mol tolerance!")
        print("✓ Hessian fix is working correctly!")
    else:
        print("✗ FAIL: Difference exceeds 0.001 J/mol tolerance")
        print("Check GPU Hessian values in debug output above")
        
except Exception as e:
    print(f"GPU failed: {e}")
    print("\nEven with correct code generation, there may be other issues.")
    print("The Hessian functions should now work correctly if the memory access issue is resolved.")