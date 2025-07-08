import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Minimal test to isolate exactly where GPU fails
print("Testing minimal GPU execution to isolate failure point...")

dbf = Database('NbTi.tdb') 
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']  # Only one phase to minimize complexity
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

print("1. CPU result:")
eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 5})
cpu_gm = float(eq_cpu.GM.values[0])
print(f"   CPU GM: {cpu_gm:.6f} J/mol")

print("\n2. GPU test with single phase:")
try:
    eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 5}, gpu=True)
    gpu_gm = float(eq_gpu.GM.values[0])
    print(f"   SUCCESS! GPU GM: {gpu_gm:.6f} J/mol")
    
    diff = abs(cpu_gm - gpu_gm)
    print(f"   Difference: {diff:.6e} J/mol")
    
    if diff <= 0.001:
        print("   ✓ GPU matches CPU within tolerance!")
    else:
        print("   ✗ Difference exceeds tolerance")
        
except Exception as e:
    print(f"   GPU failed: {e}")
    print("   Location: After struct corruption fix, during actual solver execution")