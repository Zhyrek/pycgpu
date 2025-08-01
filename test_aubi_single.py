from pycalphad import Database, equilibrium
import numpy as np
import os

# Enable GPU debug
os.environ['GPU_DEBUG'] = '1'
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load Au-Bi database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']
all_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'AU2BI_C15']

# Test a single condition with verbose output
conditions = {'T': 800, 'P': 101325, 'X(BI)': 0.2}

print("Testing Au-Bi at T=800K, X(BI)=0.2")
print("=" * 60)

# Try GPU
print("\nGPU Calculation (verbose):")
try:
    result_gpu = equilibrium(db, components, all_phases, conditions, 
                           calc_opts={'pdens': 500}, gpu=True, verbose=True)
    print("\nGPU Success!")
    for phase in np.unique(result_gpu.Phase.values):
        if phase != '':
            mask = result_gpu.Phase.values == phase
            amount = result_gpu.NP.values[mask][0]
            print(f"  {phase}: NP = {amount:.6f}")
except Exception as e:
    print(f"\nGPU Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Try CPU for comparison
print("\n\nCPU Calculation:")
try:
    result_cpu = equilibrium(db, components, all_phases, conditions, 
                           calc_opts={'pdens': 500})
    print("CPU Success!")
    for phase in np.unique(result_cpu.Phase.values):
        if phase != '':
            mask = result_cpu.Phase.values == phase
            amount = result_cpu.NP.values[mask][0]
            print(f"  {phase}: NP = {amount:.6f}")
except Exception as e:
    print(f"CPU Error: {type(e).__name__}: {e}")