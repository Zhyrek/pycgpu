from pycalphad import Database, equilibrium
import numpy as np
import os

# Enable verbose GPU output
os.environ['GPU_DEBUG'] = '1'
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load Au-Bi database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'RHOMBOHEDRAL_A7']

print("Testing Au-Bi LIQUID + RHOMBOHEDRAL_A7 at low temperatures")
print("Focus on conditions where two-phase equilibria should exist")
print("=" * 60)

# Test at 400K with different compositions
temps = [350, 400, 450]
x_bi_values = [0.8, 0.85, 0.9, 0.95]

for T in temps:
    print(f"\nTemperature: {T}K")
    print("-" * 60)
    
    for x_bi in x_bi_values:
        print(f"\n  X(BI) = {x_bi}")
        conditions = {'T': T, 'P': 101325, 'X(BI)': x_bi}
        
        # GPU
        try:
            result_gpu = equilibrium(db, components, phases, conditions, 
                                   calc_opts={'pdens': 500}, gpu=True, verbose=True)
            gpu_str = []
            for phase in np.unique(result_gpu.Phase.values):
                if phase != '':
                    mask = result_gpu.Phase.values == phase
                    amount = result_gpu.NP.values[mask][0]
                    if amount > 1e-10:
                        gpu_str.append(f"{phase}({amount:.3f})")
            print(f"    GPU: {' + '.join(gpu_str) if gpu_str else 'No phases'}")
        except Exception as e:
            print(f"    GPU Error: {e}")
        
        # CPU
        try:
            result_cpu = equilibrium(db, components, phases, conditions, 
                                   calc_opts={'pdens': 500})
            cpu_str = []
            for phase in np.unique(result_cpu.Phase.values):
                if phase != '':
                    mask = result_cpu.Phase.values == phase
                    amount = result_cpu.NP.values[mask][0]
                    if amount > 1e-10:
                        cpu_str.append(f"{phase}({amount:.3f})")
            print(f"    CPU: {' + '.join(cpu_str) if cpu_str else 'No phases'}")
        except Exception as e:
            print(f"    CPU Error: {e}")

# Test the specific case that failed in detail
print("\n" + "=" * 60)
print("DETAILED TEST: T=400K, X(BI)=0.9")
print("=" * 60)

conditions = {'T': 400, 'P': 101325, 'X(BI)': 0.9}

print("\nGPU Calculation (verbose):")
result_gpu = equilibrium(db, components, phases, conditions, 
                       calc_opts={'pdens': 100}, gpu=True, verbose=True)

print("\nGPU Final Results:")
for phase in np.unique(result_gpu.Phase.values):
    if phase != '':
        mask = result_gpu.Phase.values == phase
        amount = result_gpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")

print("\nCPU Results:")
result_cpu = equilibrium(db, components, phases, conditions, 
                       calc_opts={'pdens': 100})
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        print(f"  {phase}: NP = {amount:.6f}")