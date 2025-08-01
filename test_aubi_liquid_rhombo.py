from pycalphad import Database, equilibrium
import numpy as np
import os

# Clear cache
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load Au-Bi database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']

# Test with just LIQUID and RHOMBOHEDRAL_A7
phases = ['LIQUID', 'RHOMBOHEDRAL_A7']

print("Testing Au-Bi with LIQUID + RHOMBOHEDRAL_A7 phases only")
print("=" * 60)

# Test different conditions
test_conditions = [
    {'T': 400, 'P': 101325, 'X(BI)': 0.9, 'name': 'Low T, Bi-rich (400K, X(BI)=0.9)'},
    {'T': 500, 'P': 101325, 'X(BI)': 0.7, 'name': 'Medium T, Bi-rich (500K, X(BI)=0.7)'},
    {'T': 544, 'P': 101325, 'X(BI)': 0.5, 'name': 'Near melting point (544K, X(BI)=0.5)'},
    {'T': 600, 'P': 101325, 'X(BI)': 0.5, 'name': 'Above melting (600K, X(BI)=0.5)'},
    {'T': 800, 'P': 101325, 'X(BI)': 0.3, 'name': 'High T, Au-rich (800K, X(BI)=0.3)'},
    {'T': 1000, 'P': 101325, 'X(BI)': 0.5, 'name': 'Very high T (1000K, X(BI)=0.5)'},
]

for test in test_conditions:
    print(f"\n{test['name']}")
    print("-" * 60)
    
    conditions = {k: v for k, v in test.items() if k != 'name'}
    
    # GPU calculation
    print("GPU Results:")
    try:
        result_gpu = equilibrium(db, components, phases, conditions, 
                               calc_opts={'pdens': 1000}, gpu=True)
        
        gpu_phases = []
        for phase in np.unique(result_gpu.Phase.values):
            if phase != '':
                mask = result_gpu.Phase.values == phase
                amount = result_gpu.NP.values[mask][0]
                if amount > 1e-10:
                    gpu_phases.append((phase, amount))
                    print(f"  {phase}: NP = {amount:.6f}")
                    
                    # Get composition
                    try:
                        idx = np.where(mask)[0][0]
                        x_bi = result_gpu['X_BI'].values.flat[idx]
                        print(f"    X(BI) = {x_bi:.6f}")
                    except:
                        pass
        
        if not gpu_phases:
            print("  No stable phases found")
            
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # CPU calculation
    print("\nCPU Results:")
    try:
        result_cpu = equilibrium(db, components, phases, conditions, 
                               calc_opts={'pdens': 1000})
        
        cpu_phases = []
        for phase in np.unique(result_cpu.Phase.values):
            if phase != '':
                mask = result_cpu.Phase.values == phase
                amount = result_cpu.NP.values[mask][0]
                if amount > 1e-10:
                    cpu_phases.append((phase, amount))
                    print(f"  {phase}: NP = {amount:.6f}")
                    
                    # Get composition
                    try:
                        idx = np.where(mask)[0][0]
                        x_bi = result_cpu['X_BI'].values.flat[idx]
                        print(f"    X(BI) = {x_bi:.6f}")
                    except:
                        pass
        
        if not cpu_phases:
            print("  No stable phases found")
            
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Compare
    print("\nComparison:")
    gpu_phase_names = {p[0] for p in gpu_phases}
    cpu_phase_names = {p[0] for p in cpu_phases}
    
    if gpu_phase_names == cpu_phase_names:
        print("  ✓ Same phases found")
        if gpu_phases and cpu_phases:
            # Check amounts
            max_diff = 0
            for phase, gpu_amount in gpu_phases:
                cpu_amount = next((amt for p, amt in cpu_phases if p == phase), 0)
                diff = abs(gpu_amount - cpu_amount)
                max_diff = max(max_diff, diff)
            
            if max_diff < 1e-4:
                print(f"  ✓ Phase amounts match (max diff: {max_diff:.2e})")
            else:
                print(f"  ✗ Phase amounts differ (max diff: {max_diff:.2e})")
                for phase, gpu_amount in gpu_phases:
                    cpu_amount = next((amt for p, amt in cpu_phases if p == phase), 0)
                    print(f"    {phase}: GPU={gpu_amount:.6f}, CPU={cpu_amount:.6f}, diff={abs(gpu_amount-cpu_amount):.6f}")
    else:
        print("  ✗ Different phases found")
        print(f"    GPU: {gpu_phase_names}")
        print(f"    CPU: {cpu_phase_names}")

print("\n" + "=" * 60)
print("Summary: LIQUID + RHOMBOHEDRAL_A7 test completed")