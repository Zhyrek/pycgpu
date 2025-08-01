from pycalphad import Database, equilibrium
import numpy as np
import os

# Clear cache before testing
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load Au-Bi database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']

# Test different compositions and temperatures
test_conditions = [
    # Test 1: Low temperature, Bi-rich
    {'T': 400, 'P': 101325, 'X(BI)': 0.8, 'name': 'Bi-rich at 400K'},
    
    # Test 2: Intermediate temperature, equal composition
    {'T': 600, 'P': 101325, 'X(BI)': 0.5, 'name': 'Equal composition at 600K'},
    
    # Test 3: High temperature, Au-rich
    {'T': 800, 'P': 101325, 'X(BI)': 0.2, 'name': 'Au-rich at 800K'},
    
    # Test 4: Near eutectic composition
    {'T': 500, 'P': 101325, 'X(BI)': 0.65, 'name': 'Near eutectic at 500K'},
    
    # Test 5: Very high temperature (should be liquid)
    {'T': 1200, 'P': 101325, 'X(BI)': 0.4, 'name': 'High temp liquid at 1200K'},
]

# All phases in the system
all_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'AU2BI_C15']

print("=" * 80)
print("Au-Bi Binary System: GPU vs CPU Comparison")
print("=" * 80)

for i, test in enumerate(test_conditions):
    print(f"\nTest {i+1}: {test['name']}")
    print("-" * 60)
    
    conditions = {k: v for k, v in test.items() if k != 'name'}
    
    # GPU calculation
    try:
        result_gpu = equilibrium(db, components, all_phases, conditions, 
                               calc_opts={'pdens': 2000}, gpu=True)
        
        print("GPU Results:")
        gpu_phases = []
        for phase in np.unique(result_gpu.Phase.values):
            if phase != '':
                mask = result_gpu.Phase.values == phase
                amount = result_gpu.NP.values[mask][0]
                if amount > 1e-10:  # Only show phases with significant amount
                    gpu_phases.append((phase, amount))
                    print(f"  {phase}: NP = {amount:.6f}")
                    
                    # Get composition if binary
                    try:
                        # Find the first True index in mask
                        idx = np.where(mask)[0][0]
                        x_bi = result_gpu['X_BI'].values.flat[idx]
                        print(f"    X(BI) = {x_bi:.6f}")
                    except:
                        pass  # Skip composition if extraction fails
        
        if not gpu_phases:
            print("  No stable phases found")
            
    except Exception as e:
        print(f"GPU Error: {type(e).__name__}: {e}")
        gpu_phases = []
    
    # CPU calculation
    try:
        result_cpu = equilibrium(db, components, all_phases, conditions, 
                               calc_opts={'pdens': 2000})
        
        print("\nCPU Results:")
        cpu_phases = []
        for phase in np.unique(result_cpu.Phase.values):
            if phase != '':
                mask = result_cpu.Phase.values == phase
                amount = result_cpu.NP.values[mask][0]
                if amount > 1e-10:  # Only show phases with significant amount
                    cpu_phases.append((phase, amount))
                    print(f"  {phase}: NP = {amount:.6f}")
                    
                    # Get composition if binary
                    try:
                        # Find the first True index in mask
                        idx = np.where(mask)[0][0]
                        x_bi = result_cpu['X_BI'].values.flat[idx]
                        print(f"    X(BI) = {x_bi:.6f}")
                    except:
                        pass  # Skip composition if extraction fails
        
        if not cpu_phases:
            print("  No stable phases found")
            
    except Exception as e:
        print(f"CPU Error: {type(e).__name__}: {e}")
        cpu_phases = []
    
    # Compare results
    print("\nComparison:")
    gpu_phase_names = {p[0] for p in gpu_phases}
    cpu_phase_names = {p[0] for p in cpu_phases}
    
    if gpu_phase_names == cpu_phase_names:
        print("  ✓ Same phases found")
        # Check phase amounts
        max_diff = 0
        for phase, gpu_amount in gpu_phases:
            cpu_amount = next((amt for p, amt in cpu_phases if p == phase), 0)
            diff = abs(gpu_amount - cpu_amount)
            max_diff = max(max_diff, diff)
        
        if max_diff < 1e-4:
            print("  ✓ Phase amounts match (max diff: {:.2e})".format(max_diff))
        else:
            print("  ✗ Phase amounts differ (max diff: {:.2e})".format(max_diff))
    else:
        print("  ✗ Different phases found")
        print(f"    GPU: {gpu_phase_names}")
        print(f"    CPU: {cpu_phase_names}")

print("\n" + "=" * 80)
print("Summary: Au-Bi binary system test completed")
print("=" * 80)