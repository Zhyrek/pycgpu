from pycalphad import Database, equilibrium
import numpy as np

# Load database and check phase structures
db = Database('Al-Cu-Fe.tdb')

# Check some complex phases
complex_phases = ['ALCU_ZETA', 'TS01T3', 'GAMMA_D83', 'BCC_B2']

print("Phase sublattice structures:")
for phase_name in complex_phases:
    if phase_name in db.phases:
        phase = db.phases[phase_name]
        sublattices = phase.sublattices
        site_ratios = [float(subl) for subl in sublattices]
        constituents = phase.constituents
        print(f"\n{phase_name}:")
        print(f"  Site ratios: {site_ratios} (sum: {sum(site_ratios)})")
        print(f"  Number of sublattices: {len(sublattices)}")
        for i, subl in enumerate(sublattices):
            print(f"  Sublattice {i}: {constituents[i]} (site_ratio: {subl})")

# Test a simple case first
print("\n\nTesting GPU equilibrium with Al-Cu-Fe...")
components = ['AL', 'CU', 'FE', 'VA']
phases = ['BCC_A2', 'FCC_A1']  # Start with simple phases

conditions = {
    'T': 1000,
    'P': 101325,
    'X(AL)': 0.3,
    'X(CU)': 0.3
}

try:
    # CPU calculation
    print("\nCPU calculation...")
    result_cpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500})
    cpu_phases = list(set(result_cpu.Phase.values.flatten()) - {''})
    print(f"CPU phases: {cpu_phases}")
    
    # GPU calculation
    print("\nGPU calculation...")
    result_gpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 500}, gpu=True)
    gpu_phases = list(set(result_gpu.Phase.values.flatten()) - {''})
    print(f"GPU phases: {gpu_phases}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Now try with a complex phase
print("\n\nTesting with complex phase ALCU_ZETA...")
phases_complex = ['ALCU_ZETA', 'FCC_A1']

try:
    # Adjust conditions to be in ALCU_ZETA region
    conditions_complex = {
        'T': 800,
        'P': 101325,
        'X(AL)': 0.45,
        'X(CU)': 0.45
    }
    
    # CPU calculation
    print("\nCPU calculation with ALCU_ZETA...")
    result_cpu = equilibrium(db, components, phases_complex, conditions_complex, calc_opts={'pdens': 500})
    cpu_phases = list(set(result_cpu.Phase.values.flatten()) - {''})
    print(f"CPU phases: {cpu_phases}")
    
    # GPU calculation
    print("\nGPU calculation with ALCU_ZETA...")
    result_gpu = equilibrium(db, components, phases_complex, conditions_complex, calc_opts={'pdens': 500}, gpu=True)
    gpu_phases = list(set(result_gpu.Phase.values.flatten()) - {''})
    print(f"GPU phases: {gpu_phases}")
    
except Exception as e:
    print(f"Error with complex phase: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()