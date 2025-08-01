from pycalphad import Database, equilibrium
import numpy as np

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

# Check ALCU_ZETA structure
print("ALCU_ZETA phase structure:")
alcu_zeta = db.phases['ALCU_ZETA']
print(f"  Sublattices: {alcu_zeta.sublattices}")
print(f"  Constituents: {alcu_zeta.constituents}")
print(f"  Site ratios sum: {sum(float(s) for s in alcu_zeta.sublattices)}")

# Conditions where LIQUID and ALCU_ZETA might coexist
conditions = {
    'T': 900,
    'P': 101325,
    'X(AL)': 0.45,
    'X(CU)': 0.40
}

print("\n\nRunning CPU equilibrium calculation...")
# Patch solver to print matrix info
import pycalphad.core.minimizer as minimizer_module

# Track matrix creation
original_construct = minimizer_module.construct_equilibrium_system

def patched_construct(spec, state, num_reserved_rows):
    print(f"\n[CPU] construct_equilibrium_system at iteration {state.iteration}")
    print(f"  num_stable_phases: {state.free_stable_compset_indices.shape[0]}")
    print(f"  num_fixed_phases: {spec.fixed_stable_compset_indices.shape[0]}")
    print(f"  num_fixed_mole_fraction_conditions: {spec.prescribed_mole_fraction_rhs.shape[0]}")
    
    # Call original
    result = original_construct(spec, state, num_reserved_rows)
    
    # Get matrix and RHS
    equilibrium_matrix, equilibrium_rhs = result
    
    print(f"  Equilibrium matrix shape: {equilibrium_matrix.shape}")
    
    # Print phase information
    for idx, cs_idx in enumerate(state.free_stable_compset_indices):
        compset = state.compsets[cs_idx]
        print(f"  Free stable phase {idx}: {compset.phase_record.phase_name}, NP={compset.NP:.6f}")
    
    # Only print detailed matrix for first iteration
    if state.iteration == 0:
        print("\n[CPU] Equilibrium matrix (first 5x5):")
        for i in range(min(5, equilibrium_matrix.shape[0])):
            row_str = "  ["
            for j in range(min(5, equilibrium_matrix.shape[1])):
                row_str += f"{equilibrium_matrix[i,j]:+.3e} "
            row_str += "...]"
            print(row_str)
        print(f"  RHS (first 5): {equilibrium_rhs[:5]}")
    
    return result

minimizer_module.construct_equilibrium_system = patched_construct

try:
    # CPU calculation
    result_cpu = equilibrium(db, components, phases, conditions, 
                            calc_opts={'pdens': 1000},
                            verbose=True)
    
    cpu_phases = []
    for phase in np.unique(result_cpu.Phase.values):
        if phase != '':
            mask = result_cpu.Phase.values == phase
            amount = result_cpu.NP.values[mask][0]
            if amount > 1e-10:
                cpu_phases.append(phase)
    
    print(f"\nCPU final phases: {cpu_phases}")
    
except Exception as e:
    print(f"CPU Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Restore original
minimizer_module.construct_equilibrium_system = original_construct

print("\n" + "="*60)
print("\nRunning GPU equilibrium calculation...")

try:
    # GPU calculation
    result_gpu = equilibrium(db, components, phases, conditions, 
                           calc_opts={'pdens': 1000},
                           gpu=True)
    
    gpu_phases = []
    for phase in np.unique(result_gpu.Phase.values):
        if phase != '':
            mask = result_gpu.Phase.values == phase
            amount = result_gpu.NP.values[mask][0]
            if amount > 1e-10:
                gpu_phases.append(phase)
    
    print(f"\nGPU final phases: {gpu_phases}")
    
except Exception as e:
    print(f"GPU Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()