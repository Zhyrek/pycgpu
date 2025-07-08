import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.utils import instantiate_models, get_state_variables
from pycalphad.core.phase_rec import PhaseRecord
import warnings
warnings.filterwarnings('ignore')

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}
T = conds[v.T]
P = conds[v.P]

# Instantiate models
models = instantiate_models(dbf, comps, phases)
state_variables = get_state_variables(models=models, conds=conds)

# Create phase record factory
phase_record_factory = PhaseRecordFactory(dbf, comps, state_variables, models)

# Build phase records for CPU testing
phase_records = {}
for phase_name in phases:
    phase_records[phase_name] = PhaseRecord(phase_record_factory, phase_name)

print("=== CPU Hessian Detailed Analysis ===")
print()

cpu_hessians = {}

# Test Hessian for each phase at specific compositions
for phase_name in phases:
    print(f"\nPhase: {phase_name}")
    mod = models[phase_name]
    pr = phase_records[phase_name]
    
    # Get phase info
    num_statevars = pr.num_statevars
    phase_dof = pr.phase_dof
    print(f"Number of state variables: {num_statevars}")
    print(f"Phase degrees of freedom: {phase_dof}")
    
    if phase_dof > 0:
        # Test at composition from equilibrium starting point
        if phase_name == 'BCC_A2':
            y = np.array([0.773706, 0.226294])  # From GPU starting point
        else:  # LIQUID
            y = np.array([0.666667, 0.333333])  # From GPU starting point
            
        print(f"Test composition: Y(NB)={y[0]:.6f}, Y(TI)={y[1]:.6f}")
        
        # Prepare dof array: [N, P, T, y1, y2, ...]
        dof = np.zeros(num_statevars + phase_dof)
        dof[0] = 1.0  # N
        dof[1] = P    # P  
        dof[2] = T    # T
        dof[num_statevars:] = y
        
        print(f"DOF array: {dof}")
        
        # Calculate energy
        energy = np.array([0.0])
        pr.obj(energy, dof)
        print(f"Energy: {energy[0]:.12e} J/mol")
        
        # Calculate gradient
        grad = np.zeros(num_statevars + phase_dof)
        pr.formulagrad(grad, dof)
        print(f"Gradient: {grad}")
        print(f"Site fraction gradient: {grad[num_statevars:]}")
        
        # Calculate Hessian
        hess_size = num_statevars + phase_dof
        hess = np.zeros((hess_size, hess_size))
        pr.formulahess(hess, dof)
        
        # Store for comparison
        cpu_hessians[phase_name] = hess.copy()
        
        print(f"Full Hessian matrix ({hess_size}x{hess_size}):")
        for i in range(hess_size):
            row_str = f"  [{i}] "
            for j in range(hess_size):
                row_str += f"{hess[i,j]:12.6e} "
            print(row_str)
        
        # Extract site fraction block
        sf_hess = hess[num_statevars:, num_statevars:]
        print(f"\nSite fraction Hessian block:")
        for i, row in enumerate(sf_hess):
            print(f"  [{i}] {row}")
        
        # Check specific values that should match GPU
        print(f"\nKey Hessian elements:")
        print(f"  H[{num_statevars},{num_statevars}] = {hess[num_statevars, num_statevars]:.12e}")
        print(f"  H[{num_statevars},{num_statevars+1}] = {hess[num_statevars, num_statevars+1]:.12e}")
        print(f"  H[{num_statevars+1},{num_statevars}] = {hess[num_statevars+1, num_statevars]:.12e}")
        print(f"  H[{num_statevars+1},{num_statevars+1}] = {hess[num_statevars+1, num_statevars+1]:.12e}")

print("\n" + "="*60)
print("CPU Reference Values Summary:")
print("="*60)
for phase_name in phases:
    if phase_name in cpu_hessians:
        hess = cpu_hessians[phase_name]
        pr = phase_records[phase_name]
        num_statevars = pr.num_statevars
        print(f"\n{phase_name}:")
        print(f"  Site fraction Hessian block elements:")
        print(f"    H[{num_statevars},{num_statevars}] = {hess[num_statevars, num_statevars]:.12e}")
        print(f"    H[{num_statevars},{num_statevars+1}] = {hess[num_statevars, num_statevars+1]:.12e}")
        print(f"    H[{num_statevars+1},{num_statevars}] = {hess[num_statevars+1, num_statevars]:.12e}")
        print(f"    H[{num_statevars+1},{num_statevars+1}] = {hess[num_statevars+1, num_statevars+1]:.12e}")

print("\n" + "="*60)
print("Now test GPU to compare against these values...")
print("="*60)

# Try to run GPU test
try:
    print("\nRunning GPU equilibrium (first few iterations only)...")
    eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 5}, gpu=True)
    print(f"GPU completed successfully!")
    print(f"GPU GM: {float(eq_gpu.GM.values[0]):.12e}")
    
    # Compare with CPU
    eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 5})
    print(f"CPU GM: {float(eq_cpu.GM.values[0]):.12e}")
    
    diff = abs(float(eq_cpu.GM.values[0]) - float(eq_gpu.GM.values[0]))
    print(f"GM Difference: {diff:.12e} J/mol")
    
    if diff > 0.001:
        print("ERROR: Difference exceeds 0.001 J/mol threshold!")
    else:
        print("SUCCESS: Difference is within 0.001 J/mol threshold!")
        
except Exception as e:
    print(f"GPU test failed: {e}")
    print("\nTo manually check GPU Hessian values, look for these patterns in GPU debug output:")
    print("  'GPU DEBUG: Hessian calculated for phase record X'")
    print("  'Site fraction Hessian block:'")
    print("  Compare the printed values against the CPU reference values above")