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

print("=== CPU Hessian Analysis ===")
print()

# Test Hessian for each phase
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
        # Test at a specific composition
        y = np.array([0.3, 0.7])  # Y(NB)=0.3, Y(TI)=0.7
        print(f"\nTest point: Y(NB)={y[0]:.2f}, Y(TI)={y[1]:.2f}")
        
        # Prepare dof array: [T, P, y1, y2, ...]
        dof = np.zeros(num_statevars + phase_dof)
        dof[0] = T
        dof[1] = P
        dof[num_statevars:] = y
        
        # Calculate energy
        energy = np.array([0.0])
        pr.obj(energy, dof)
        print(f"Energy: {energy[0]:.6f} J/mol")
        
        # Calculate gradient
        grad = np.zeros(num_statevars + phase_dof)
        pr.formulagrad(grad, dof)
        print(f"Gradient (site fraction part): {grad[num_statevars:]}")
        
        # Calculate Hessian
        hess_size = num_statevars + phase_dof
        hess = np.zeros((hess_size, hess_size))
        pr.formulahess(hess, dof)
        
        # Extract site fraction block
        sf_hess = hess[num_statevars:, num_statevars:]
        print(f"Site fraction Hessian block:")
        for i, row in enumerate(sf_hess):
            print(f"  [{i}] {row}")

print("\n\n=== Running GPU Equilibrium ===")
# Run GPU equilibrium to generate kernel
eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 10}, gpu=True)
print(f"GPU GM: {float(eq_gpu.GM.values[0]):.6f}")

# Also run CPU for comparison
print("\n=== Running CPU Equilibrium ===")
eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 10})
print(f"CPU GM: {float(eq_cpu.GM.values[0]):.6f}")

print(f"\nGM Difference: {abs(eq_cpu.GM.values[0] - eq_gpu.GM.values[0]):.6e} J/mol")
if abs(eq_cpu.GM.values[0] - eq_gpu.GM.values[0]) > 0.001:
    print("WARNING: Difference exceeds 0.001 J/mol threshold!")