import numpy as np
import cupy as cp
from pycalphad import Database, calculate, equilibrium, variables as v
from pycalphad.core.utils import instantiate_models
from pycalphad.core.starting_point import starting_point
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.gpu.gpu_equilibrium import equilibrium_gpu
import symengine as se
from collections import OrderedDict

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}
T = conds[v.T]
P = conds[v.P]

# Instantiate models
models = instantiate_models(dbf, comps, phases)
phase_records = {}

# Build phase records for CPU testing
for phase_name in phases:
    mod = models[phase_name]
    phase_records[phase_name] = PhaseRecord(comps, mod.state_variables, mod.components, phase_name)

print("=== CPU Hessian Analysis ===")
print()

# Test Hessian for each phase
for phase_name in phases:
    print(f"\nPhase: {phase_name}")
    mod = models[phase_name]
    pr = phase_records[phase_name]
    
    # Get phase info
    num_statevars = len(mod.state_variables)
    phase_dof = mod.phase_dof
    print(f"Number of state variables: {num_statevars}")
    print(f"Phase degrees of freedom: {phase_dof}")
    print(f"Components: {mod.components}")
    
    # Create test points in composition space
    if phase_dof > 0:
        # Test at different compositions
        test_points = [
            np.array([0.3, 0.7]),  # Y(BCC_A2,NB)=0.3, Y(BCC_A2,TI)=0.7
            np.array([0.5, 0.5]),  # Equal composition
            np.array([0.1, 0.9]),  # Low NB
            np.array([0.9, 0.1]),  # High NB
        ]
        
        for i, y in enumerate(test_points):
            print(f"\n  Test point {i+1}: Y(NB)={y[0]:.2f}, Y(TI)={y[1]:.2f}")
            
            # Prepare dof array: [T, P, y1, y2, ...]
            dof = np.zeros(num_statevars + phase_dof)
            dof[0] = T
            dof[1] = P
            dof[num_statevars:] = y
            
            # Calculate Hessian
            hess_size = num_statevars + phase_dof
            hess = np.zeros((hess_size, hess_size))
            pr.obj.formulahess(hess, dof)
            
            # Extract site fraction block
            sf_hess = hess[num_statevars:, num_statevars:]
            print(f"  Site fraction Hessian block:")
            for row in sf_hess:
                print(f"    {row}")
            
            # Check symmetry
            if phase_dof > 1:
                sym_error = np.abs(sf_hess - sf_hess.T).max()
                print(f"  Symmetry error: {sym_error:.2e}")
            
            # Calculate eigenvalues to check positive definiteness
            if phase_dof > 0:
                try:
                    eigvals = np.linalg.eigvalsh(sf_hess)
                    print(f"  Eigenvalues: {eigvals}")
                except:
                    print("  Could not compute eigenvalues")

print("\n=== GPU Hessian Test ===")
print()

# Run GPU equilibrium to get the generated kernel
print("Running GPU equilibrium to generate kernel...")
eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 100}, gpu=True)

# Read the generated kernel to see the Hessian functions
print("\nChecking generated Hessian functions...")
with open('generated_equilibrium_kernel.cu', 'r') as f:
    kernel_code = f.read()
    
# Count Hessian functions
hess_count = kernel_code.count('__device__ void phase_hess_')
print(f"Number of Hessian functions generated: {hess_count}")

# Extract and display Hessian function signatures
import re
hess_funcs = re.findall(r'__device__ void (phase_hess_\d+)\(double\* hess, const double\* dof\)', kernel_code)
print(f"Hessian functions found: {hess_funcs}")

print("\n=== Detailed Hessian Comparison ===")

# Now let's create a more detailed comparison by adding debug output to the GPU code
# We'll modify the GPU minimizer to print Hessian values

# Create a test script that runs both CPU and GPU with detailed output
test_code = '''
import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import instantiate_models
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.minimizer import Minimizer
from pycalphad.core.solver import InteriorPointSolver
import warnings
warnings.filterwarnings('ignore')

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

# Run CPU equilibrium with minimal conditions
print("\\n=== CPU Equilibrium ===")
eq_cpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 10})
print(f"CPU GM: {eq_cpu.GM.values[0]:.6f}")

# Run GPU equilibrium
print("\\n=== GPU Equilibrium ===")
eq_gpu = equilibrium(dbf, comps, phases, conds, calc_opts={'pdens': 10}, gpu=True)
print(f"GPU GM: {eq_gpu.GM.values[0]:.6f}")

print(f"\\nGM Difference: {abs(eq_cpu.GM.values[0] - eq_gpu.GM.values[0]):.6e}")
'''

with open('test_hessian_minimal.py', 'w') as f:
    f.write(test_code)

print("\nCreated test_hessian_minimal.py")
print("Run this script to see the equilibrium comparison.")