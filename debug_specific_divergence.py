#!/usr/bin/env python
"""Debug a specific divergence case in detail."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Pick a case with moderate divergence
conditions = {
    'T': 1000,
    'P': 101325,
    'X(TI)': 0.01  # This showed 5.67 J/mol difference
}

print("="*80)
print("DEBUGGING SPECIFIC DIVERGENCE: T=1000K, X(TI)=0.01")
print("="*80)

# Run CPU calculation
print("\nCPU CALCULATION:")
print("-"*40)
reset_debug_session()
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = eq_cpu.GM.values.item()
cpu_np = eq_cpu.NP.values.flatten()
cpu_phases = eq_cpu.Phase.values.flatten()

print(f"GM: {cpu_gm:.6f} J/mol")
print("Phase amounts:")
for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
    if amount > 1e-6:
        print(f"  {phase}: {amount:.6f}")
        
# Get site fractions if available
if hasattr(eq_cpu, 'Y'):
    cpu_y = eq_cpu.Y.values
    print("Site fractions:")
    for i in range(cpu_y.shape[1]):
        if cpu_np[i] > 1e-6:
            phase = cpu_phases[i]
            y_vals = cpu_y[:, i, 0, 0]
            valid_y = y_vals[~np.isnan(y_vals)]
            if len(valid_y) == 2:  # BCC has 2 site fractions
                print(f"  {phase}: Y(NB)={valid_y[0]:.6f}, Y(TI)={valid_y[1]:.6f}")

# Run GPU calculation
print("\nGPU CALCULATION:")
print("-"*40)
reset_debug_session()
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = eq_gpu.GM.values.item()
gpu_np = eq_gpu.NP.values.flatten()
gpu_phases = eq_gpu.Phase.values.flatten()

print(f"GM: {gpu_gm:.6f} J/mol")
print("Phase amounts:")
for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
    if amount > 1e-6:
        print(f"  {phase}: {amount:.6f}")
        
# Get site fractions if available
if hasattr(eq_gpu, 'Y'):
    gpu_y = eq_gpu.Y.values
    print("Site fractions:")
    for i in range(gpu_y.shape[1]):
        if gpu_np[i] > 1e-6:
            phase = gpu_phases[i]
            y_vals = gpu_y[:, i, 0, 0]
            valid_y = y_vals[~np.isnan(y_vals)]
            if len(valid_y) == 2:  # BCC has 2 site fractions
                print(f"  {phase}: Y(NB)={valid_y[0]:.6f}, Y(TI)={valid_y[1]:.6f}")

print("\nCOMPARISON:")
print("-"*40)
print(f"GM difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
print(f"Relative error: {abs(cpu_gm - gpu_gm)/abs(cpu_gm)*100:.3f}%")

# Check if same phases are stable
cpu_stable = set(cpu_phases[cpu_np > 1e-6])
gpu_stable = set(gpu_phases[gpu_np > 1e-6])
print(f"\nStable phases match: {cpu_stable == gpu_stable}")
if cpu_stable != gpu_stable:
    print(f"  CPU stable phases: {cpu_stable}")
    print(f"  GPU stable phases: {gpu_stable}")

# Check phase amounts
print("\nPhase amount differences:")
for i in range(min(len(cpu_np), len(gpu_np))):
    if cpu_np[i] > 1e-6 or gpu_np[i] > 1e-6:
        diff = abs(cpu_np[i] - gpu_np[i])
        print(f"  Phase {i}: CPU={cpu_np[i]:.6f}, GPU={gpu_np[i]:.6f}, Δ={diff:.6f}")

# Likely cause analysis
print("\n" + "="*80)
print("LIKELY CAUSES OF DIVERGENCE:")
print("="*80)

# Check for two-phase region
num_stable_cpu = sum(cpu_np > 1e-6)
num_stable_gpu = sum(gpu_np > 1e-6)
print(f"Number of stable phases: CPU={num_stable_cpu}, GPU={num_stable_gpu}")

if num_stable_cpu != num_stable_gpu:
    print("\n*** DIFFERENT NUMBER OF STABLE PHASES ***")
    print("This suggests divergence in phase stability determination:")
    print("- Phase addition/removal logic may differ")
    print("- Driving force calculations may have numerical differences")
    print("- Convergence criteria may be different")
elif num_stable_cpu == 2:
    print("\n*** TWO-PHASE REGION ***")
    print("Both found 2 phases but with different amounts/compositions")
    print("This suggests:")
    print("- Different tie-line calculation")
    print("- Matrix solver precision differences")
    print("- Different starting points for iteration")
else:
    print("\n*** SINGLE PHASE REGION ***")
    print("Divergence in single phase suggests:")
    print("- Different energy function evaluation")
    print("- Numerical precision in site fraction optimization")
    print("- Different convergence paths")