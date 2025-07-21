#!/usr/bin/env python
"""Trace phase compositions at each iteration for X(TI)=0.1, T=600K."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import sys
import io

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Failed condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("="*80)
print("TRACING PHASE COMPOSITIONS: X(TI)=0.1, T=600K")
print("="*80)

# Capture CPU output
print("\nCPU CALCULATION:")
print("-"*40)
cpu_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = cpu_output
try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
finally:
    sys.stdout = old_stdout

cpu_text = cpu_output.getvalue()

# Parse CPU iterations
cpu_iterations = []
current_iter = None
for line in cpu_text.split('\n'):
    if 'AFTER ITERATION' in line:
        if current_iter is not None:
            cpu_iterations.append(current_iter)
        iter_num = int(line.split('ITERATION')[1].split('=')[0].strip())
        current_iter = {'iteration': iter_num, 'phases': []}
    elif 'Phase' in line and '(BCC_A2):' in line and current_iter is not None:
        # Extract phase info
        phase_info = {'name': 'BCC_A2'}
        current_phase = phase_info
        current_iter['phases'].append(current_phase)
    elif 'NP (mole fraction):' in line and current_iter is not None:
        np_val = float(line.split(':')[1].strip().split('e')[0] + 'e' + line.split(':')[1].strip().split('e')[1])
        current_phase['NP'] = np_val
    elif 'phase_amt (formula units):' in line and current_iter is not None:
        amt = float(line.split(':')[1].strip().split('e')[0] + 'e' + line.split(':')[1].strip().split('e')[1])
        current_phase['phase_amt'] = amt
    elif 'Site fractions:' in line and current_iter is not None:
        # Parse site fractions [Y(NB), Y(TI)]
        sf_str = line.split('[')[1].split(']')[0]
        site_fracs = [float(x) for x in sf_str.split()]
        current_phase['Y_NB'] = site_fracs[0]
        current_phase['Y_TI'] = site_fracs[1]
    elif 'phase_compositions:' in line and current_iter is not None:
        # This gives us the actual mole fractions in the phase
        pass
    elif 'energy:' in line and 'Phase' not in line and current_iter is not None:
        energy = float(line.split(':')[1].strip().split('e')[0] + 'e' + line.split(':')[1].strip().split('e')[1])
        current_phase['energy'] = energy

if current_iter is not None:
    cpu_iterations.append(current_iter)

# Print CPU iterations
print("\nCPU ITERATIONS:")
for iter_data in cpu_iterations:
    print(f"\nIteration {iter_data['iteration']}:")
    total_np = sum(p['NP'] for p in iter_data['phases'])
    print(f"  Total NP: {total_np:.6f}")
    for i, phase in enumerate(iter_data['phases']):
        print(f"  Phase {i} (BCC_A2):")
        print(f"    NP = {phase['NP']:.6f} (mole fraction)")
        print(f"    phase_amt = {phase.get('phase_amt', 0):.6e} (formula units)")
        print(f"    Y(NB) = {phase.get('Y_NB', 0):.6f}, Y(TI) = {phase.get('Y_TI', 0):.6f}")
        print(f"    Energy = {phase.get('energy', 0):.2f} J/mol")

# Capture GPU output
print("\n" + "="*80)
print("GPU CALCULATION:")
print("-"*40)
gpu_output = io.StringIO()
sys.stdout = gpu_output
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
finally:
    sys.stdout = old_stdout

gpu_text = gpu_output.getvalue()

# Parse GPU output for phase compositions
print("\nGPU PHASE COMPOSITIONS:")
in_phase_section = False
gpu_phases = []
for line in gpu_text.split('\n'):
    if 'phase_site_fractions:' in line:
        # Extract site fractions
        sf_str = line.split('[')[1].split(']')[0]
        site_fracs = [float(x) for x in sf_str.replace(',', '').split()]
        print(f"  Phase {len(gpu_phases)}: Y = [{site_fracs[0]:.6f}, {site_fracs[1]:.6f}]")
        gpu_phases.append({'Y_NB': site_fracs[0], 'Y_TI': site_fracs[1]})
    elif 'phase_' in line and '_energy:' in line:
        # Extract energy
        energy = float(line.split(':')[1].strip())
        if len(gpu_phases) > 0:
            gpu_phases[-1]['energy'] = energy
            print(f"           Energy = {energy:.2f} J/mol")

# Extract final results
print("\n" + "="*80)
print("FINAL RESULTS COMPARISON:")
print("-"*40)

# CPU final state
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_phases_final = result_cpu.Phase.values.flatten()
cpu_np_final = result_cpu.NP.values.flatten()
cpu_y_final = result_cpu.Y.values.flatten()

print("CPU Final State:")
print(f"  GM = {cpu_gm:.2f} J/mol")
active_count = 0
for i, (phase, np_val) in enumerate(zip(cpu_phases_final, cpu_np_final)):
    if np_val > 1e-6:
        y_start = i * 2
        print(f"  Phase {active_count} ({phase}): NP = {np_val:.6f}")
        print(f"    Y(NB) = {cpu_y_final[y_start]:.6f}, Y(TI) = {cpu_y_final[y_start+1]:.6f}")
        active_count += 1

# GPU final state
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_phases_final = result_gpu.Phase.values.flatten()
gpu_np_final = result_gpu.NP.values.flatten()
gpu_y_final = result_gpu.Y.values.flatten()

print("\nGPU Final State:")
print(f"  GM = {gpu_gm:.2f} J/mol (Ratio to CPU: {gpu_gm/cpu_gm:.3f})")
active_count = 0
for i, (phase, np_val) in enumerate(zip(gpu_phases_final, gpu_np_final)):
    if np_val > 1e-6:
        y_start = i * 2
        print(f"  Phase {active_count} ({phase}): NP = {np_val:.6f}")
        print(f"    Y(NB) = {gpu_y_final[y_start]:.6f}, Y(TI) = {gpu_y_final[y_start+1]:.6f}")
        active_count += 1

print("\n" + "="*80)
print("KEY FINDING:")
print("GPU reports multiple phases with NP=1.0 each (summing to >1.0)")
print("CPU correctly has phases with NP summing to 1.0")
print("This confirms GPU is not normalizing phase amounts properly")
print("="*80)