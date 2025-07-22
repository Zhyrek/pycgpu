#!/usr/bin/env python
"""Detailed trace of every iteration to find exact divergence point."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import re

# Load database
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 600, v.P: 101325, v.X('TI'): 0.1, v.N: 1}

print("Tracing iterations for 600K, X(TI)=0.1 case...")
print("Looking for divergence > 0.000001 J")
print("=" * 80)

# CPU calculation with full output
print("\n=== CPU CALCULATION ===")
import subprocess
import os
# Set debug environment variable
env = os.environ.copy()
env['PYCALPHAD_DEBUG_EQSOLVER'] = '1'
cpu_proc = subprocess.Popen(
    ['python', '-c', '''
import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = '1'

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = ["BCC_A2"]
conds = {v.T: 600, v.P: 101325, v.X("TI"): 0.1, v.N: 1}

cpu_result = equilibrium(dbf, comps, phases, conds, calc_opts={"pdens": 50})
print(f"CPU_FINAL_GM: {float(cpu_result.GM.values)}")
'''],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env
)
cpu_output, cpu_err = cpu_proc.communicate()

# Save CPU debug output
with open('cpu_debug_output.txt', 'w') as f:
    f.write(cpu_err)

# GPU calculation with full output  
print("\n=== GPU CALCULATION ===")
gpu_proc = subprocess.Popen(
    ['python', '-c', '''
import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = '1'

dbf = Database("NbTi.tdb")
comps = ["NB", "TI", "VA"]
phases = ["BCC_A2"]
conds = {v.T: 600, v.P: 101325, v.X("TI"): 0.1, v.N: 1}

gpu_result = equilibrium(dbf, comps, phases, conds, calc_opts={"pdens": 50}, gpu=True)
print(f"GPU_FINAL_GM: {float(gpu_result.GM.values)}")
'''],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env
)
gpu_output, gpu_err = gpu_proc.communicate()

# Save GPU debug output
with open('gpu_debug_output.txt', 'w') as f:
    f.write(gpu_err)

# Parse iteration data from debug output
def parse_iterations(output_text):
    """Extract iteration data from debug output."""
    iterations = {}
    current_iter = None
    
    # Patterns to match
    iter_pattern = r'\[(?:CPU|GPU).*?\] ===== (?:AFTER )?ITERATION (\d+)'
    energy_pattern = r'energy[:\s=]+(-?\d+\.?\d*(?:e[+-]?\d+)?)'
    phase_amt_pattern = r'phase_amt.*?formula units.*?[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)'
    chem_pot_pattern = r'Chemical potentials.*?\[([-\d.e+, ]+)\]'
    site_frac_pattern = r'Site fractions.*?\[([\d.e+, -]+)\]'
    gm_pattern = r'GM[:\s=]+(-?\d+\.?\d*(?:e[+-]?\d+)?)'
    
    lines = output_text.split('\n')
    for i, line in enumerate(lines):
        # Check for iteration marker
        iter_match = re.search(iter_pattern, line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            iterations[current_iter] = {
                'phases': [],
                'chemical_potentials': None,
                'gm': None
            }
        
        if current_iter is not None:
            # Extract energy
            energy_match = re.search(energy_pattern, line)
            if energy_match and 'phase' in line.lower():
                energy = float(energy_match.group(1))
                
                # Extract phase amount
                amt_match = re.search(phase_amt_pattern, line)
                if amt_match:
                    amt = float(amt_match.group(1))
                    
                    # Extract site fractions
                    sf_match = re.search(site_frac_pattern, lines[i+5] if i+5 < len(lines) else "")
                    if sf_match:
                        site_fracs = [float(x) for x in sf_match.group(1).split(',')]
                    else:
                        site_fracs = []
                    
                    iterations[current_iter]['phases'].append({
                        'energy': energy,
                        'amount': amt,
                        'site_fractions': site_fracs
                    })
            
            # Extract chemical potentials
            cp_match = re.search(chem_pot_pattern, line)
            if cp_match:
                iterations[current_iter]['chemical_potentials'] = [
                    float(x.strip()) for x in cp_match.group(1).split(',') if x.strip()
                ]
            
            # Extract GM
            gm_match = re.search(gm_pattern, line)
            if gm_match and 'energy' not in line:
                iterations[current_iter]['gm'] = float(gm_match.group(1))
    
    return iterations

# Parse both outputs
cpu_iterations = parse_iterations(cpu_err)
gpu_iterations = parse_iterations(gpu_err)

print("\n=== ITERATION COMPARISON ===")
print("Threshold: 0.000001 J")

# Compare iterations
max_iter = max(max(cpu_iterations.keys(), default=0), max(gpu_iterations.keys(), default=0))

first_divergence = None
for i in range(max_iter + 1):
    if i in cpu_iterations and i in gpu_iterations:
        cpu_data = cpu_iterations[i]
        gpu_data = gpu_iterations[i]
        
        print(f"\nIteration {i}:")
        
        # Compare number of phases
        cpu_phases = len([p for p in cpu_data['phases'] if p['amount'] > 1e-10])
        gpu_phases = len([p for p in gpu_data['phases'] if p['amount'] > 1e-10])
        print(f"  Active phases: CPU={cpu_phases}, GPU={gpu_phases}")
        
        # Compare phase energies and amounts
        for j, (cpu_phase, gpu_phase) in enumerate(zip(cpu_data['phases'], gpu_data['phases'])):
            if cpu_phase['amount'] > 1e-10 or gpu_phase['amount'] > 1e-10:
                energy_diff = abs(cpu_phase['energy'] - gpu_phase['energy'])
                amt_diff = abs(cpu_phase['amount'] - gpu_phase['amount'])
                
                print(f"  Phase {j}:")
                print(f"    Energy: CPU={cpu_phase['energy']:.9f}, GPU={gpu_phase['energy']:.9f}, diff={energy_diff:.9e}")
                print(f"    Amount: CPU={cpu_phase['amount']:.9e}, GPU={gpu_phase['amount']:.9e}, diff={amt_diff:.9e}")
                
                if cpu_phase['site_fractions'] and gpu_phase['site_fractions']:
                    sf_diff = max(abs(c - g) for c, g in zip(cpu_phase['site_fractions'], gpu_phase['site_fractions']))
                    print(f"    Max site fraction diff: {sf_diff:.9e}")
                
                if energy_diff > 0.000001 and first_divergence is None:
                    first_divergence = (i, j, 'energy', energy_diff)
                    print(f"    *** FIRST DIVERGENCE > 0.000001 J ***")
        
        # Compare chemical potentials
        if cpu_data['chemical_potentials'] and gpu_data['chemical_potentials']:
            cp_diffs = [abs(c - g) for c, g in zip(cpu_data['chemical_potentials'], gpu_data['chemical_potentials'])]
            max_cp_diff = max(cp_diffs)
            print(f"  Max chemical potential diff: {max_cp_diff:.9e}")
            
            if max_cp_diff > 0.000001 and first_divergence is None:
                first_divergence = (i, -1, 'chemical_potential', max_cp_diff)
                print(f"    *** FIRST DIVERGENCE > 0.000001 J ***")

# Extract final values
cpu_final_gm = float(re.search(r'CPU_FINAL_GM: ([-\d.]+)', cpu_output).group(1))
gpu_final_gm = float(re.search(r'GPU_FINAL_GM: ([-\d.]+)', gpu_output).group(1))

print(f"\n=== FINAL RESULTS ===")
print(f"CPU GM: {cpu_final_gm:.9f}")
print(f"GPU GM: {gpu_final_gm:.9f}")
print(f"Difference: {abs(cpu_final_gm - gpu_final_gm):.9e}")

if first_divergence:
    iter_num, phase_num, var_type, diff = first_divergence
    print(f"\n*** FIRST DIVERGENCE FOUND ***")
    print(f"Iteration: {iter_num}")
    if phase_num >= 0:
        print(f"Phase: {phase_num}")
    print(f"Variable: {var_type}")
    print(f"Difference: {diff:.9e} J")
else:
    print(f"\nNo divergence > 0.000001 J found in iteration data")