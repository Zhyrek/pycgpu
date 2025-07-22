#!/usr/bin/env python
"""Analyze debug output to find first divergence > 0.000001 J."""

import re

def extract_iteration_values(text, prefix):
    """Extract chemical potentials and phase data per iteration."""
    iterations = {}
    
    # Pattern to find iteration blocks
    iter_pattern = rf'\[{prefix} TRACE\] ===== (?:AFTER )?ITERATION (\d+) ====='
    chem_pot_pattern = rf'\[{prefix} TRACE\] Chemical potentials: \[([-\d.e+\s]+)\]'
    phase_pattern = rf'\[{prefix} TRACE\] Phase (\d+).*?:\s*\n.*?phase_amt.*?:\s*([\d.e+-]+)\s*\n.*?energy:\s*([-\d.e+]+)'
    sf_pattern = rf'Site fractions: \[([\d.e+\s-]+)\]'
    
    # Find all iterations
    iter_matches = list(re.finditer(iter_pattern, text))
    
    for i, match in enumerate(iter_matches):
        iter_num = int(match.group(1))
        start = match.start()
        end = iter_matches[i+1].start() if i+1 < len(iter_matches) else len(text)
        iter_block = text[start:end]
        
        # Extract chemical potentials
        cp_match = re.search(chem_pot_pattern, iter_block)
        if cp_match:
            cp_text = cp_match.group(1)
            chemical_potentials = [float(x) for x in cp_text.split()]
        else:
            chemical_potentials = []
        
        # Extract phase data
        phases = []
        for phase_match in re.finditer(phase_pattern, iter_block, re.MULTILINE | re.DOTALL):
            phase_idx = int(phase_match.group(1))
            amount = float(phase_match.group(2))
            energy = float(phase_match.group(3))
            
            # Get site fractions
            sf_search_start = phase_match.start()
            sf_search_end = min(phase_match.end() + 500, end)
            sf_block = iter_block[sf_search_start:sf_search_end]
            sf_match = re.search(sf_pattern, sf_block)
            site_fractions = []
            if sf_match:
                sf_text = sf_match.group(1)
                site_fractions = [float(x) for x in sf_text.split()]
            
            phases.append({
                'index': phase_idx,
                'amount': amount,
                'energy': energy,
                'site_fractions': site_fractions
            })
        
        iterations[iter_num] = {
            'chemical_potentials': chemical_potentials,
            'phases': phases
        }
    
    return iterations

# Read the debug output
with open('new_debug_trace.txt', 'r') as f:
    content = f.read()

# Extract CPU and GPU iterations
cpu_iters = extract_iteration_values(content, 'CPU')
gpu_iters = extract_iteration_values(content, 'GPU')

print("Analyzing iterations for first divergence > 0.000001 J")
print("=" * 80)

# Find first divergence
first_divergence = None
threshold = 0.000001

for iter_num in sorted(set(cpu_iters.keys()) | set(gpu_iters.keys())):
    if iter_num in cpu_iters and iter_num in gpu_iters:
        cpu_data = cpu_iters[iter_num]
        gpu_data = gpu_iters[iter_num]
        
        print(f"\nIteration {iter_num}:")
        
        # Compare chemical potentials
        if cpu_data['chemical_potentials'] and gpu_data['chemical_potentials']:
            print("  Chemical potentials:")
            for i, (cpu_mu, gpu_mu) in enumerate(zip(cpu_data['chemical_potentials'], gpu_data['chemical_potentials'])):
                diff = abs(cpu_mu - gpu_mu)
                print(f"    μ[{i}]: CPU={cpu_mu:.9f}, GPU={gpu_mu:.9f}, diff={diff:.9e} J/mol")
                if diff > threshold and first_divergence is None:
                    first_divergence = (iter_num, 'chemical_potential', i, diff)
                    print(f"      *** FIRST DIVERGENCE > {threshold} J ***")
        
        # Compare phases
        print(f"  Active phases: CPU={len([p for p in cpu_data['phases'] if p['amount'] > 1e-10])}, GPU={len([p for p in gpu_data['phases'] if p['amount'] > 1e-10])}")
        
        for cpu_phase in cpu_data['phases']:
            gpu_phase = next((p for p in gpu_data['phases'] if p['index'] == cpu_phase['index']), None)
            
            if gpu_phase and (cpu_phase['amount'] > 1e-10 or gpu_phase['amount'] > 1e-10):
                energy_diff = abs(cpu_phase['energy'] - gpu_phase['energy'])
                amount_diff = abs(cpu_phase['amount'] - gpu_phase['amount'])
                
                print(f"  Phase {cpu_phase['index']}:")
                print(f"    Amount: CPU={cpu_phase['amount']:.9e}, GPU={gpu_phase['amount']:.9e}, diff={amount_diff:.9e}")
                print(f"    Energy: CPU={cpu_phase['energy']:.9f}, GPU={gpu_phase['energy']:.9f}, diff={energy_diff:.9e} J/mol")
                
                if energy_diff > threshold and first_divergence is None:
                    first_divergence = (iter_num, 'phase_energy', cpu_phase['index'], energy_diff)
                    print(f"      *** FIRST DIVERGENCE > {threshold} J ***")
                
                # Site fractions
                if cpu_phase['site_fractions'] and gpu_phase['site_fractions']:
                    sf_diffs = [abs(c - g) for c, g in zip(cpu_phase['site_fractions'], gpu_phase['site_fractions'])]
                    max_sf_diff = max(sf_diffs)
                    if max_sf_diff > 1e-9:
                        print(f"    Site fractions:")
                        for j, (cpu_sf, gpu_sf, diff) in enumerate(zip(cpu_phase['site_fractions'], gpu_phase['site_fractions'], sf_diffs)):
                            print(f"      Y[{j}]: CPU={cpu_sf:.9f}, GPU={gpu_sf:.9f}, diff={diff:.9e}")

if first_divergence:
    iter_num, var_type, index, diff = first_divergence
    print(f"\n*** SUMMARY: FIRST DIVERGENCE > {threshold} J ***")
    print(f"  Iteration: {iter_num}")
    print(f"  Variable type: {var_type}")
    print(f"  Index: {index}")
    print(f"  Difference: {diff:.9e} J")
    
    # Additional context
    if iter_num == 0:
        print("\n  Context: Divergence occurs immediately after the first equilibrium solve")
        print("  This suggests the equilibrium solver produces slightly different results")
        print("  even with identical starting conditions and constraints.")
    else:
        print(f"\n  Context at iteration {iter_num}:")
        if iter_num-1 in cpu_iters and iter_num-1 in gpu_iters:
            cpu_prev = cpu_iters[iter_num-1]
            gpu_prev = gpu_iters[iter_num-1]
            cpu_phases = len([p for p in cpu_prev['phases'] if p['amount'] > 1e-10])
            gpu_phases = len([p for p in gpu_prev['phases'] if p['amount'] > 1e-10])
            print(f"    Previous iteration had CPU={cpu_phases}, GPU={gpu_phases} active phases")
else:
    print("\nNo divergence > 0.000001 J found in iteration data")