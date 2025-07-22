#!/usr/bin/env python
"""Analyze iteration-by-iteration debug output to find divergence."""

import re

def extract_iteration_data(filename):
    """Extract detailed iteration data from debug output."""
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split into CPU and GPU sections
    cpu_section = content.split("GPU Calculation:")[0]
    gpu_section = content.split("GPU Calculation:")[1] if "GPU Calculation:" in content else ""
    
    def parse_section(text, prefix):
        """Parse iteration data from a section."""
        iterations = {}
        
        # Pattern to find iteration blocks
        iter_pattern = rf'\[{prefix}.*?\] ===== (?:AFTER )?ITERATION (\d+) ====='
        
        # Split by iteration markers
        iter_matches = list(re.finditer(iter_pattern, text))
        
        for i, match in enumerate(iter_matches):
            iter_num = int(match.group(1))
            start = match.start()
            end = iter_matches[i+1].start() if i+1 < len(iter_matches) else len(text)
            iter_block = text[start:end]
            
            # Extract data from iteration block
            data = {
                'chemical_potentials': [],
                'phases': [],
                'phase_amt_sum': None,
                'mass_residual': None,
                'num_stable_phases': None
            }
            
            # Chemical potentials
            cp_match = re.search(rf'\[{prefix}.*?\] Chemical potentials: \[([-\d.e+\s]+)\]', iter_block)
            if cp_match:
                cp_text = cp_match.group(1)
                # Split by whitespace and filter out empty strings
                cp_values = [x for x in re.split(r'\s+', cp_text.strip()) if x]
                data['chemical_potentials'] = [float(x) for x in cp_values]
            
            # Phase data
            phase_pattern = rf'\[{prefix}.*?\] Phase (\d+).*?:\s*\n.*?phase_amt.*?formula units.*?: ([\d.e+-]+)\s*\n.*?energy: ([-\d.e+]+)'
            for phase_match in re.finditer(phase_pattern, iter_block, re.MULTILINE | re.DOTALL):
                phase_num = int(phase_match.group(1))
                amount = float(phase_match.group(2))
                energy = float(phase_match.group(3))
                
                # Get site fractions
                sf_pattern = rf'Site fractions: \[([\d.e+\s-]+)\]'
                sf_match = re.search(sf_pattern, iter_block[phase_match.start():phase_match.end()+200])
                site_fractions = []
                if sf_match:
                    sf_text = sf_match.group(1)
                    # Split by whitespace
                    sf_values = [x for x in re.split(r'\s+', sf_text.strip()) if x]
                    site_fractions = [float(x) for x in sf_values]
                
                data['phases'].append({
                    'index': phase_num,
                    'amount': amount,
                    'energy': energy,
                    'site_fractions': site_fractions
                })
            
            # Number of stable phases
            stable_match = re.search(rf'\[{prefix}.*?\] num_stable_phases = (\d+)', iter_block)
            if stable_match:
                data['num_stable_phases'] = int(stable_match.group(1))
            
            iterations[iter_num] = data
        
        return iterations
    
    cpu_iterations = parse_section(cpu_section, 'CPU')
    gpu_iterations = parse_section(gpu_section, 'GPU')
    
    return cpu_iterations, gpu_iterations

# Analyze the debug output
print("Analyzing iteration data from full_debug_trace.txt...")
print("Threshold: 0.000001 J")
print("=" * 80)

try:
    cpu_iters, gpu_iters = extract_iteration_data('full_debug_trace.txt')
    
    # Compare iterations
    max_iter = max(max(cpu_iters.keys(), default=0), max(gpu_iters.keys(), default=0))
    
    first_divergence = None
    
    for i in range(max_iter + 1):
        if i in cpu_iters and i in gpu_iters:
            cpu_data = cpu_iters[i]
            gpu_data = gpu_iters[i]
            
            print(f"\nIteration {i}:")
            
            # Check number of stable phases
            if cpu_data['num_stable_phases'] and gpu_data['num_stable_phases']:
                print(f"  Stable phases: CPU={cpu_data['num_stable_phases']}, GPU={gpu_data['num_stable_phases']}")
                if cpu_data['num_stable_phases'] != gpu_data['num_stable_phases']:
                    print("  *** PHASE COUNT MISMATCH ***")
            
            # Compare chemical potentials
            if cpu_data['chemical_potentials'] and gpu_data['chemical_potentials']:
                for j, (cpu_cp, gpu_cp) in enumerate(zip(cpu_data['chemical_potentials'], gpu_data['chemical_potentials'])):
                    diff = abs(cpu_cp - gpu_cp)
                    print(f"  μ[{j}]: CPU={cpu_cp:.9f}, GPU={gpu_cp:.9f}, diff={diff:.9e}")
                    if diff > 0.000001 and first_divergence is None:
                        first_divergence = (i, 'chemical_potential', j, diff)
                        print(f"    *** FIRST DIVERGENCE > 0.000001 J ***")
            
            # Compare phases
            active_cpu_phases = [p for p in cpu_data['phases'] if p['amount'] > 1e-10]
            active_gpu_phases = [p for p in gpu_data['phases'] if p['amount'] > 1e-10]
            
            print(f"  Active phases: CPU={len(active_cpu_phases)}, GPU={len(active_gpu_phases)}")
            
            # Compare each phase
            for cpu_phase in cpu_data['phases']:
                # Find matching GPU phase
                gpu_phase = next((p for p in gpu_data['phases'] if p['index'] == cpu_phase['index']), None)
                
                if gpu_phase:
                    if cpu_phase['amount'] > 1e-10 or gpu_phase['amount'] > 1e-10:
                        energy_diff = abs(cpu_phase['energy'] - gpu_phase['energy'])
                        amount_diff = abs(cpu_phase['amount'] - gpu_phase['amount'])
                        
                        print(f"  Phase {cpu_phase['index']}:")
                        print(f"    Amount: CPU={cpu_phase['amount']:.9e}, GPU={gpu_phase['amount']:.9e}, diff={amount_diff:.9e}")
                        print(f"    Energy: CPU={cpu_phase['energy']:.9f}, GPU={gpu_phase['energy']:.9f}, diff={energy_diff:.9e}")
                        
                        if energy_diff > 0.000001 and first_divergence is None:
                            first_divergence = (i, 'phase_energy', cpu_phase['index'], energy_diff)
                            print(f"      *** FIRST DIVERGENCE > 0.000001 J ***")
                        
                        # Compare site fractions
                        if cpu_phase['site_fractions'] and gpu_phase['site_fractions']:
                            for k, (cpu_sf, gpu_sf) in enumerate(zip(cpu_phase['site_fractions'], gpu_phase['site_fractions'])):
                                sf_diff = abs(cpu_sf - gpu_sf)
                                if sf_diff > 1e-9:
                                    print(f"    Y[{k}]: CPU={cpu_sf:.9f}, GPU={gpu_sf:.9f}, diff={sf_diff:.9e}")
    
    if first_divergence:
        iter_num, var_type, index, diff = first_divergence
        print(f"\n*** SUMMARY: FIRST DIVERGENCE > 0.000001 J ***")
        print(f"  Iteration: {iter_num}")
        print(f"  Variable type: {var_type}")
        print(f"  Index: {index}")
        print(f"  Difference: {diff:.9e} J")
        
        # Provide context about what happened
        if iter_num > 0:
            print(f"\n  Context:")
            print(f"    CPU phases in iter {iter_num-1}: {cpu_iters.get(iter_num-1, {}).get('num_stable_phases', 'unknown')}")
            print(f"    GPU phases in iter {iter_num-1}: {gpu_iters.get(iter_num-1, {}).get('num_stable_phases', 'unknown')}")
            print(f"    CPU phases in iter {iter_num}: {cpu_iters[iter_num].get('num_stable_phases', 'unknown')}")
            print(f"    GPU phases in iter {iter_num}: {gpu_iters[iter_num].get('num_stable_phases', 'unknown')}")
    else:
        print("\nNo divergence > 0.000001 J found in captured iteration data")
        
except Exception as e:
    print(f"Error analyzing data: {e}")
    import traceback
    traceback.print_exc()