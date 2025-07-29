#!/usr/bin/env python
"""Trace through solver iterations to find first divergence between CPU and GPU."""

import re
import numpy as np

def extract_iteration_data(content, iteration, is_gpu=False):
    """Extract data for a specific iteration."""
    prefix = "" if is_gpu else "\\[CPU.*?\\] "
    
    data = {
        'chemical_potentials': [],
        'phase_amounts': [],
        'phase_names': [],
        'site_fractions': [],
        'matrix_rows': [],
        'rhs': [],
        'solution': [],
        'converged': False,
        'gm': None
    }
    
    # Find iteration section
    if is_gpu:
        iter_pattern = f"GPU.*?iteration {iteration}(?::|\\s)"
        # For GPU, matrix info comes right after iteration marker
        matrix_section_start = content.find(f"[GPU EQUILIBRIUM MATRIX] Filling equilibrium system at iteration {iteration}")
    else:
        iter_pattern = f"CPU.*?Iteration {iteration}:"
        matrix_section_start = content.find(f"[CPU MATRIX DEBUG] construct_equilibrium_system called, state.iteration={iteration}")
    
    if matrix_section_start == -1:
        return data
    
    # Extract a reasonable section around this iteration
    section_end = matrix_section_start + 5000
    section = content[matrix_section_start:section_end]
    
    # Extract matrix rows
    if is_gpu:
        # GPU format: "Row N: +X.XXXe+00 ..."
        row_pattern = r"Row (\d+): ([\+\-\d\.e\s]+) \| RHS: ([\+\-\d\.e]+)"
        matches = re.findall(row_pattern, section)
        for row_num, coeffs, rhs in matches[:10]:  # Limit to first 10 rows
            data['matrix_rows'].append({
                'row': int(row_num),
                'coefficients': coeffs.strip(),
                'rhs': float(rhs)
            })
    else:
        # CPU format varies, look for Row patterns
        row_pattern = r"Row (\d+): ([\+\-\d\.e\s]+) \| RHS: ([\+\-\d\.e]+)"
        matches = re.findall(row_pattern, section)
        for row_num, coeffs, rhs in matches[:10]:
            data['matrix_rows'].append({
                'row': int(row_num),
                'coefficients': coeffs.strip(),
                'rhs': float(rhs)
            })
    
    # Extract solution
    if is_gpu:
        sol_match = re.search(r"solution.*?\[([\+\-\d\.e\s,]+)\]", section)
        if not sol_match:
            sol_match = re.search(r"Equilibrium solution at iteration \d+.*?: \[([\+\-\d\.e\s,]+)\]", section)
    else:
        sol_match = re.search(r"solution.*?\[([\+\-\d\.e\s,]+)\]", section)
    
    if sol_match:
        sol_str = sol_match.group(1).replace(',', ' ')
        data['solution'] = [float(x) for x in sol_str.split()]
    
    # Extract chemical potentials after solve
    if is_gpu:
        chem_pot_match = re.search(r"Chemical potentials: \[([\+\-\d\.e\s,]+)\]", section)
    else:
        chem_pot_match = re.search(r"chemical_potentials.*?\[([\+\-\d\.e\s,]+)\]", section)
    
    if chem_pot_match:
        pot_str = chem_pot_match.group(1).replace(',', ' ')
        data['chemical_potentials'] = [float(x) for x in pot_str.split()]
    
    # Extract phase data
    if is_gpu:
        # Look for phase updates
        phase_updates = re.findall(r"Phase (\d+):.*?old=([\d\.e\+\-]+).*?delta=([\d\.e\+\-]+).*?new=([\d\.e\+\-]+)", section)
        for phase_idx, old, delta, new in phase_updates:
            data['phase_amounts'].append({
                'phase_idx': int(phase_idx),
                'old': float(old),
                'delta': float(delta),
                'new': float(new)
            })
    
    return data

def compare_matrices(cpu_data, gpu_data, iteration):
    """Compare CPU and GPU matrices at a given iteration."""
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration} COMPARISON")
    print(f"{'='*80}")
    
    # Compare matrix dimensions
    cpu_rows = len(cpu_data['matrix_rows'])
    gpu_rows = len(gpu_data['matrix_rows'])
    
    print(f"\nMatrix dimensions:")
    print(f"  CPU: {cpu_rows} rows")
    print(f"  GPU: {gpu_rows} rows")
    
    if cpu_rows == 0 and gpu_rows == 0:
        print("  No matrix data found for this iteration")
        return False
    
    # Compare matrix rows
    if gpu_rows > 0:
        print(f"\nGPU Matrix at iteration {iteration}:")
        for row_data in gpu_data['matrix_rows']:
            print(f"  Row {row_data['row']}: {row_data['coefficients'][:50]}... | RHS: {row_data['rhs']:.6e}")
    
    # Compare solutions
    if gpu_data['solution'] and len(gpu_data['solution']) > 0:
        print(f"\nGPU Solution: {gpu_data['solution']}")
    
    # Compare chemical potentials
    if gpu_data['chemical_potentials']:
        print(f"\nGPU Chemical potentials: {gpu_data['chemical_potentials']}")
    
    # Show phase updates
    if gpu_data['phase_amounts']:
        print(f"\nGPU Phase updates:")
        for phase in gpu_data['phase_amounts']:
            print(f"  Phase {phase['phase_idx']}: {phase['old']:.6e} -> {phase['new']:.6e} (delta: {phase['delta']:.6e})")
            if phase['new'] < 1e-10:
                print(f"    *** PHASE {phase['phase_idx']} REMOVED! ***")
    
    return True

def main():
    """Main analysis function."""
    print("Loading debug output...")
    with open('med_t_debug.txt', 'r') as f:
        content = f.read()
    
    print("Analyzing solver iterations to find first divergence...")
    
    # First, let's understand the initial state
    print("\n" + "="*80)
    print("INITIAL STATE")
    print("="*80)
    
    # Find initial phase setup
    cpu_initial_match = re.search(r"\[CPU\] Phase values: \[(.*?)\]", content)
    gpu_initial_match = re.search(r"\[GPU\].*?phase_names: \[(.*?)\]", content)
    
    if cpu_initial_match:
        print(f"CPU initial phases: [{cpu_initial_match.group(1)}]")
    if gpu_initial_match:
        print(f"GPU initial phases: [{gpu_initial_match.group(1)}]")
    
    # Extract initial amounts
    cpu_np_match = re.search(r"\[CPU\] NP values: \[([\d\.\s]+)\]", content)
    gpu_np_match = re.search(r"\[GPU\].*?np_values: \[([\d\.\s]+)\]", content)
    
    if cpu_np_match:
        print(f"CPU initial NP: [{cpu_np_match.group(1)}]")
    if gpu_np_match:
        print(f"GPU initial NP: [{gpu_np_match.group(1)}]")
    
    # Now trace through iterations
    for iteration in range(5):  # Check first 5 iterations
        cpu_data = extract_iteration_data(content, iteration, is_gpu=False)
        gpu_data = extract_iteration_data(content, iteration, is_gpu=True)
        
        if compare_matrices(cpu_data, gpu_data, iteration):
            # Check for divergence
            if iteration == 0 and gpu_data['phase_amounts']:
                # Check if any phase was removed
                for phase in gpu_data['phase_amounts']:
                    if phase['new'] < 1e-10:
                        print(f"\n*** DIVERGENCE FOUND AT ITERATION {iteration} ***")
                        print(f"GPU removes phase {phase['phase_idx']} while CPU likely keeps it")
                        return
    
    # Additional analysis of the key iteration 0
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF ITERATION 0")
    print("="*80)
    
    # Extract the full GPU matrix at iteration 0
    gpu_matrix_match = re.search(
        r"\[GPU EQUILIBRIUM MATRIX\] Complete matrix at iteration 0.*?\n((?:.*?\n){8})",
        content
    )
    
    if gpu_matrix_match:
        print("\nGPU Complete Matrix at iteration 0:")
        print(gpu_matrix_match.group(1))
    
    # Look for the key difference in Row 5
    row5_match = re.search(r"Row 5: (.*?) \| RHS: ([\+\-\d\.e]+)", content)
    if row5_match:
        print(f"\nKey finding - GPU Row 5 (system amount constraint):")
        print(f"  Coefficients: {row5_match.group(1)}")
        print(f"  Note the coefficient 2.000e+01 (20.0) for ALCU_ZETA phase!")
        print(f"  This large coefficient may affect numerical stability")

if __name__ == "__main__":
    main()