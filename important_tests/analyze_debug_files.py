#!/usr/bin/env python
"""Analyze the debug output files to find key differences."""

import re

def extract_iterations(content):
    """Extract solver iterations from debug output."""
    iterations = []
    
    # Look for iteration markers
    patterns = [
        r'Iteration (\d+)',
        r'iter=(\d+)',
        r'ITERATION (\d+)',
        r'\[Iteration (\d+)\]'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if matches:
            iterations.extend([int(m) for m in matches])
    
    return sorted(set(iterations))

def extract_chemical_potentials(content):
    """Extract chemical potential values."""
    chem_pots = []
    
    # Pattern to match chemical potential arrays
    patterns = [
        r'chemical_potentials: \[([-\d.e+]+), ([-\d.e+]+), ([-\d.e+]+)',
        r'chemical_potentials\[0\] = ([-\d.e+]+)',
        r'mu\[0\] = ([-\d.e+]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if matches:
            if isinstance(matches[0], tuple):
                chem_pots.extend([tuple(float(x) for x in match) for match in matches])
            else:
                chem_pots.extend([float(m) for m in matches])
    
    return chem_pots

def extract_phase_amounts(content):
    """Extract phase amount values."""
    phase_amounts = []
    
    patterns = [
        r'phase_amt\[(\d+)\] = ([-\d.e+]+)',
        r'NP\[(\d+)\] = ([-\d.e+]+)',
        r'phase_amounts: \[([-\d.e+, ]+)\]'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if isinstance(match, tuple):
                phase_amounts.append((int(match[0]), float(match[1])))
            else:
                # Parse array string
                values = [float(x.strip()) for x in match.split(',') if x.strip()]
                phase_amounts.extend(enumerate(values))
    
    return phase_amounts

def extract_convergence_info(content):
    """Extract convergence-related information."""
    convergence = {
        'converged': False,
        'final_residual': None,
        'num_iterations': None,
        'messages': []
    }
    
    # Check for convergence
    if 'CONVERGED' in content.upper():
        convergence['converged'] = True
    
    # Extract residuals
    residual_pattern = r'residual[:\s]+([-\d.e+]+)'
    matches = re.findall(residual_pattern, content, re.IGNORECASE)
    if matches:
        convergence['final_residual'] = float(matches[-1])
    
    # Extract convergence messages
    for line in content.split('\n'):
        if 'converg' in line.lower() or 'residual' in line.lower():
            convergence['messages'].append(line.strip())
    
    return convergence

def extract_gm_values(content):
    """Extract GM (Gibbs energy) values."""
    gm_values = []
    
    patterns = [
        r'GM[:\s]+([-\d.e+]+)',
        r'FINAL GM VALUE[:\s]+([-\d.e+]+)',
        r'gm = ([-\d.e+]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        gm_values.extend([float(m) for m in matches])
    
    return gm_values

def main():
    """Analyze debug files."""
    
    print("=" * 100)
    print("DEBUG FILE ANALYSIS")
    print("=" * 100)
    
    # Read files
    with open('cpu_debug_output.txt', 'r') as f:
        cpu_content = f.read()
    
    with open('gpu_debug_output.txt', 'r') as f:
        gpu_content = f.read()
    
    print(f"\nFile sizes:")
    print(f"  CPU: {len(cpu_content):,} bytes")
    print(f"  GPU: {len(gpu_content):,} bytes")
    
    # Extract iterations
    print("\n" + "-" * 50)
    print("SOLVER ITERATIONS:")
    print("-" * 50)
    
    cpu_iterations = extract_iterations(cpu_content)
    gpu_iterations = extract_iterations(gpu_content)
    
    print(f"CPU iterations found: {cpu_iterations}")
    print(f"GPU iterations found: {gpu_iterations[:20]}..." if len(gpu_iterations) > 20 else f"GPU iterations found: {gpu_iterations}")
    print(f"CPU total iterations: {max(cpu_iterations) if cpu_iterations else 0}")
    print(f"GPU total iterations: {max(gpu_iterations) if gpu_iterations else 0}")
    
    # Extract chemical potentials
    print("\n" + "-" * 50)
    print("CHEMICAL POTENTIALS:")
    print("-" * 50)
    
    cpu_chempots = extract_chemical_potentials(cpu_content)
    gpu_chempots = extract_chemical_potentials(gpu_content)
    
    print(f"CPU chemical potential entries: {len(cpu_chempots)}")
    print(f"GPU chemical potential entries: {len(gpu_chempots)}")
    
    if gpu_chempots:
        print("\nFirst few GPU chemical potentials:")
        for i, cp in enumerate(gpu_chempots[:5]):
            print(f"  {i}: {cp}")
        
        print("\nLast few GPU chemical potentials:")
        for i, cp in enumerate(gpu_chempots[-5:], len(gpu_chempots)-5):
            print(f"  {i}: {cp}")
    
    # Extract GM values
    print("\n" + "-" * 50)
    print("GIBBS ENERGY (GM) VALUES:")
    print("-" * 50)
    
    cpu_gm = extract_gm_values(cpu_content)
    gpu_gm = extract_gm_values(gpu_content)
    
    if cpu_gm:
        print(f"CPU final GM: {cpu_gm[-1]:.6f}")
    if gpu_gm:
        print(f"GPU final GM: {gpu_gm[-1]:.6f}")
    
    if cpu_gm and gpu_gm:
        diff = abs(cpu_gm[-1] - gpu_gm[-1])
        print(f"Difference: {diff:.6f} J/mol")
    
    # Extract convergence info
    print("\n" + "-" * 50)
    print("CONVERGENCE INFORMATION:")
    print("-" * 50)
    
    cpu_conv = extract_convergence_info(cpu_content)
    gpu_conv = extract_convergence_info(gpu_content)
    
    print(f"CPU converged: {cpu_conv['converged']}")
    print(f"GPU converged: {gpu_conv['converged']}")
    
    if cpu_conv['final_residual']:
        print(f"CPU final residual: {cpu_conv['final_residual']}")
    if gpu_conv['final_residual']:
        print(f"GPU final residual: {gpu_conv['final_residual']}")
    
    # Look for specific error patterns
    print("\n" + "-" * 50)
    print("ERROR/WARNING PATTERNS:")
    print("-" * 50)
    
    # Check for NaN or Inf
    if 'nan' in gpu_content.lower() or 'inf' in gpu_content.lower():
        print("⚠ GPU output contains NaN or Inf values!")
    
    # Check for matrix singularity
    if 'singular' in gpu_content.lower():
        print("⚠ GPU encountered singular matrix!")
    
    # Check for huge values (potential overflow)
    huge_value_pattern = r'[-\d.]+e\+(\d{2,})'
    huge_matches = re.findall(huge_value_pattern, gpu_content)
    if huge_matches:
        max_exp = max(int(m) for m in huge_matches)
        if max_exp > 10:
            print(f"⚠ GPU has very large values (up to e+{max_exp})")
    
    # Extract specific iteration details
    print("\n" + "-" * 50)
    print("DETAILED ITERATION ANALYSIS:")
    print("-" * 50)
    
    # Look for equilibrium matrix output
    cpu_matrices = re.findall(r'EQUILIBRIUM_MATRIX.*?RHS: [-\d.e+]+', cpu_content, re.DOTALL)
    gpu_matrices = re.findall(r'EQUILIBRIUM_MATRIX.*?RHS: [-\d.e+]+', gpu_content, re.DOTALL)
    
    print(f"CPU equilibrium matrices found: {len(cpu_matrices)}")
    print(f"GPU equilibrium matrices found: {len(gpu_matrices)}")
    
    # Look for divergence point
    print("\n" + "-" * 50)
    print("FINDING DIVERGENCE POINT:")
    print("-" * 50)
    
    # Extract all chemical potential updates
    gpu_chempot_updates = re.findall(r'\[GPU\]\s+chemical_potentials: \[([-\d.e+]+), ([-\d.e+]+)', gpu_content)
    
    if gpu_chempot_updates:
        print(f"Found {len(gpu_chempot_updates)} GPU chemical potential updates")
        
        # Look for where values start diverging significantly
        prev_values = None
        for i, values in enumerate(gpu_chempot_updates):
            mu0, mu1 = float(values[0]), float(values[1])
            
            if prev_values:
                delta0 = abs(mu0 - prev_values[0])
                delta1 = abs(mu1 - prev_values[1])
                
                if delta0 > 1e6 or delta1 > 1e6:
                    print(f"\n⚠ LARGE JUMP at update {i}:")
                    print(f"  Previous: [{prev_values[0]:.3e}, {prev_values[1]:.3e}]")
                    print(f"  Current:  [{mu0:.3e}, {mu1:.3e}]")
                    print(f"  Delta:    [{delta0:.3e}, {delta1:.3e}]")
                    break
            
            prev_values = (mu0, mu1)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()