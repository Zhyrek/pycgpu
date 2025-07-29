#\!/usr/bin/env python
"""Detailed trace of GPU solver iterations to find divergence."""

import re

def analyze_gpu_iteration_0():
    """Analyze GPU iteration 0 in detail."""
    with open('med_t_debug.txt', 'r') as f:
        content = f.read()
    
    print("="*80)
    print("GPU ITERATION 0 DETAILED ANALYSIS")
    print("="*80)
    
    # Find the GPU iteration 0 section
    gpu_iter0_start = content.find("[GPU EQUILIBRIUM MATRIX] Complete matrix at iteration 0")
    if gpu_iter0_start == -1:
        print("Could not find GPU iteration 0")
        return
    
    # Extract section up to next iteration or 5000 chars
    gpu_iter0_end = content.find("[GPU EQUILIBRIUM MATRIX] Filling equilibrium system at iteration 1", gpu_iter0_start)
    if gpu_iter0_end == -1:
        gpu_iter0_end = gpu_iter0_start + 5000
    
    gpu_section = content[gpu_iter0_start:gpu_iter0_end]
    
    # Extract key information
    print("\n1. EQUILIBRIUM MATRIX:")
    matrix_match = re.search(r"Complete matrix at iteration 0.*?\n((?:.*?Row \d+:.*?\n){6})", gpu_section, re.DOTALL)
    if matrix_match:
        print(matrix_match.group(1).strip())
    
    # Extract solution
    print("\n2. SOLUTION VECTOR:")
    sol_match = re.search(r"RHS after lstsq \(solution\): \[(.*?)\]", gpu_section)
    if sol_match:
        sol_values = sol_match.group(1).split()
        print(f"   Solution: {sol_values}")
        print(f"   Chemical potentials: μ₀={sol_values[0]}, μ₁={sol_values[1]}")
        print(f"   Phase amount changes: Δφ₀={sol_values[2]}, Δφ₁={sol_values[3]}, Δφ₂={sol_values[4]}")
    
    # Extract phase updates
    print("\n3. PHASE AMOUNT UPDATES:")
    phase_updates = re.findall(r"Phase (\d+): old=([\d\.e\+\-]+).*?delta=([\d\.e\+\-]+).*?new=([\d\.e\+\-]+)", gpu_section)
    for phase_idx, old, delta, new in phase_updates:
        old_f = float(old)
        delta_f = float(delta)
        new_f = float(new)
        print(f"   Phase {phase_idx}:")
        print(f"     Old amount: {old_f:.6e}")
        print(f"     Delta (from solution): {delta_f:.6e}")
        print(f"     New amount: {new_f:.6e}")
        if new_f < 1e-10:
            print(f"     *** REMOVED (below threshold) ***")
    
    # Extract step size info
    print("\n4. STEP SIZE LIMITING:")
    step_match = re.search(r"Step size limited to ([\d\.e\+\-]+)", gpu_section)
    if step_match:
        step_size = float(step_match.group(1))
        print(f"   Step size: {step_size:.6e}")
        print(f"   This means actual changes are: delta * step_size")
    
    # Show the phase removal
    print("\n5. KEY FINDING - PHASE REMOVAL:")
    removal_match = re.search(r"Phase 2 amount became very small.*?at iteration 0", gpu_section)
    if removal_match:
        print(f"   {removal_match.group(0)}")
        print(f"   This is where GPU diverges from CPU\!")
    
    # Extract the mass balance info
    print("\n6. MASS BALANCE:")
    mass_match = re.search(r"sum\(phase_amt\) = ([\d\.e\+\-]+)", gpu_section)
    if mass_match:
        total = float(mass_match.group(1))
        print(f"   Total phase amounts after update: {total:.6f}")
        if abs(total - 1.0) > 0.01:
            print(f"   *** WARNING: Mass not conserved\! Should be 1.0 ***")

def analyze_cpu_iteration_0():
    """Analyze CPU iteration 0 for comparison."""
    with open('med_t_debug.txt', 'r') as f:
        content = f.read()
    
    print("\n" + "="*80)
    print("CPU ITERATION 0 ANALYSIS")
    print("="*80)
    
    # Find CPU iteration 0
    cpu_iter0_start = content.find("[CPU MATRIX DEBUG] construct_equilibrium_system called, state.iteration=0")
    if cpu_iter0_start == -1:
        print("Could not find CPU iteration 0")
        return
    
    # Look for matrix rows
    cpu_section = content[cpu_iter0_start:cpu_iter0_start+3000]
    
    # Extract phase information
    print("\n1. CPU PHASE SETUP:")
    stable_phases_match = re.search(r"num_stable_phases = (\d+)", cpu_section)
    if stable_phases_match:
        print(f"   Number of stable phases: {stable_phases_match.group(1)}")
    
    # Look for matrix rows  
    print("\n2. CPU MATRIX ROWS (if available):")
    cpu_rows = re.findall(r"Row (\d+): ([\+\-\d\.e\s]+) \ < /dev/null |  RHS: ([\+\-\d\.e]+)", cpu_section)
    for row_num, coeffs, rhs in cpu_rows[:6]:
        print(f"   Row {row_num}: {coeffs[:50]}... | RHS: {rhs}")

def trace_divergence_cause():
    """Analyze why the equilibrium solution causes phase 2 removal."""
    print("\n" + "="*80)
    print("DIVERGENCE ROOT CAUSE ANALYSIS")
    print("="*80)
    
    print("\nThe GPU removes Phase 2 (LIQUID) because:")
    print("1. The equilibrium matrix solution gives Δφ₂ = -8.753686")
    print("2. With old amount = 0.0536594, this would give new = 0.0536594 - 8.753686 < 0")
    print("3. Step size limiting reduces this to 6.13e-3, giving:")
    print("   actual_change = -8.753686 * 0.00613 = -0.0536594")
    print("   new = 0.0536594 - 0.0536594 ≈ 0")
    print("\n4. The large negative delta for phase 2 comes from the equilibrium solution")
    print("   where the system amount constraint (Row 5) has coefficient 20.0 for ALCU_ZETA")
    print("\n5. This large coefficient (20x larger than LIQUID phases) affects the linear system")
    print("   solution, making the solver favor removing small LIQUID phases")

def main():
    """Main analysis."""
    print("Tracing GPU/CPU divergence in Al-Cu-Fe system with ALCU_ZETA phase\n")
    
    analyze_gpu_iteration_0()
    analyze_cpu_iteration_0()
    trace_divergence_cause()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The divergence occurs at iteration 0 when:")
    print("- GPU calculates large negative Δφ₂ = -8.75 for the second LIQUID phase")
    print("- This removes the phase (sets amount to ~0)")
    print("- CPU likely handles the large coefficient (20.0) differently")
    print("- The issue is numerical, related to solving systems with very different scales")

if __name__ == "__main__":
    main()
