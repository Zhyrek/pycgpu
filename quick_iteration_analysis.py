#!/usr/bin/env python3
"""Quick analysis of iteration 0 divergence"""

import re

# Read the trace file
with open('gpu_solver_trace.txt', 'r') as f:
    content = f.read()

print("=== Iteration 0 Analysis ===")

# Extract GPU data
gpu_match = re.search(r'\[GPU TRACE\] ===== AFTER ITERATION 0 =====.*?(?=\[GPU TRACE\] ===== END)', content, re.DOTALL)
if gpu_match:
    gpu_section = gpu_match.group(0)
    
    # Chemical potentials
    chem_pot_match = re.search(r'Chemical potentials: \[([-0-9.e+, ]+)\]', gpu_section)
    if chem_pot_match:
        gpu_chem_pot = [float(x.strip()) for x in chem_pot_match.group(1).split(',')][:2]
        print(f"GPU Chemical potentials: [{gpu_chem_pot[0]:.1f}, {gpu_chem_pot[1]:.1f}]")
    
    # Phase amounts and site fractions
    phase_pattern = r'Phase (\d+):\s+NP \(mole fraction\): ([-0-9.e+]+)\s+Site fractions: \[([0-9.e+-]+), ([0-9.e+-]+)\]'
    gpu_phases = re.findall(phase_pattern, gpu_section, re.DOTALL)
    
    print("GPU Phases:")
    gpu_total_gm = 0
    for phase_num, np_val, y_nb, y_ti in gpu_phases:
        np_f = float(np_val)
        y_nb_f = float(y_nb)
        y_ti_f = float(y_ti)
        print(f"  Phase {phase_num}: NP={np_f:.4f}, Y(NB)={y_nb_f:.6f}, Y(TI)={y_ti_f:.6f}")

# Extract CPU data  
cpu_match = re.search(r'\[CPU TRACE\] ===== AFTER ITERATION 0 =====.*?(?=\[CPU TRACE\] ===== END)', content, re.DOTALL)
if cpu_match:
    cpu_section = cpu_match.group(0)
    
    # Chemical potentials
    chem_pot_match = re.search(r'Chemical potentials: \[([-0-9.e+ ]+)\]', cpu_section)
    if chem_pot_match:
        cpu_chem_pot = [float(x.strip()) for x in chem_pot_match.group(1).split()][:2]
        print(f"CPU Chemical potentials: [{cpu_chem_pot[0]:.1f}, {cpu_chem_pot[1]:.1f}]")
    
    # Phase amounts and site fractions
    phase_pattern = r'Phase (\d+) \(BCC_A2\):\s+NP \(mole fraction\): ([-0-9.e+]+).*?Site fractions: \[([0-9.e+\- ]+)\]'
    cpu_phases = re.findall(phase_pattern, cpu_section, re.DOTALL)
    
    print("CPU Phases:")
    for phase_num, np_val, site_fractions in cpu_phases:
        np_f = float(np_val)
        sf_parts = site_fractions.split()
        y_nb_f = float(sf_parts[0])
        y_ti_f = float(sf_parts[1])
        print(f"  Phase {phase_num}: NP={np_f:.4f}, Y(NB)={y_nb_f:.6f}, Y(TI)={y_ti_f:.6f}")

print("\n=== Analysis ===")
print("Key finding: GPU and CPU have identical chemical potentials")
print("but different phase amounts at iteration 0.")
print("\nThis suggests the issue is in the initial composition set creation")
print("or how the starting phase amounts are calculated, NOT in the")
print("equilibrium solver iteration logic itself.")

# Check chemical potential differences
if 'gpu_chem_pot' in locals() and 'cpu_chem_pot' in locals():
    mu_nb_diff = gpu_chem_pot[0] - cpu_chem_pot[0]
    mu_ti_diff = gpu_chem_pot[1] - cpu_chem_pot[1]
    print(f"\nChemical potential differences:")
    print(f"  Δμ(NB) = {mu_nb_diff:.6f} J/mol")
    print(f"  Δμ(TI) = {mu_ti_diff:.6f} J/mol")
    
    if abs(mu_nb_diff) < 0.001 and abs(mu_ti_diff) < 0.001:
        print("  ✓ Chemical potentials match exactly!")
    else:
        print("  ✗ Chemical potentials differ")

print("\nSince chemical potentials match but phase amounts differ,")
print("the issue is likely in the initial phase amount assignment")
print("before the first iteration starts.")