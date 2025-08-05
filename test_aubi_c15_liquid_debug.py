#!/usr/bin/env python
"""Debug specific conditions for AU2BI_C15 + LIQUID phases in Au-Bi system."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

def test_single_condition(dbf, comps, phases, x_bi, temp):
    """Test a single condition and show detailed results."""
    
    print(f"\n{'='*60}")
    print(f"Testing X(BI)={x_bi}, T={temp}K")
    print(f"{'='*60}")
    
    conditions = {
        v.X('BI'): x_bi,
        v.T: temp,
        v.P: 101325
    }
    
    # CPU calculation with verbose output
    print("\nCPU Calculation:")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    
    # Extract CPU results
    cpu_gm = float(result_cpu.GM.values)
    cpu_phases = []
    cpu_amounts = []
    cpu_compositions = []
    
    for i in range(result_cpu.Phase.shape[-1]):
        phase = str(result_cpu.Phase.values.flat[i])
        amount = float(result_cpu.NP.values.flat[i])
        if phase and amount > 1e-6:
            cpu_phases.append(phase)
            cpu_amounts.append(amount)
            # Get composition of this phase
            x_au = float(result_cpu.X.sel(component='AU').values.flat[i])
            x_bi = float(result_cpu.X.sel(component='BI').values.flat[i])
            cpu_compositions.append((x_au, x_bi))
    
    print(f"  GM = {cpu_gm:.6f} J/mol")
    print(f"  Active phases:")
    for phase, amount, (x_au, x_bi) in zip(cpu_phases, cpu_amounts, cpu_compositions):
        print(f"    {phase}: NP = {amount:.6f}, X(AU) = {x_au:.6f}, X(BI) = {x_bi:.6f}")
    
    # GPU calculation
    print("\nGPU Calculation:")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    # Extract GPU results
    gpu_gm = float(result_gpu.GM.values)
    gpu_phases = []
    gpu_amounts = []
    gpu_compositions = []
    
    for i in range(result_gpu.Phase.shape[-1]):
        phase = str(result_gpu.Phase.values.flat[i])
        amount = float(result_gpu.NP.values.flat[i])
        if phase and amount > 1e-6:
            gpu_phases.append(phase)
            gpu_amounts.append(amount)
            # Get composition of this phase
            x_au = float(result_gpu.X.sel(component='AU').values.flat[i])
            x_bi = float(result_gpu.X.sel(component='BI').values.flat[i])
            gpu_compositions.append((x_au, x_bi))
    
    print(f"  GM = {gpu_gm:.6f} J/mol")
    print(f"  Active phases:")
    for phase, amount, (x_au, x_bi) in zip(gpu_phases, gpu_amounts, gpu_compositions):
        print(f"    {phase}: NP = {amount:.6f}, X(AU) = {x_au:.6f}, X(BI) = {x_bi:.6f}")
    
    # Comparison
    print("\nComparison:")
    gm_diff = abs(gpu_gm - cpu_gm)
    print(f"  GM difference: {gm_diff:.6f} J/mol")
    
    # Check if same phases
    cpu_phase_set = set(cpu_phases)
    gpu_phase_set = set(gpu_phases)
    if cpu_phase_set == gpu_phase_set:
        print(f"  ✓ Same phases found: {cpu_phase_set}")
    else:
        print(f"  ✗ Different phases!")
        print(f"    CPU: {cpu_phase_set}")
        print(f"    GPU: {gpu_phase_set}")
    
    return gm_diff < 1.0  # 1 J/mol tolerance

def main():
    """Run debug tests for specific conditions."""
    
    # Load database and set up calculation
    dbf = Database('AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'LIQUID']
    
    print("Testing Au-Bi system with AU2BI_C15 + LIQUID phases")
    print("AU2BI_C15 has vacancy sublattice: (AU)(AU,BI,VA)(BI)")
    
    # Test specific problematic conditions
    test_conditions = [
        (0.3, 600),  # Mid-range composition and temperature
    ]
    
    passed = 0
    for x_bi, temp in test_conditions:
        if test_single_condition(dbf, comps, phases, x_bi, temp):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{len(test_conditions)} conditions passed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()