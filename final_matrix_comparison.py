#!/usr/bin/env python
"""Final comparison of CPU vs GPU after all fixes."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

def final_comparison():
    """Final comparison to check if CPU and GPU are now identical."""
    
    print("FINAL CPU vs GPU COMPARISON AFTER ALL FIXES")
    print("="*50)
    
    tdb = Database('Al-Cu-Fe.tdb')
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    print("Test conditions:")
    print("  Al-Cu-Fe ternary system")
    print("  T = 1200 K")
    print("  X(CU) = 0.3")
    print("  X(FE) = 0.2")
    print("  X(AL) = 0.5 (balance)")
    
    print("\n" + "-"*50)
    
    # CPU calculation
    print("CPU Equilibrium:")
    eq_cpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=False)
    cpu_gm = float(eq_cpu.GM.values.flatten()[0])
    
    print(f"  GM = {cpu_gm:.8f} J/mol")
    
    # Also get phase fractions from CPU
    phase_fractions_cpu = eq_cpu.Phase.values.flatten()
    print(f"  Phase fractions: {phase_fractions_cpu}")
    
    print("\n" + "-"*50)
    
    # GPU calculation
    print("GPU Equilibrium:")
    eq_gpu = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], conditions, gpu=True)
    gpu_gm = float(eq_gpu.GM.values.flatten()[0])
    
    print(f"  GM = {gpu_gm:.8f} J/mol")
    
    # Also get phase fractions from GPU
    phase_fractions_gpu = eq_gpu.Phase.values.flatten()
    print(f"  Phase fractions: {phase_fractions_gpu}")
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS:")
    print("="*50)
    
    # Calculate differences
    gm_diff = abs(gpu_gm - cpu_gm)
    relative_error = gm_diff / abs(cpu_gm) * 100
    
    print(f"Absolute GM difference: {gm_diff:.8f} J/mol")
    print(f"Relative error: {relative_error:.6f} %")
    
    # Assessment
    if gm_diff < 1e-10:
        print("✓ PERFECT: Numerical precision match (< 1e-10 J/mol)")
        status = "PERFECT"
    elif gm_diff < 1e-6:
        print("✓ EXCELLENT: Essentially identical (< 1 µJ/mol)")
        status = "EXCELLENT"  
    elif gm_diff < 1e-3:
        print("✓ VERY GOOD: Submillijoule precision (< 1 mJ/mol)")
        status = "VERY_GOOD"
    elif gm_diff < 1.0:
        print("✓ GOOD: Within 1 J/mol")
        status = "GOOD"
    elif gm_diff < 10.0:
        print("○ ACCEPTABLE: Within 10 J/mol")
        status = "ACCEPTABLE"
    elif gm_diff < 100.0:
        print("△ NEEDS WORK: Significant difference (< 100 J/mol)")
        status = "NEEDS_WORK"
    else:
        print("✗ PROBLEM: Large difference (> 100 J/mol)")
        status = "PROBLEM"
    
    print(f"\nProgress assessment:")
    print(f"  Original error: 239 J/mol")
    print(f"  Current error:  {gm_diff:.1f} J/mol")
    if gm_diff < 239:
        improvement = (239 - gm_diff) / 239 * 100
        print(f"  Improvement: {improvement:.1f}% reduction")
    
    # Phase fraction comparison
    print(f"\nPhase fraction comparison:")
    phase_diff = np.abs(phase_fractions_cpu - phase_fractions_gpu)
    max_phase_diff = np.max(phase_diff)
    print(f"  Max phase fraction difference: {max_phase_diff:.6f}")
    
    if max_phase_diff < 1e-6:
        print("  ✓ Phase fractions match to numerical precision")
    elif max_phase_diff < 1e-3:
        print("  ✓ Phase fractions match very well")
    else:
        print("  △ Phase fractions have some differences")
    
    print(f"\n" + "="*50)
    print("MATRIX EQUIVALENCE ASSESSMENT:")
    print("="*50)
    
    if status in ["PERFECT", "EXCELLENT"]:
        print("✓ CPU and GPU equilibrium matrices are now EFFECTIVELY IDENTICAL")
        print("  The gradient mapping fixes have successfully resolved the issues.")
        print("  Any remaining differences are within numerical precision.")
        
    elif status in ["VERY_GOOD", "GOOD"]:
        print("✓ CPU and GPU equilibrium matrices are now VERY SIMILAR")
        print("  The gradient mapping fixes have largely resolved the issues.")
        print("  Remaining small differences could be due to:")
        print("    - Numerical precision differences in matrix operations")
        print("    - Slight differences in convergence criteria")
        print("    - Minor implementation differences in solver algorithms")
        
    elif status == "ACCEPTABLE":
        print("○ CPU and GPU matrices are much closer than before")
        print("  Major progress achieved, but some differences remain:")
        print("    - Possible differences in constraint formulation")
        print("    - Solver tolerance or iteration differences")
        print("    - Numerical precision accumulation")
        
    else:
        print("△ Significant matrix differences still exist")
        print("  Further investigation needed into:")
        print("    - Matrix construction algorithms")
        print("    - Constraint handling differences") 
        print("    - Solver implementation differences")
    
    return gm_diff, status

if __name__ == "__main__":
    final_comparison()