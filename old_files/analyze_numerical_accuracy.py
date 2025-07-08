#!/usr/bin/env python3
"""
Detailed numerical accuracy analysis between CPU and GPU pycalphad calculations
Focus on identifying the source of the ~1000 J/mol GM difference
"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
import time
import os
import glob

def clear_cupy_kernel_cache():
    """Clear CuPy kernel cache to ensure fresh compilation"""
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        if cubin_files:
            print(f"[CACHE] Clearing {len(cubin_files)} .cubin files from CuPy kernel cache...")
            for cubin_file in cubin_files:
                try:
                    os.remove(cubin_file)
                except OSError as e:
                    print(f"[CACHE] Warning: Could not remove {cubin_file}: {e}")
        else:
            print("[CACHE] No .cubin files found in CuPy kernel cache.")
    else:
        print("[CACHE] CuPy kernel cache directory not found.")

def analyze_single_point_accuracy():
    """Detailed analysis of a single thermodynamic point"""
    print("=== SINGLE POINT NUMERICAL ACCURACY ANALYSIS ===")
    
    # Clear cache for fresh compilation
    clear_cupy_kernel_cache()
    
    # Load database and set up test case
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]  # Single phase for simplicity
    comps = ["NB", "TI", "VA"]
    
    # Simple conditions - same as previous tests
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    print(f"Analyzing: {conditions}")
    print(f"Phases: {phases}")
    print(f"Components: {comps}")
    
    # CPU calculation
    print("\n" + "="*50)
    print("CPU CALCULATION")
    print("="*50)
    
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    
    # Extract key CPU values
    cpu_gm = float(cpu_result.GM.values.flatten()[0])
    cpu_mu = cpu_result.MU.values.flatten()[:2]  # Only NB, TI components
    cpu_np = cpu_result.NP.values.flatten()[:2]  # Active phases
    cpu_x = cpu_result.X.values.reshape(-1, 2)[:2]  # Compositions for active phases
    cpu_phase = cpu_result.Phase.values.flatten()[:3]  # Phase names
    
    print(f"CPU GM: {cpu_gm:.6f} J/mol")
    print(f"CPU MU: {cpu_mu}")
    print(f"CPU NP: {cpu_np}")
    print(f"CPU X: {cpu_x}")
    print(f"CPU Phases: {cpu_phase}")
    
    # GPU calculation
    print("\n" + "="*50)
    print("GPU CALCULATION")
    print("="*50)
    
    gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    
    # Extract key GPU values
    gpu_gm = float(gpu_result.GM.values.flatten()[0])
    gpu_mu = gpu_result.MU.values.flatten()[:2]  # Only NB, TI components  
    gpu_np = gpu_result.NP.values.flatten()[:2]  # Active phases
    gpu_x = gpu_result.X.values.reshape(-1, 3)[:2, :2]  # Compositions, limit to NB,TI
    gpu_phase = gpu_result.Phase.values.flatten()[:3]  # Phase names
    
    print(f"GPU GM: {gpu_gm:.6f} J/mol")
    print(f"GPU MU: {gpu_mu}")
    print(f"GPU NP: {gpu_np}")
    print(f"GPU X: {gpu_x}")
    print(f"GPU Phases: {gpu_phase}")
    
    # Detailed comparison
    print("\n" + "="*50)
    print("NUMERICAL ACCURACY ANALYSIS")
    print("="*50)
    
    gm_diff = abs(cpu_gm - gpu_gm)
    gm_rel_diff = gm_diff / abs(cpu_gm) * 100
    
    print(f"GM Difference: {gm_diff:.6f} J/mol ({gm_rel_diff:.3f}%)")
    
    if gm_diff > 1.0:  # More than 1 J/mol difference is significant
        print(f"⚠️  SIGNIFICANT GM DIFFERENCE DETECTED!")
        print(f"   This suggests a potential issue in:")
        print(f"   - Phase composition calculation")
        print(f"   - Chemical potential calculation") 
        print(f"   - Energy minimization convergence")
        print(f"   - Numerical precision in GPU kernel")
    
    # Check chemical potentials
    mu_diff = np.abs(cpu_mu - gpu_mu)
    max_mu_diff = np.max(mu_diff)
    print(f"\nChemical Potential Differences:")
    print(f"  MU_NB: {mu_diff[0]:.6f} J/mol")
    print(f"  MU_TI: {mu_diff[1]:.6f} J/mol")
    print(f"  Max MU diff: {max_mu_diff:.6f} J/mol")
    
    if max_mu_diff > 10.0:  # More than 10 J/mol difference in MU is concerning
        print(f"⚠️  SIGNIFICANT CHEMICAL POTENTIAL DIFFERENCE!")
    
    # Check phase amounts
    np_diff = np.abs(cpu_np - gpu_np)
    max_np_diff = np.max(np_diff)
    print(f"\nPhase Amount Differences:")
    for i, diff in enumerate(np_diff):
        if i < len(cpu_phase) and cpu_phase[i] != '':
            print(f"  Phase {i} ({cpu_phase[i]}): {diff:.6f}")
    print(f"  Max NP diff: {max_np_diff:.6f}")
    
    if max_np_diff > 0.001:  # More than 0.1% difference in phase amounts
        print(f"⚠️  SIGNIFICANT PHASE AMOUNT DIFFERENCE!")
    
    # Check compositions
    x_diff = np.abs(cpu_x - gpu_x)
    max_x_diff = np.max(x_diff)
    print(f"\nComposition Differences:")
    print(f"  Phase compositions diff:\n{x_diff}")
    print(f"  Max X diff: {max_x_diff:.6f}")
    
    if max_x_diff > 0.001:  # More than 0.1% difference in composition
        print(f"⚠️  SIGNIFICANT COMPOSITION DIFFERENCE!")
    
    return {
        'gm_diff': gm_diff,
        'mu_diff': mu_diff,
        'np_diff': np_diff,
        'x_diff': x_diff,
        'cpu_gm': cpu_gm,
        'gpu_gm': gpu_gm
    }

def analyze_temperature_series():
    """Test accuracy across a range of temperatures"""
    print("\n" + "="*60)
    print("TEMPERATURE SERIES ACCURACY ANALYSIS")
    print("="*60)
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    # Temperature range
    temperatures = [600, 700, 800, 900, 1000]
    x_ti = 0.1  # Fixed composition
    
    results = []
    
    for T in temperatures:
        print(f"\n--- Temperature: {T} K ---")
        
        conditions = {
            v.X("TI"): x_ti,
            v.T: T,
            v.P: 101325
        }
        
        # CPU
        cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = float(cpu_result.GM.values.flatten()[0])
        
        # GPU  
        gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = float(gpu_result.GM.values.flatten()[0])
        
        gm_diff = abs(cpu_gm - gpu_gm)
        gm_rel_diff = gm_diff / abs(cpu_gm) * 100
        
        print(f"  CPU GM: {cpu_gm:.2f} J/mol")
        print(f"  GPU GM: {gpu_gm:.2f} J/mol")
        print(f"  Difference: {gm_diff:.2f} J/mol ({gm_rel_diff:.3f}%)")
        
        results.append({
            'T': T,
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'gm_diff': gm_diff,
            'gm_rel_diff': gm_rel_diff
        })
    
    # Summary
    print(f"\n--- TEMPERATURE SERIES SUMMARY ---")
    avg_diff = np.mean([r['gm_diff'] for r in results])
    max_diff = np.max([r['gm_diff'] for r in results])
    max_rel_diff = np.max([r['gm_rel_diff'] for r in results])
    
    print(f"Average GM difference: {avg_diff:.2f} J/mol")
    print(f"Maximum GM difference: {max_diff:.2f} J/mol")
    print(f"Maximum relative difference: {max_rel_diff:.3f}%")
    
    if max_diff > 100:
        print("⚠️  Large differences detected across temperature range!")
    
    return results

def analyze_composition_series():
    """Test accuracy across a range of compositions"""
    print("\n" + "="*60)
    print("COMPOSITION SERIES ACCURACY ANALYSIS")  
    print("="*60)
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    # Composition range
    x_ti_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    T = 800  # Fixed temperature
    
    results = []
    
    for x_ti in x_ti_values:
        print(f"\n--- X_TI: {x_ti} ---")
        
        conditions = {
            v.X("TI"): x_ti,
            v.T: T,
            v.P: 101325
        }
        
        # CPU
        cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = float(cpu_result.GM.values.flatten()[0])
        
        # GPU
        gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = float(gpu_result.GM.values.flatten()[0])
        
        gm_diff = abs(cpu_gm - gpu_gm)
        gm_rel_diff = gm_diff / abs(cpu_gm) * 100
        
        print(f"  CPU GM: {cpu_gm:.2f} J/mol")
        print(f"  GPU GM: {gpu_gm:.2f} J/mol")
        print(f"  Difference: {gm_diff:.2f} J/mol ({gm_rel_diff:.3f}%)")
        
        results.append({
            'x_ti': x_ti,
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'gm_diff': gm_diff,
            'gm_rel_diff': gm_rel_diff
        })
    
    # Summary
    print(f"\n--- COMPOSITION SERIES SUMMARY ---")
    avg_diff = np.mean([r['gm_diff'] for r in results])
    max_diff = np.max([r['gm_diff'] for r in results])
    max_rel_diff = np.max([r['gm_rel_diff'] for r in results])
    
    print(f"Average GM difference: {avg_diff:.2f} J/mol")
    print(f"Maximum GM difference: {max_diff:.2f} J/mol")
    print(f"Maximum relative difference: {max_rel_diff:.3f}%")
    
    if max_diff > 100:
        print("⚠️  Large differences detected across composition range!")
    
    return results

if __name__ == "__main__":
    print("Starting comprehensive numerical accuracy analysis...")
    
    # Single point detailed analysis
    single_point_results = analyze_single_point_accuracy()
    
    # Temperature series analysis
    temp_series_results = analyze_temperature_series()
    
    # Composition series analysis  
    comp_series_results = analyze_composition_series()
    
    # Overall summary
    print("\n" + "="*60)
    print("OVERALL ACCURACY ASSESSMENT")
    print("="*60)
    
    max_single_gm_diff = single_point_results['gm_diff']
    max_temp_gm_diff = max([r['gm_diff'] for r in temp_series_results])
    max_comp_gm_diff = max([r['gm_diff'] for r in comp_series_results])
    
    overall_max_diff = max(max_single_gm_diff, max_temp_gm_diff, max_comp_gm_diff)
    
    print(f"Maximum GM difference observed: {overall_max_diff:.2f} J/mol")
    
    if overall_max_diff < 1.0:
        print("✅ EXCELLENT: Differences < 1 J/mol")
    elif overall_max_diff < 10.0:
        print("✅ GOOD: Differences < 10 J/mol")
    elif overall_max_diff < 100.0:
        print("⚠️  MODERATE: Differences 10-100 J/mol - investigation recommended")
    else:
        print("❌ POOR: Differences > 100 J/mol - significant issues detected")
        print("\nRecommended investigation areas:")
        print("1. GPU kernel numerical precision")
        print("2. Convergence criteria differences")
        print("3. Initial guess differences") 
        print("4. Energy calculation implementation")
        print("5. Chemical potential calculation")