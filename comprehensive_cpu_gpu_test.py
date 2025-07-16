#!/usr/bin/env python
"""Comprehensive test of CPU vs GPU calculations across many conditions."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np
import itertools
import sys

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Define test ranges
temperatures = [300, 500, 700, 900, 1000, 1200, 1500, 1800, 2000, 2500, 3000]
ti_fractions = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
pressures = [101325]  # Standard pressure

# Store results
results = []
divergences = []

print("="*80)
print("COMPREHENSIVE CPU vs GPU EQUILIBRIUM TEST")
print("="*80)
print(f"Testing {len(temperatures)} temperatures × {len(ti_fractions)} compositions = {len(temperatures)*len(ti_fractions)} conditions")
print("="*80)

# Suppress verbose output during bulk testing
import io
import warnings
warnings.filterwarnings('ignore')

total_conditions = len(temperatures) * len(ti_fractions) * len(pressures)
condition_count = 0

for T, X_TI, P in itertools.product(temperatures, ti_fractions, pressures):
    condition_count += 1
    
    conditions = {
        'T': T,
        'P': P,
        'X(TI)': X_TI
    }
    
    # Progress indicator
    if condition_count % 10 == 0:
        print(f"Progress: {condition_count}/{total_conditions} conditions tested...", end='\r', flush=True)
    
    try:
        # Run CPU calculation
        cpu_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = cpu_output
        reset_debug_session()
        eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
        sys.stdout = old_stdout
        
        # Run GPU calculation
        gpu_output = io.StringIO()
        sys.stdout = gpu_output
        reset_debug_session()
        eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
        sys.stdout = old_stdout
        
        # Extract results
        cpu_gm = eq_cpu.GM.values.item()
        gpu_gm = eq_gpu.GM.values.item()
        
        # Calculate difference
        abs_diff = abs(cpu_gm - gpu_gm)
        rel_diff = abs_diff / abs(cpu_gm) if cpu_gm != 0 else float('inf')
        
        # Check phase amounts
        cpu_np = eq_cpu.NP.values.flatten()
        gpu_np = eq_gpu.NP.values.flatten()
        np_diff = np.max(np.abs(cpu_np - gpu_np))
        
        # Check which phases are stable
        cpu_stable_phases = set(eq_cpu.Phase.values.flatten()[cpu_np > 1e-6])
        gpu_stable_phases = set(eq_gpu.Phase.values.flatten()[gpu_np > 1e-6])
        phases_match = cpu_stable_phases == gpu_stable_phases
        
        result = {
            'T': T,
            'P': P,
            'X_TI': X_TI,
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'gm_abs_diff': abs_diff,
            'gm_rel_diff': rel_diff,
            'np_max_diff': np_diff,
            'cpu_phases': cpu_stable_phases,
            'gpu_phases': gpu_stable_phases,
            'phases_match': phases_match,
            'converged': True
        }
        
        results.append(result)
        
        # Flag divergences (using 1e-6 threshold as specified)
        if abs_diff > 1e-6 or np_diff > 1e-6 or not phases_match:
            divergences.append(result)
            
    except Exception as e:
        result = {
            'T': T,
            'P': P,
            'X_TI': X_TI,
            'error': str(e),
            'converged': False
        }
        results.append(result)
        divergences.append(result)

print("\n\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Total conditions tested: {len(results)}")
print(f"Successful calculations: {sum(1 for r in results if r.get('converged', False))}")
print(f"Failed calculations: {sum(1 for r in results if not r.get('converged', False))}")
print(f"Divergences found: {len(divergences)}")

if divergences:
    print("\n" + "="*80)
    print("DIVERGENCES DETAIL (|error| > 1e-6)")
    print("="*80)
    
    # Sort by relative difference
    sorted_divergences = sorted([d for d in divergences if d.get('converged', False)], 
                               key=lambda x: x.get('gm_rel_diff', 0), reverse=True)
    
    print(f"\n{'T':>6} {'X(TI)':>6} | {'CPU GM':>15} {'GPU GM':>15} | {'Abs Diff':>12} {'Rel Diff %':>10} | {'Phase Diff':>10} | Phases")
    print("-"*100)
    
    for div in sorted_divergences[:20]:  # Show top 20 divergences
        if div.get('converged', False):
            print(f"{div['T']:>6.0f} {div['X_TI']:>6.2f} | {div['cpu_gm']:>15.6f} {div['gpu_gm']:>15.6f} | "
                  f"{div['gm_abs_diff']:>12.6e} {div['gm_rel_diff']*100:>9.4f}% | "
                  f"{div['np_max_diff']:>10.6e} | "
                  f"{'DIFF' if not div['phases_match'] else 'same'}")
            if not div['phases_match']:
                print(f"    CPU phases: {div['cpu_phases']}")
                print(f"    GPU phases: {div['gpu_phases']}")
    
    # Show any errors
    errors = [d for d in divergences if not d.get('converged', False)]
    if errors:
        print("\nERRORS:")
        for err in errors[:10]:
            print(f"  T={err['T']}, X(TI)={err['X_TI']}: {err.get('error', 'Unknown error')}")

# Statistical analysis
if results:
    converged_results = [r for r in results if r.get('converged', False)]
    if converged_results:
        all_diffs = [r['gm_abs_diff'] for r in converged_results]
        all_rel_diffs = [r['gm_rel_diff'] for r in converged_results if r['gm_rel_diff'] != float('inf')]
        
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY")
        print("="*80)
        print(f"GM Absolute Differences:")
        print(f"  Mean: {np.mean(all_diffs):.6e}")
        print(f"  Median: {np.median(all_diffs):.6e}")
        print(f"  Max: {np.max(all_diffs):.6e}")
        print(f"  Min: {np.min(all_diffs):.6e}")
        print(f"  Std Dev: {np.std(all_diffs):.6e}")
        
        if all_rel_diffs:
            print(f"\nGM Relative Differences (%):")
            print(f"  Mean: {np.mean(all_rel_diffs)*100:.6f}%")
            print(f"  Median: {np.median(all_rel_diffs)*100:.6f}%")
            print(f"  Max: {np.max(all_rel_diffs)*100:.6f}%")
            
        # Count how many exceed threshold
        exceed_1e6 = sum(1 for d in all_diffs if d > 1e-6)
        exceed_1e9 = sum(1 for d in all_diffs if d > 1e-9)
        exceed_1e12 = sum(1 for d in all_diffs if d > 1e-12)
        
        print(f"\nConditions exceeding error thresholds:")
        print(f"  > 1e-6: {exceed_1e6} ({exceed_1e6/len(all_diffs)*100:.1f}%)")
        print(f"  > 1e-9: {exceed_1e9} ({exceed_1e9/len(all_diffs)*100:.1f}%)")
        print(f"  > 1e-12: {exceed_1e12} ({exceed_1e12/len(all_diffs)*100:.1f}%)")

# Save detailed results for analysis
print("\n" + "="*80)
print("Saving detailed results to 'cpu_gpu_comparison_results.txt'...")

with open('cpu_gpu_comparison_results.txt', 'w') as f:
    f.write("T,P,X_TI,CPU_GM,GPU_GM,ABS_DIFF,REL_DIFF,NP_DIFF,PHASES_MATCH,CPU_PHASES,GPU_PHASES\n")
    for r in results:
        if r.get('converged', False):
            f.write(f"{r['T']},{r['P']},{r['X_TI']},{r['cpu_gm']},{r['gpu_gm']},"
                   f"{r['gm_abs_diff']},{r['gm_rel_diff']},{r['np_max_diff']},"
                   f"{r['phases_match']},{r['cpu_phases']},{r['gpu_phases']}\n")

print("Done!")