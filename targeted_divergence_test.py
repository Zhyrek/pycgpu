#!/usr/bin/env python
"""Targeted test to find CPU vs GPU divergences in specific scenarios."""

from pycalphad import Database, equilibrium
from pycalphad.core.debug_output import reset_debug_session, close_debug_output
import numpy as np
import sys
import io
import warnings

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test specific scenarios that might cause divergence:
# 1. Near phase boundaries
# 2. Extreme compositions
# 3. Single-phase regions
# 4. Two-phase regions
# 5. Near critical temperatures

test_conditions = [
    # Extreme compositions
    {'T': 1000, 'P': 101325, 'X(TI)': 0.001, 'description': 'Very low Ti'},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.999, 'description': 'Very high Ti'},
    
    # Near melting points
    {'T': 1944, 'P': 101325, 'X(TI)': 0.5, 'description': 'Near Ti melting point'},
    {'T': 2750, 'P': 101325, 'X(TI)': 0.5, 'description': 'Near Nb melting point'},
    
    # Low temperatures
    {'T': 300, 'P': 101325, 'X(TI)': 0.3, 'description': 'Low T, 30% Ti'},
    {'T': 300, 'P': 101325, 'X(TI)': 0.7, 'description': 'Low T, 70% Ti'},
    
    # Phase boundary search (typical two-phase region)
    {'T': 1200, 'P': 101325, 'X(TI)': 0.2, 'description': 'Likely two-phase'},
    {'T': 1200, 'P': 101325, 'X(TI)': 0.8, 'description': 'Likely two-phase'},
    
    # Mid-range temperatures with various compositions
    {'T': 1500, 'P': 101325, 'X(TI)': 0.1, 'description': 'Mid T, low Ti'},
    {'T': 1500, 'P': 101325, 'X(TI)': 0.5, 'description': 'Mid T, equiatomic'},
    {'T': 1500, 'P': 101325, 'X(TI)': 0.9, 'description': 'Mid T, high Ti'},
    
    # High temperature (liquid region)
    {'T': 3000, 'P': 101325, 'X(TI)': 0.3, 'description': 'High T liquid'},
    {'T': 3000, 'P': 101325, 'X(TI)': 0.7, 'description': 'High T liquid'},
]

print("="*80)
print("TARGETED CPU vs GPU DIVERGENCE TEST")
print("="*80)
print(f"Testing {len(test_conditions)} specific conditions")
print("="*80)

warnings.filterwarnings('ignore')
results = []
divergences = []

for i, cond_dict in enumerate(test_conditions):
    conditions = {k: v for k, v in cond_dict.items() if k not in ['description']}
    description = cond_dict.get('description', '')
    
    print(f"\nTest {i+1}/{len(test_conditions)}: T={conditions['T']}K, X(TI)={conditions['X(TI)']} - {description}")
    
    try:
        # Run CPU calculation with verbose to capture any issues
        cpu_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = cpu_output
        reset_debug_session()
        eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=True)
        sys.stdout = old_stdout
        cpu_text = cpu_output.getvalue()
        
        # Run GPU calculation
        gpu_output = io.StringIO()
        sys.stdout = gpu_output
        reset_debug_session()
        eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
        sys.stdout = old_stdout
        gpu_text = gpu_output.getvalue()
        
        # Extract results
        cpu_gm = eq_cpu.GM.values.item()
        gpu_gm = eq_gpu.GM.values.item()
        
        # Calculate differences
        abs_diff = abs(cpu_gm - gpu_gm)
        rel_diff = abs_diff / abs(cpu_gm) if cpu_gm != 0 else float('inf')
        
        # Check phase amounts
        cpu_np = eq_cpu.NP.values.flatten()
        gpu_np = eq_gpu.NP.values.flatten()
        np_diff = np.max(np.abs(cpu_np - gpu_np))
        
        # Check site fractions if available
        y_diff = 0
        if hasattr(eq_cpu, 'Y') and hasattr(eq_gpu, 'Y'):
            cpu_y = eq_cpu.Y.values.flatten()
            gpu_y = eq_gpu.Y.values.flatten()
            # Only compare non-NaN values
            valid_mask = ~(np.isnan(cpu_y) | np.isnan(gpu_y))
            if np.any(valid_mask):
                y_diff = np.max(np.abs(cpu_y[valid_mask] - gpu_y[valid_mask]))
        
        # Check stable phases
        cpu_stable_phases = list(eq_cpu.Phase.values.flatten()[cpu_np > 1e-6])
        gpu_stable_phases = list(eq_gpu.Phase.values.flatten()[gpu_np > 1e-6])
        phases_match = set(cpu_stable_phases) == set(gpu_stable_phases)
        
        # Count iterations (if available in output)
        import re
        cpu_iter_match = re.search(r"Converged in (\d+) iterations", cpu_text)
        gpu_iter_match = re.search(r"iteration (\d+)", gpu_text)
        cpu_iterations = int(cpu_iter_match.group(1)) if cpu_iter_match else None
        gpu_iterations = int(gpu_iter_match.group(1)) if gpu_iter_match else None
        
        print(f"  CPU: GM={cpu_gm:.6f}, phases={cpu_stable_phases}")
        print(f"  GPU: GM={gpu_gm:.6f}, phases={gpu_stable_phases}")
        print(f"  |ΔGM|={abs_diff:.6e}, |ΔNP|={np_diff:.6e}, |ΔY|={y_diff:.6e}")
        
        if abs_diff > 1e-6:
            print(f"  *** GM DIVERGENCE: {abs_diff:.6e} > 1e-6 ***")
        if np_diff > 1e-6:
            print(f"  *** NP DIVERGENCE: {np_diff:.6e} > 1e-6 ***")
        if y_diff > 1e-6:
            print(f"  *** Y DIVERGENCE: {y_diff:.6e} > 1e-6 ***")
        if not phases_match:
            print(f"  *** PHASE MISMATCH: CPU={cpu_stable_phases}, GPU={gpu_stable_phases} ***")
            
        result = {
            'conditions': conditions,
            'description': description,
            'cpu_gm': cpu_gm,
            'gpu_gm': gpu_gm,
            'gm_abs_diff': abs_diff,
            'gm_rel_diff': rel_diff,
            'np_max_diff': np_diff,
            'y_max_diff': y_diff,
            'cpu_phases': cpu_stable_phases,
            'gpu_phases': gpu_stable_phases,
            'phases_match': phases_match,
            'cpu_iterations': cpu_iterations,
            'gpu_iterations': gpu_iterations,
            'converged': True
        }
        
        results.append(result)
        
        # Flag divergences
        if abs_diff > 1e-6 or np_diff > 1e-6 or y_diff > 1e-6 or not phases_match:
            divergences.append(result)
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        result = {
            'conditions': conditions,
            'description': description,
            'error': str(e),
            'converged': False
        }
        results.append(result)
        divergences.append(result)

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total tests: {len(results)}")
print(f"Successful: {sum(1 for r in results if r.get('converged', False))}")
print(f"Failed: {sum(1 for r in results if not r.get('converged', False))}")
print(f"Divergences: {len([d for d in divergences if d.get('converged', False)])}")

if divergences:
    print("\n" + "="*80)
    print("DIVERGENCE DETAILS")
    print("="*80)
    
    for div in [d for d in divergences if d.get('converged', False)]:
        cond = div['conditions']
        print(f"\nT={cond['T']}K, X(TI)={cond['X(TI)']} - {div['description']}")
        print(f"  GM: CPU={div['cpu_gm']:.6f}, GPU={div['gpu_gm']:.6f}, Δ={div['gm_abs_diff']:.6e}")
        print(f"  NP max diff: {div['np_max_diff']:.6e}")
        print(f"  Y max diff: {div['y_max_diff']:.6e}")
        print(f"  Phases match: {div['phases_match']}")
        if not div['phases_match']:
            print(f"    CPU phases: {div['cpu_phases']}")
            print(f"    GPU phases: {div['gpu_phases']}")

# Check if we need to examine solver behavior
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if not divergences:
    print("No divergences found! CPU and GPU results match within tolerance.")
else:
    print(f"Found {len(divergences)} conditions with divergences.")
    print("\nPossible causes to investigate:")
    
    # Analyze patterns
    extreme_comp_divs = [d for d in divergences if d.get('converged') and 
                         (d['conditions']['X(TI)'] < 0.01 or d['conditions']['X(TI)'] > 0.99)]
    if extreme_comp_divs:
        print(f"- {len(extreme_comp_divs)} divergences at extreme compositions")
    
    phase_mismatch_divs = [d for d in divergences if d.get('converged') and not d['phases_match']]
    if phase_mismatch_divs:
        print(f"- {len(phase_mismatch_divs)} cases with different stable phases")
        
    print("\nNext steps:")
    print("1. Run specific divergent conditions with full debug output")
    print("2. Check phase addition/removal logic differences")
    print("3. Examine convergence criteria differences")
    print("4. Compare matrix solving precision")