#!/usr/bin/env python
"""Analyze LIQUID and other simple phases to find working examples."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import pycalphad.variables as v

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Create workspace
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Analyzing phase complexity for Al-Cu-Fe database...")
print("="*80)

# Analyze all phases
all_stats = []

for phase in db.phases.keys():
    try:
        # Create workspace with single phase
        wks = Workspace(db, components, [phase], conditions, verbose=False)
        
        # Generate code
        result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
        all_device_functions, init_calls, unique_models, phase_map = result
        
        # Analyze the code
        lines = all_device_functions.split('\n')
        
        # Find the longest line
        max_line_length = 0
        longest_line_num = 0
        line_lengths = []
        
        for i, line in enumerate(lines):
            line_lengths.append(len(line))
            if len(line) > max_line_length:
                max_line_length = len(line)
                longest_line_num = i + 1
        
        # Count sublattices
        phase_obj = db.phases[phase]
        num_sublattices = len(phase_obj.sublattices)
        total_species = sum(len(subl) for subl in phase_obj.sublattices)
        
        # Find which function has the longest line
        longest_func = "unknown"
        line_count = 0
        for i, line in enumerate(lines):
            if '__device__' in line:
                # Extract function name
                if 'formulahess' in line:
                    longest_func = "hessian"
                elif 'formulagrad' in line:
                    longest_func = "gradient"
                elif 'formulaobj' in line:
                    longest_func = "formula_obj"
                elif '_obj(' in line and 'formula' not in line:
                    longest_func = "energy"
                elif 'internal_cons_jac' in line:
                    longest_func = "constraint_jac"
                else:
                    longest_func = "other"
                line_count = 0
            line_count += 1
            if i + 1 == longest_line_num:
                longest_func += f" (line {line_count} in func)"
        
        stats = {
            'phase': phase,
            'num_sublattices': num_sublattices,
            'total_species': total_species,
            'total_size': len(all_device_functions),
            'max_line': max_line_length,
            'max_line_num': longest_line_num,
            'longest_func': longest_func,
            'num_lines': len(lines),
            'avg_line': sum(line_lengths) / len(line_lengths) if line_lengths else 0
        }
        all_stats.append(stats)
        
    except Exception as e:
        print(f"Error processing {phase}: {e}")

# Sort by max line length
all_stats.sort(key=lambda x: x['max_line'])

print("\nPhases sorted by maximum line length (shortest to longest):")
print("-"*120)
print(f"{'Phase':<15} {'Sublattices':<12} {'Species':<10} {'Max Line':<12} {'Avg Line':<10} {'Total Size':<12} {'Longest In':<20}")
print("-"*120)

for stats in all_stats:
    print(f"{stats['phase']:<15} {stats['num_sublattices']:<12} {stats['total_species']:<10} "
          f"{stats['max_line']:<12} {stats['avg_line']:<10.0f} {stats['total_size']:<12} {stats['longest_func']:<20}")

# Group by complexity
print("\n" + "="*80)
print("PHASES GROUPED BY COMPLEXITY:")
print("="*80)

simple_phases = [s for s in all_stats if s['max_line'] < 5000]
moderate_phases = [s for s in all_stats if 5000 <= s['max_line'] < 20000]
complex_phases = [s for s in all_stats if s['max_line'] >= 20000]

print(f"\nSimple phases (max line < 5000 chars): {len(simple_phases)}")
for s in simple_phases:
    print(f"  {s['phase']:<15} - {s['num_sublattices']} sublattices, max line: {s['max_line']}")

print(f"\nModerate phases (5000 <= max line < 20000 chars): {len(moderate_phases)}")
for s in moderate_phases:
    print(f"  {s['phase']:<15} - {s['num_sublattices']} sublattices, max line: {s['max_line']}")

print(f"\nComplex phases (max line >= 20000 chars): {len(complex_phases)}")
for s in complex_phases:
    print(f"  {s['phase']:<15} - {s['num_sublattices']} sublattices, max line: {s['max_line']}")

# Analyze correlation with sublattices
print("\n" + "="*80)
print("CORRELATION ANALYSIS:")
print("="*80)

# Group by number of sublattices
by_sublattices = {}
for s in all_stats:
    n = s['num_sublattices']
    if n not in by_sublattices:
        by_sublattices[n] = []
    by_sublattices[n].append(s)

for n_subl in sorted(by_sublattices.keys()):
    phases = by_sublattices[n_subl]
    max_lines = [p['max_line'] for p in phases]
    avg_max = sum(max_lines) / len(max_lines)
    print(f"\n{n_subl} sublattices ({len(phases)} phases):")
    print(f"  Average max line: {avg_max:.0f} chars")
    print(f"  Range: {min(max_lines)} - {max(max_lines)} chars")
    print(f"  Phases: {', '.join(p['phase'] for p in phases)}")

# Find phases similar to LIQUID
liquid_stats = next(s for s in all_stats if s['phase'] == 'LIQUID')
print(f"\n" + "="*80)
print(f"LIQUID PHASE ANALYSIS:")
print(f"="*80)
print(f"  Sublattices: {liquid_stats['num_sublattices']}")
print(f"  Max line: {liquid_stats['max_line']} chars")
print(f"  Total size: {liquid_stats['total_size']} chars")
print(f"  Longest line in: {liquid_stats['longest_func']}")

print(f"\nPhases with similar complexity to LIQUID (±50% max line length):")
liquid_range = (liquid_stats['max_line'] * 0.5, liquid_stats['max_line'] * 1.5)
similar = [s for s in all_stats if liquid_range[0] <= s['max_line'] <= liquid_range[1] and s['phase'] != 'LIQUID']
for s in similar:
    print(f"  {s['phase']:<15} - {s['num_sublattices']} sublattices, max line: {s['max_line']}")