#!/usr/bin/env python
"""Analyze line lengths for all phases, focusing on LIQUID."""

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

print("Analyzing line lengths for all phases...")
print("="*80)

# Analyze each phase
phase_stats = []

for phase_name in db.phases.keys():
    print(f"\nProcessing {phase_name}...")
    try:
        # Create workspace with single phase  
        wks = Workspace(db, components, [phase_name], conditions, verbose=False)
        
        # Generate code
        result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
        all_device_functions, init_calls, unique_models, phase_map = result
        
        # Analyze line lengths
        lines = all_device_functions.split('\n')
        line_lengths = [len(line) for line in lines]
        
        if line_lengths:
            max_line = max(line_lengths)
            max_line_idx = line_lengths.index(max_line) + 1
            avg_line = sum(line_lengths) / len(line_lengths)
            
            # Get phase info
            phase_obj = db.phases[phase_name]
            num_sublattices = len(phase_obj.sublattices)
            
            # Find which function contains the longest line
            current_func = "unknown"
            for i, line in enumerate(lines[:max_line_idx]):
                if '__device__' in line:
                    if 'formulahess' in line:
                        current_func = "hessian"
                    elif 'formulagrad' in line:
                        current_func = "gradient"
                    elif '_obj(' in line and 'formula' not in line:
                        current_func = "energy"
                    elif 'internal_cons_jac' in line:
                        current_func = "constraint_jac"
            
            stats = {
                'phase': phase_name,
                'sublattices': num_sublattices,
                'max_line': max_line,
                'avg_line': avg_line,
                'total_chars': len(all_device_functions),
                'num_lines': len(lines),
                'longest_in': current_func
            }
            phase_stats.append(stats)
            
            print(f"  Sublattices: {num_sublattices}")
            print(f"  Max line: {max_line:,} chars")
            print(f"  Average line: {avg_line:.0f} chars")
            print(f"  Longest line in: {current_func}")
            
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:100]}")

# Sort and display results
phase_stats.sort(key=lambda x: x['max_line'])

print("\n" + "="*80)
print("PHASES BY MAX LINE LENGTH:")
print("="*80)
print(f"{'Phase':<15} {'Sublattices':<12} {'Max Line':<15} {'Avg Line':<10} {'Total Chars':<12} {'In Function':<15}")
print("-"*80)

for stats in phase_stats:
    print(f"{stats['phase']:<15} {stats['sublattices']:<12} {stats['max_line']:<15,} {stats['avg_line']:<10.0f} "
          f"{stats['total_chars']:<12,} {stats['longest_in']:<15}")

# Find LIQUID specifically
liquid_stats = next((s for s in phase_stats if s['phase'] == 'LIQUID'), None)

if liquid_stats:
    print(f"\n" + "="*80)
    print("LIQUID PHASE DETAILS:")
    print("="*80)
    print(f"  Sublattices: {liquid_stats['sublattices']}")
    print(f"  Max line length: {liquid_stats['max_line']:,} chars")
    print(f"  Average line length: {liquid_stats['avg_line']:.0f} chars")
    print(f"  Total characters: {liquid_stats['total_chars']:,}")
    print(f"  Number of lines: {liquid_stats['num_lines']:,}")
    print(f"  Longest line in: {liquid_stats['longest_in']}")
    
    # Find phases with similar max line length
    tolerance = 0.5  # 50% tolerance
    lower = liquid_stats['max_line'] * (1 - tolerance)
    upper = liquid_stats['max_line'] * (1 + tolerance)
    
    similar = [s for s in phase_stats if lower <= s['max_line'] <= upper and s['phase'] != 'LIQUID']
    
    if similar:
        print(f"\nPhases with similar max line length (±50%):")
        for s in similar:
            print(f"  {s['phase']:<15} - {s['sublattices']} sublattices, max: {s['max_line']:,} chars")
    else:
        print(f"\nNo phases with similar line lengths to LIQUID")

# Group by line length categories
print("\n" + "="*80)
print("PHASES GROUPED BY LINE LENGTH:")
print("="*80)

short_lines = [s for s in phase_stats if s['max_line'] < 5000]
medium_lines = [s for s in phase_stats if 5000 <= s['max_line'] < 20000]
long_lines = [s for s in phase_stats if 20000 <= s['max_line'] < 50000]
very_long_lines = [s for s in phase_stats if s['max_line'] >= 50000]

print(f"\nShort lines (<5,000 chars): {len(short_lines)} phases")
if short_lines:
    for s in short_lines[:5]:
        print(f"  {s['phase']:<15} - {s['sublattices']} sublattices, max: {s['max_line']:,}")

print(f"\nMedium lines (5,000-20,000 chars): {len(medium_lines)} phases")
if medium_lines:
    for s in medium_lines[:5]:
        print(f"  {s['phase']:<15} - {s['sublattices']} sublattices, max: {s['max_line']:,}")

print(f"\nLong lines (20,000-50,000 chars): {len(long_lines)} phases")
if long_lines:
    for s in long_lines[:5]:
        print(f"  {s['phase']:<15} - {s['sublattices']} sublattices, max: {s['max_line']:,}")

print(f"\nVery long lines (≥50,000 chars): {len(very_long_lines)} phases")
if very_long_lines:
    for s in very_long_lines:
        print(f"  {s['phase']:<15} - {s['sublattices']} sublattices, max: {s['max_line']:,}")

# Analyze correlation
print("\n" + "="*80)
print("SUBLATTICE CORRELATION:")
print("="*80)

by_sublattices = {}
for s in phase_stats:
    n = s['sublattices']
    if n not in by_sublattices:
        by_sublattices[n] = []
    by_sublattices[n].append(s['max_line'])

for n in sorted(by_sublattices.keys()):
    lines = by_sublattices[n]
    avg = sum(lines) / len(lines)
    print(f"\n{n} sublattices ({len(lines)} phases):")
    print(f"  Average max line: {avg:,.0f} chars")
    print(f"  Range: {min(lines):,} - {max(lines):,} chars")