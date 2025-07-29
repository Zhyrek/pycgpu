#!/usr/bin/env python
"""Extract Hessian function lines for each phase and save to files."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import pycalphad.variables as v
import os

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Create workspace conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Extracting Hessian functions for all phases...")
print("="*80)

# Create output directory
output_dir = "hessian_functions"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}/")

# Process each phase
phase_stats = []

for phase_name in db.phases.keys():
    print(f"\nProcessing {phase_name}...")
    try:
        # Create workspace with single phase
        wks = Workspace(db, components, [phase_name], conditions, verbose=False)
        
        # Generate code
        result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
        all_device_functions, init_calls, unique_models, phase_map = result
        
        # Find the Hessian function
        lines = all_device_functions.split('\n')
        
        # Look for formulahess function
        hessian_start = -1
        hessian_end = -1
        for i, line in enumerate(lines):
            if '__device__' in line and 'formulahess' in line:
                hessian_start = i
            elif hessian_start >= 0 and line.strip() == '}':
                hessian_end = i
                break
        
        if hessian_start >= 0 and hessian_end >= 0:
            # Extract Hessian function
            hessian_lines = lines[hessian_start:hessian_end+1]
            
            # Find the longest line
            max_length = 0
            max_line_idx = 0
            for i, line in enumerate(hessian_lines):
                if len(line) > max_length:
                    max_length = len(line)
                    max_line_idx = i
            
            # Save to file
            filename = f"{output_dir}/hessian_{phase_name}.c"
            with open(filename, 'w') as f:
                # Write header info
                f.write(f"// Hessian function for phase: {phase_name}\n")
                f.write(f"// Total lines: {len(hessian_lines)}\n")
                f.write(f"// Longest line: {max_length} characters (line {max_line_idx+1})\n")
                f.write("// " + "="*70 + "\n\n")
                
                # Write the function
                for line in hessian_lines:
                    f.write(line + '\n')
            
            print(f"  ✓ Saved Hessian to: {filename}")
            print(f"    Function lines: {len(hessian_lines)}")
            print(f"    Longest line: {max_length:,} chars (line {max_line_idx+1})")
            
            # Store stats
            phase_stats.append({
                'phase': phase_name,
                'lines': len(hessian_lines),
                'max_length': max_length,
                'max_line_num': max_line_idx + 1,
                'filename': filename
            })
            
        else:
            print(f"  ✗ Could not find Hessian function")
            
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {str(e)[:100]}")

# Sort by max line length
phase_stats.sort(key=lambda x: x['max_length'], reverse=True)

# Create summary file
summary_file = f"{output_dir}/SUMMARY.txt"
with open(summary_file, 'w') as f:
    f.write("HESSIAN FUNCTION SUMMARY FOR Al-Cu-Fe PHASES\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"{'Phase':<20} {'Lines':<10} {'Max Line Length':<20} {'Max Line #':<12} {'File':<30}\n")
    f.write("-" * 80 + "\n")
    
    for stats in phase_stats:
        f.write(f"{stats['phase']:<20} {stats['lines']:<10} {stats['max_length']:<20,} {stats['max_line_num']:<12} {stats['filename']:<30}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("PHASES BY COMPLEXITY:\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Simple (< 5,000 chars max line):\n")
    for s in phase_stats:
        if s['max_length'] < 5000:
            f.write(f"  {s['phase']:<20} - {s['max_length']:,} chars\n")
    
    f.write("\nModerate (5,000 - 20,000 chars max line):\n")
    for s in phase_stats:
        if 5000 <= s['max_length'] < 20000:
            f.write(f"  {s['phase']:<20} - {s['max_length']:,} chars\n")
    
    f.write("\nComplex (20,000 - 50,000 chars max line):\n")
    for s in phase_stats:
        if 20000 <= s['max_length'] < 50000:
            f.write(f"  {s['phase']:<20} - {s['max_length']:,} chars\n")
    
    f.write("\nVery Complex (>= 50,000 chars max line):\n")
    for s in phase_stats:
        if s['max_length'] >= 50000:
            f.write(f"  {s['phase']:<20} - {s['max_length']:,} chars\n")

print(f"\n{'='*80}")
print(f"Summary saved to: {summary_file}")
print(f"All Hessian functions saved in: {output_dir}/")

# Also create a file with just the problematic long lines
problem_file = f"{output_dir}/PROBLEMATIC_LINES.txt"
with open(problem_file, 'w') as f:
    f.write("PROBLEMATIC LONG HESSIAN LINES (> 10,000 chars)\n")
    f.write("=" * 80 + "\n\n")
    
    for stats in phase_stats:
        if stats['max_length'] > 10000:
            f.write(f"\nPhase: {stats['phase']}\n")
            f.write(f"Line length: {stats['max_length']:,} chars\n")
            f.write("-" * 80 + "\n")
            
            # Read the file and extract the long line
            with open(stats['filename'], 'r') as hf:
                lines = hf.readlines()
                # Skip header comments
                for i, line in enumerate(lines):
                    if len(line.strip()) > 10000:
                        f.write(f"Line {i+1} ({len(line.strip())} chars):\n")
                        # Show first and last 500 chars
                        if len(line) > 1000:
                            f.write(line[:500] + "\n...\n[MIDDLE SECTION OMITTED]\n...\n" + line[-500:])
                        else:
                            f.write(line)
                        f.write("\n" + "-"*60 + "\n")

print(f"Problematic long lines saved to: {problem_file}")

# Extract specific phases for detailed comparison
comparison_phases = ['LIQUID', 'ALCU_ZETA', 'BCC_B2', 'AL5FE4']
comparison_file = f"{output_dir}/COMPARISON.txt"

with open(comparison_file, 'w') as f:
    f.write("DETAILED COMPARISON OF SELECTED PHASES\n")
    f.write("=" * 80 + "\n\n")
    
    for phase in comparison_phases:
        stats = next((s for s in phase_stats if s['phase'] == phase), None)
        if stats:
            f.write(f"\n{phase}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total lines: {stats['lines']}\n")
            f.write(f"  Max line length: {stats['max_length']:,} chars\n")
            f.write(f"  Max line at: line {stats['max_line_num']}\n")
            
            # Show structure of the longest line
            with open(stats['filename'], 'r') as hf:
                lines = hf.readlines()
                for i, line in enumerate(lines):
                    if i + 1 == stats['max_line_num']:
                        # Count operations
                        f.write(f"\n  Structure of longest line:\n")
                        f.write(f"    Additions (+): {line.count('+')}\n")
                        f.write(f"    Multiplications (*): {line.count('*')}\n")
                        f.write(f"    pow() calls: {line.count('pow(')}\n")
                        f.write(f"    log() calls: {line.count('log(')}\n")
                        f.write(f"    Ternary operators (?): {line.count('?')}\n")
                        f.write(f"    Parentheses pairs: {line.count('(')}\n")
                        break

print(f"Comparison of selected phases saved to: {comparison_file}")