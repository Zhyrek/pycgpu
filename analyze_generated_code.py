#!/usr/bin/env python
"""Analyze generated GPU code to find why compilation fails."""

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

print("Analyzing generated GPU code for Al-Cu-Fe phases...")
print("="*80)

# Analyze each phase
phase_stats = []

for phase in list(db.phases.keys())[:10]:
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
        for i, line in enumerate(lines):
            if len(line) > max_line_length:
                max_line_length = len(line)
                longest_line_num = i + 1
        
        # Count function types
        obj_count = all_device_functions.count('_obj(')
        grad_count = all_device_functions.count('_formulagrad(')
        hess_count = all_device_functions.count('_formulahess(')
        
        # Find the energy function size
        energy_func_size = 0
        in_energy_func = False
        for line in lines:
            if '_obj(const double* x)' in line and 'formula' not in line:
                in_energy_func = True
                energy_func_size = 0
            elif in_energy_func:
                energy_func_size += len(line)
                if line.strip() == '}':
                    in_energy_func = False
                    break
        
        stats = {
            'phase': phase,
            'total_size': len(all_device_functions),
            'max_line': max_line_length,
            'max_line_num': longest_line_num,
            'num_functions': all_device_functions.count('__device__'),
            'energy_func_size': energy_func_size,
            'obj_funcs': obj_count,
            'grad_funcs': grad_count,
            'hess_funcs': hess_count
        }
        phase_stats.append(stats)
        
    except Exception as e:
        print(f"Error processing {phase}: {e}")

# Sort by max line length
phase_stats.sort(key=lambda x: x['max_line'], reverse=True)

print("\nPhases sorted by maximum line length:")
print("-"*80)
print(f"{'Phase':<15} {'Max Line':<10} {'Total Size':<12} {'Energy Func':<12} {'Functions':<10}")
print("-"*80)

for stats in phase_stats:
    print(f"{stats['phase']:<15} {stats['max_line']:<10} {stats['total_size']:<12} {stats['energy_func_size']:<12} {stats['num_functions']:<10}")

# Identify problematic phases
print("\n" + "="*80)
print("PROBLEMATIC PHASES (line length > 5000 chars):")
print("="*80)

problematic = [s for s in phase_stats if s['max_line'] > 5000]
if problematic:
    for stats in problematic:
        print(f"\n{stats['phase']}:")
        print(f"  Max line length: {stats['max_line']} chars (line {stats['max_line_num']})")
        print(f"  Energy function size: {stats['energy_func_size']} chars")
        print(f"  This is likely causing nvcc compilation failures!")
        
        # Save a sample
        wks = Workspace(db, components, [stats['phase']], conditions, verbose=False)
        result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
        all_device_functions, init_calls, unique_models, phase_map = result
        
        lines = all_device_functions.split('\n')
        with open(f"problematic_{stats['phase']}.txt", 'w') as f:
            f.write(f"Phase: {stats['phase']}\n")
            f.write(f"Max line length: {stats['max_line']} at line {stats['max_line_num']}\n")
            f.write("="*80 + "\n\n")
            
            # Write the problematic line
            if stats['max_line_num'] <= len(lines):
                f.write(f"Line {stats['max_line_num']} ({len(lines[stats['max_line_num']-1])} chars):\n")
                f.write(lines[stats['max_line_num']-1][:500] + "...\n")
else:
    print("No phases with extremely long lines found in the first 10 phases.")

print("\nRecommendation: The GPU code generation needs to break long expressions into")
print("multiple lines or intermediate variables to avoid compiler limits.")