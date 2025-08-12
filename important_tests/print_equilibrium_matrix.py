#!/usr/bin/env python

import sys
import os
import subprocess
import tempfile

sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

def main():
    if len(sys.argv) < 4:
        sys.exit(1)
    
    db_path = os.path.abspath(sys.argv[1])
    components = sys.argv[2].split(',') if sys.argv[2].lower() != 'all' else None
    phases = sys.argv[3].split(',') if sys.argv[3].lower() != 'all' else None
    
    conditions = {'T': 1000.0, 'P': 101325.0}
    for arg in sys.argv[4:]:
        if arg.startswith('--'):
            key_val = arg[2:].split('=')
            if len(key_val) == 2:
                key, val = key_val
                if key in ['T', 'P']:
                    conditions[key] = float(val)
                elif key.startswith('X_'):
                    comp = key[2:]
                    conditions[f'X({comp})'] = float(val)
    
    # Handle 'all' keyword
    if components is None or phases is None:
        from pycalphad import Database
        dbf = Database(db_path)
        if components is None:
            components = [elem for elem in dbf.elements if elem != '/-']
        if phases is None:
            phases = list(dbf.phases.keys())
    
    # CPU calculation
    print("=== CPU MATRIX ===")
    cpu_script = f"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')
from pycalphad import Database, equilibrium
from pycalphad.core.minimizer import set_debug_mode
set_debug_mode(True)
dbf = Database(r'{db_path}')
result = equilibrium(dbf, {components}, {phases}, {conditions}, verbose=False, calc_opts={{'pdens': 50}})
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(cpu_script)
        cpu_script_path = f.name
    
    try:
        cpu_result = subprocess.run(['python', cpu_script_path], 
                                  capture_output=True, text=True, timeout=60)
        cpu_output = cpu_result.stdout + cpu_result.stderr
        
        cpu_lines = cpu_output.split('\n')
        for i, line in enumerate(cpu_lines):
            if '[CPU EQUILIBRIUM MATRIX]' in line and 'Iteration 0' in line:
                print(line)
                for j in range(1, 20):
                    if i+j < len(cpu_lines):
                        next_line = cpu_lines[i+j]
                        if next_line.strip().startswith('Row ') and '|' in next_line and 'RHS:' in next_line:
                            print(next_line)
                break
        
    finally:
        os.unlink(cpu_script_path)
    
    # GPU calculation
    print("\n=== GPU MATRIX ===")
    gpu_script = f"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')
from pycalphad import Database, equilibrium
dbf = Database(r'{db_path}')
result = equilibrium(dbf, {components}, {phases}, {conditions}, verbose=True, calc_opts={{'pdens': 50}}, gpu=True)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(gpu_script)
        gpu_script_path = f.name
    
    try:
        gpu_result = subprocess.run(['python', gpu_script_path], 
                                  capture_output=True, text=True, timeout=60)
        gpu_output = gpu_result.stdout + gpu_result.stderr
        
        gpu_lines = gpu_output.split('\n')
        for i, line in enumerate(gpu_lines):
            if '[GPU EQUILIBRIUM MATRIX]' in line and 'Iteration 0' in line:
                print(line)
                for j in range(1, 20):
                    if i+j < len(gpu_lines):
                        next_line = gpu_lines[i+j]
                        if next_line.strip().startswith('Row ') and '|' in next_line and 'RHS:' in next_line:
                            print(next_line)
                break
        
    finally:
        os.unlink(gpu_script_path)

if __name__ == '__main__':
    main()