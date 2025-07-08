#\!/usr/bin/env python3
"""Check GPU hessian values at workspace indices"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Modify minimizer.h to print hessian at workspace indices
import shutil
shutil.copy('/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h', 
            '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h.backup')

with open('/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h', 'r') as f:
    content = f.read()

# Replace the GPU HESSIAN debug print to show values at indices [3,3] and [3,4]
old_print = '''                    printf("  H[1,1] (index %d) = %e\\n", 1 * csst->hess_cols + 1, csst->hess[1 * csst->hess_cols + 1]);
                    printf("  H[1,2] (index %d) = %e\\n", 1 * csst->hess_cols + 2, csst->hess[1 * csst->hess_cols + 2]);
                    printf("  H[2,1] (index %d) = %e\\n", 2 * csst->hess_cols + 1, csst->hess[2 * csst->hess_cols + 1]);
                    printf("  H[2,2] (index %d) = %e\\n", 2 * csst->hess_cols + 2, csst->hess[2 * csst->hess_cols + 2]);'''

new_print = '''                    printf("  H[3,3] (index %d) = %e\\n", 3 * csst->hess_cols + 3, csst->hess[3 * csst->hess_cols + 3]);
                    printf("  H[3,4] (index %d) = %e\\n", 3 * csst->hess_cols + 4, csst->hess[3 * csst->hess_cols + 4]);
                    printf("  H[4,3] (index %d) = %e\\n", 4 * csst->hess_cols + 3, csst->hess[4 * csst->hess_cols + 3]);
                    printf("  H[4,4] (index %d) = %e\\n", 4 * csst->hess_cols + 4, csst->hess[4 * csst->hess_cols + 4]);'''

content = content.replace(old_print, new_print)

with open('/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h', 'w') as f:
    f.write(content)

# Run test
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

try:
    result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=True, to='GM', calc_opts={'pdens': 50})
    print(f"GPU GM: {float(result.GM.values)} J/mol")
except Exception as e:
    print(f"Error: {e}")

# Restore backup
shutil.move('/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h.backup',
            '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h')
