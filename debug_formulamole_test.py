
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database
from pycalphad.core.utils import filter_phases
from pycalphad.gpu.gpu_codegen import _generate_phase_records
import numpy as np

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Generate phase records to see the formulamole function
phase_records, models = _generate_phase_records(dbf, comps, phases, {}, (('NB', 'TI'), 'VA'), None)

# Check BCC_A2 formulamole
for pr, model in zip(phase_records, models):
    if pr['phase_name'] == 'BCC_A2':
        print(f"\nBCC_A2 formulamole function:")
        print(f"Number of elements: {model.components}")
        print(f"Site ratios: {model._site_ratios}")
        
        # Test the function with the DOF values we see
        dof = np.array([1.0, 101325.0, 1000.0, 0.991525, 0.008475])
        
        # The mole fractions should be calculated as:
        # For a single sublattice with NB and TI:
        # moles(NB) = Y(NB) * site_ratio
        # moles(TI) = Y(TI) * site_ratio
        # X(NB) = moles(NB) / (moles(NB) + moles(TI))
        # X(TI) = moles(TI) / (moles(NB) + moles(TI))
        
        y_nb = dof[3]
        y_ti = dof[4]
        print(f"\nInput site fractions: Y(NB)={y_nb:.6f}, Y(TI)={y_ti:.6f}")
        print(f"Sum of site fractions: {y_nb + y_ti:.6f}")
        
        # For single sublattice, X should equal Y
        print(f"\nExpected X(NB) = {y_nb:.6f}")
        print(f"Expected X(TI) = {y_ti:.6f}")
        
        # But we're seeing X(NB)=0.989890, X(TI)=0.010110
        # Let's check if this is a normalization issue
        sum_expected = 0.991525 + 0.008475  # = 1.0
        sum_observed = 0.989890 + 0.010110  # = 1.0
        
        print(f"\nObserved X(NB) = 0.989890, X(TI) = 0.010110")
        print(f"Sum of observed: {sum_observed:.6f}")
        
        # The pattern suggests the issue might be in the DOF indexing
        # or in how the dependent site fraction is handled
        break
