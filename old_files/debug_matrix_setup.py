#!/usr/bin/env python
"""
Debug script to examine the equilibrium matrix setup between CPU and GPU.
"""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.starting_point import starting_point
from pycalphad import calculate
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

def debug_matrix_setup():
    """Compare matrix setup between CPU and GPU"""
    print("="*60)
    print("MATRIX SETUP DEBUG: CPU vs GPU")
    print("="*60)
    
    # Use the same problematic condition
    dbf = Database("NbTi.tdb")
    conditions = {v.X("TI"): 1e-10, v.T: 600, v.P: 101325}
    
    print(f"Test condition: T=600K, X_TI=1e-10")
    print(f"This should create a simple 3x3 equilibrium matrix for single BCC_A2 phase")
    
    # Get the CPU equilibrium result
    print(f"\n1. Running CPU equilibrium for reference...")
    cpu_result = equilibrium(dbf, ['NB', 'TI', 'VA'], ['LIQUID', 'BCC_A2'], 
                           conditions, gpu=False, verbose=False)
    
    print(f"CPU result: GM={cpu_result.GM.values[0,0,0,0]:.6f}")
    print(f"CPU chemical potentials: {cpu_result.MU.values[0,0,0,0,:]}")
    
    # Get the starting workspace to examine what GPU sees
    print(f"\n2. Examining workspace state...")
    wks = Workspace(database=dbf, components=['NB', 'TI', 'VA'], 
                    phases=['LIQUID', 'BCC_A2'], conditions=conditions)
    
    starting_props = wks.eq
    print(f"Starting point GM: {np.array(starting_props.GM)[0,0,0,0]:.6f}")
    print(f"Starting point MU: {np.array(starting_props.MU)[0,0,0,0,:]}")
    print(f"Starting point phases: {np.array(starting_props.Phase)[0,0,0,0,:]}")
    print(f"Starting point amounts: {np.array(starting_props.NP)[0,0,0,0,:]}")
    
    # Check the expected matrix dimensions
    num_components = 3  # NB, TI, VA (excluding VA for chemical potentials)
    num_free_chemical_potentials = 2  # NB, TI (VA is dependent)
    num_free_stable_compsets = 1  # Single BCC_A2 phase
    num_free_statevars = 0  # All are fixed (P, T)
    
    expected_matrix_size = num_free_chemical_potentials + num_free_stable_compsets + num_free_statevars
    print(f"\n3. Expected matrix analysis:")
    print(f"   num_free_chemical_potentials = {num_free_chemical_potentials}")
    print(f"   num_free_stable_compsets = {num_free_stable_compsets}")
    print(f"   num_free_statevars = {num_free_statevars}")
    print(f"   Expected matrix size = {expected_matrix_size} x {expected_matrix_size}")
    
    # Check what the GPU debug output showed
    print(f"\n4. GPU debug output analysis:")
    print(f"   GPU reported: eq_soln_len=3, matrix_rows=3, matrix_cols=3 ✓")
    print(f"   GPU eq_soln result: [0.0, 0.0, 1.0]")
    print(f"   Expected interpretation:")
    print(f"     eq_soln[0] = change in μ_NB = 0.0")
    print(f"     eq_soln[1] = change in μ_TI = 0.0") 
    print(f"     eq_soln[2] = change in phase_amount = 1.0")
    
    # The solution [0,0,1] suggests:
    # - No change to chemical potentials (which might be correct if already at equilibrium)
    # - Change in phase amount by 1.0 (which doesn't make sense - amount is already 1.0)
    
    print(f"\n5. Analysis of eq_soln=[0,0,1]:")
    print(f"   If system is already at equilibrium (which it should be from starting_point):")
    print(f"     - Changes to μ should be ~0 ✓") 
    print(f"     - Changes to phase amounts should be ~0 ❌")
    print(f"   The fact that eq_soln[2]=1.0 suggests the matrix setup is wrong")
    
    print(f"\n6. Possible issues:")
    print(f"   a) Matrix RHS vector is incorrectly populated")
    print(f"   b) Equilibrium constraints are wrong")
    print(f"   c) SVD solver is failing") 
    print(f"   d) advance_state is applying the solution incorrectly")
    
    return wks, starting_props

def analyze_expected_equilibrium_conditions():
    """Analyze what the equilibrium conditions should be"""
    print(f"\n" + "="*60)
    print("EQUILIBRIUM CONDITION ANALYSIS")
    print("="*60)
    
    print(f"For a single-phase system at equilibrium, the matrix equation should be:")
    print(f"")
    print(f"[ ∂²G/∂μ²  | ∂²G/∂μ∂n  ] [ Δμ ]   [ -∂G/∂μ ]")
    print(f"[----------|----------  ] [----] = [-------]")  
    print(f"[ ∂²G/∂n∂μ| ∂²G/∂n²   ] [ Δn ]   [ -∂G/∂n ]")
    print(f"")
    print(f"For BCC_A2 at T=600K, X_TI=1e-10:")
    print(f"  - The system should already be at equilibrium from starting_point")
    print(f"  - Therefore RHS should be ~[0, 0, 0]")
    print(f"  - Therefore solution should be ~[0, 0, 0]")
    print(f"  - The fact we get [0, 0, 1] suggests RHS is wrong")

def main():
    """Main debugging function"""
    print("EQUILIBRIUM MATRIX SETUP DEBUG")
    print("Investigating why GPU solver gives eq_soln=[0,0,1]")
    
    wks, props = debug_matrix_setup()
    analyze_expected_equilibrium_conditions()
    
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print(f"1. Add debug output to fill_equilibrium_system to check RHS vector")
    print(f"2. Add debug output to SVD solver to verify matrix solution")
    print(f"3. Add debug output to advance_state to see how solution is applied")
    print(f"4. Compare CPU matrix setup with GPU matrix setup")

if __name__ == "__main__":
    main()