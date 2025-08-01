#!/usr/bin/env python3
"""Simple test to understand vacancy handling differences between CPU and GPU"""

import numpy as np
from pycalphad import Database, equilibrium, Model
import pycalphad.variables as v

def test_simple_vacancy():
    """Test with simple conditions to trace vacancy handling"""
    
    # Initialize database and system
    db = Database('AuBi-07Wan.tdb')
    phases = ['FCC_A1']  # Only FCC which has vacancy
    comps = ['AU', 'BI', 'VA']
    
    # Test condition
    T = 600
    X_BI = 0.3
    
    print(f"\n{'='*80}")
    print(f"Testing single phase FCC_A1 at T={T}K, X(BI)={X_BI}")
    print(f"{'='*80}\n")
    
    # Check phase model
    mod_fcc = Model(db, comps, 'FCC_A1')
    print(f"FCC_A1 Phase Model:")
    print(f"  Sublattices: {mod_fcc.site_fractions}")
    print(f"  Site ratios: {mod_fcc.site_ratios}")
    print(f"  Sum of site ratios: {sum(mod_fcc.site_ratios)}")
    
    # Calculate formulamole_obj for a test composition
    print(f"\nFormula mole calculation:")
    # For FCC_A1: first sublattice has AU/BI, second has VA
    # Test with Y(FCC_A1,0,AU)=0.7, Y(FCC_A1,0,BI)=0.3, Y(FCC_A1,1,VA)=1.0
    test_y = [0.7, 0.3, 1.0]
    
    # Calculate moles of each component per formula unit
    # AU: site_ratio[0] * Y(FCC_A1,0,AU) = 1.0 * 0.7 = 0.7
    # BI: site_ratio[0] * Y(FCC_A1,0,BI) = 1.0 * 0.3 = 0.3
    # VA: site_ratio[1] * Y(FCC_A1,1,VA) = 1.0 * 1.0 = 1.0
    
    moles_au = mod_fcc.site_ratios[0] * 0.7
    moles_bi = mod_fcc.site_ratios[0] * 0.3
    moles_va = mod_fcc.site_ratios[1] * 1.0
    
    print(f"  Moles AU per formula unit: {moles_au}")
    print(f"  Moles BI per formula unit: {moles_bi}")
    print(f"  Moles VA per formula unit: {moles_va}")
    print(f"  Total moles per formula unit: {moles_au + moles_bi + moles_va}")
    
    # But VA has number_of_atoms = 0, so it contributes 0 to system moles
    moles_atoms_au = moles_au * 1  # AU has 1 atom
    moles_atoms_bi = moles_bi * 1  # BI has 1 atom  
    moles_atoms_va = moles_va * 0  # VA has 0 atoms
    
    print(f"\nMoles of atoms per formula unit:")
    print(f"  Moles atoms AU: {moles_atoms_au}")
    print(f"  Moles atoms BI: {moles_atoms_bi}")
    print(f"  Moles atoms VA: {moles_atoms_va}")
    print(f"  Total moles atoms per formula unit: {moles_atoms_au + moles_atoms_bi + moles_atoms_va}")
    
    print(f"\nMole fractions in system:")
    total_atoms = moles_atoms_au + moles_atoms_bi
    print(f"  X(AU) = {moles_atoms_au / total_atoms}")
    print(f"  X(BI) = {moles_atoms_bi / total_atoms}")
    print(f"  X(VA) = {moles_atoms_va / total_atoms} (should be 0)")
    
    # Run CPU equilibrium with fixed composition
    print(f"\n{'='*40} CPU Equilibrium {'='*40}")
    result_cpu = equilibrium(
        db, comps, phases,
        {v.X('BI'): X_BI, v.T: T, v.P: 101325},
        output='GM',
        calc_opts={'pdens': 10},
        to_xarray=False
    )
    
    cpu_gm = float(result_cpu.GM.squeeze())
    cpu_np = float(result_cpu.NP.squeeze()[0])
    cpu_x = result_cpu.X.squeeze()[0]
    
    print(f"CPU Results:")
    print(f"  GM: {cpu_gm:.6f} J/mol")
    print(f"  NP: {cpu_np:.6f}")
    print(f"  X(AU): {cpu_x[0]:.6f}")
    print(f"  X(BI): {cpu_x[1]:.6f}")
    print(f"  (Note: X array only contains non-vacant elements)")
    
    # Run GPU equilibrium
    print(f"\n{'='*40} GPU Equilibrium {'='*40}")
    result_gpu = equilibrium(
        db, comps, phases,
        {v.X('BI'): X_BI, v.T: T, v.P: 101325},
        output='GM',
        calc_opts={'pdens': 10},
        to_xarray=False,
        gpu=True
    )
    
    gpu_gm = float(result_gpu.GM.squeeze())
    gpu_np = float(result_gpu.NP.squeeze()[0])
    gpu_x = result_gpu.X.squeeze()[0]
    
    print(f"GPU Results:")
    print(f"  GM: {gpu_gm:.6f} J/mol")
    print(f"  NP: {gpu_np:.6f}")
    print(f"  X(AU): {gpu_x[0]:.6f}")
    print(f"  X(BI): {gpu_x[1]:.6f}")
    print(f"  (Note: X array only contains non-vacant elements)")
    
    print(f"\n{'='*80}")
    print(f"Differences:")
    print(f"  GM difference: {gpu_gm - cpu_gm:.6f} J/mol")
    print(f"  NP difference: {gpu_np - cpu_np:.6f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_simple_vacancy()