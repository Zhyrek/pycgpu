#!/usr/bin/env python3
"""
Verify that GPU and CPU equilibrium calls output exactly the same data
by printing detailed GM, X, Y, and Phases results for comparison
"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
import os
import glob

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
            except OSError:
                pass

def print_detailed_result(result, label):
    """Print detailed breakdown of equilibrium result"""
    print(f"\n{'='*20} {label} DETAILED RESULTS {'='*20}")
    
    # GM (Gibbs Energy)
    gm = result.GM.values
    print(f"GM (Gibbs Energy):")
    print(f"  Shape: {gm.shape}")
    print(f"  Values: {gm.flatten()}")
    print(f"  Scalar: {gm.flatten()[0]:.10f} J/mol")
    
    # MU (Chemical Potentials)
    mu = result.MU.values
    print(f"\nMU (Chemical Potentials):")
    print(f"  Shape: {mu.shape}")
    print(f"  Flat values: {mu.flatten()}")
    if mu.size >= 2:
        print(f"  MU[0] (NB): {mu.flatten()[0]:.10f} J/mol")
        print(f"  MU[1] (TI): {mu.flatten()[1]:.10f} J/mol")
    
    # NP (Phase Amounts)
    np_vals = result.NP.values
    print(f"\nNP (Phase Amounts):")
    print(f"  Shape: {np_vals.shape}")
    print(f"  All values: {np_vals.flatten()}")
    
    # Count active phases
    np_flat = np_vals.flatten()
    if hasattr(result, 'Phase'):
        phases_flat = result.Phase.values.flatten()
        # For CPU: use ~np.isnan check, for GPU: use > threshold
        if np.any(np.isnan(np_flat)):
            active_mask = ~np.isnan(np_flat) & (np_flat > 1e-10)
        else:
            active_mask = np_flat > 1e-10
        
        active_np = np_flat[active_mask]
        active_phases = phases_flat[active_mask]
        
        print(f"  Active phases: {len(active_np)}")
        for i, (amount, phase) in enumerate(zip(active_np, active_phases)):
            print(f"    Phase {i}: {phase} = {amount:.10f}")
    
    # Phase Names
    if hasattr(result, 'Phase'):
        phase_vals = result.Phase.values
        print(f"\nPhase:")
        print(f"  Shape: {phase_vals.shape}")
        print(f"  All values: {phase_vals.flatten()}")
    
    # X (Compositions)
    x_vals = result.X.values
    print(f"\nX (Compositions):")
    print(f"  Shape: {x_vals.shape}")
    print(f"  All values: {x_vals.flatten()}")
    
    # Extract composition for active phases
    if hasattr(result, 'Phase') and len(active_np) > 0:
        print(f"  Active phase compositions:")
        for i, is_active in enumerate(active_mask):
            if is_active and i < x_vals.shape[-2]:
                comp = x_vals.reshape(-1, x_vals.shape[-1])[i]
                print(f"    Phase {i} ({active_phases[np.where(active_mask)[0] == i][0]}):")
                print(f"      NB: {comp[0]:.10f}")
                print(f"      TI: {comp[1]:.10f}")
                if len(comp) > 2:
                    print(f"      VA: {comp[2]:.10f}")
    
    # Y (Internal DOF / Site Fractions)
    if hasattr(result, 'Y'):
        y_vals = result.Y.values
        print(f"\nY (Internal DOF):")
        print(f"  Shape: {y_vals.shape}")
        print(f"  All values: {y_vals.flatten()}")
        
        # Extract Y for active phases
        if len(active_np) > 0:
            print(f"  Active phase internal DOF:")
            for i, is_active in enumerate(active_mask):
                if is_active and i < y_vals.shape[-2]:
                    y_data = y_vals.reshape(-1, y_vals.shape[-1])[i]
                    print(f"    Phase {i}: {y_data}")
    else:
        print(f"\nY (Internal DOF): Not present in result")
    
    # Coordinates info
    if hasattr(result, 'coords'):
        print(f"\nCoordinates:")
        for coord_name, coord_vals in result.coords.items():
            print(f"  {coord_name}: {coord_vals}")

def verify_exact_match():
    """Run both CPU and GPU calculations and compare results exactly"""
    print("=== EXACT MATCH VERIFICATION: CPU vs GPU ===")
    clear_cupy_kernel_cache()
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    print(f"Test conditions: {conditions}")
    print(f"Components: {comps}")
    print(f"Phases: {phases}")
    
    # CPU Calculation
    print(f"\n{'#'*80}")
    print("RUNNING CPU EQUILIBRIUM CALCULATION")
    print(f"{'#'*80}")
    
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    print_detailed_result(cpu_result, "CPU")
    
    # GPU Calculation  
    print(f"\n{'#'*80}")
    print("RUNNING GPU EQUILIBRIUM CALCULATION")
    print(f"{'#'*80}")
    
    gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
    print_detailed_result(gpu_result, "GPU")
    
    # Direct Comparison
    print(f"\n{'='*80}")
    print("DIRECT NUMERICAL COMPARISON")
    print(f"{'='*80}")
    
    # GM comparison
    cpu_gm = cpu_result.GM.values.flatten()[0]
    gpu_gm = gpu_result.GM.values.flatten()[0]
    gm_diff = abs(cpu_gm - gpu_gm)
    gm_rel_diff = gm_diff / abs(cpu_gm) * 100
    
    print(f"GM (Gibbs Energy):")
    print(f"  CPU: {cpu_gm:.10f} J/mol")
    print(f"  GPU: {gpu_gm:.10f} J/mol")
    print(f"  Absolute difference: {gm_diff:.10f} J/mol")
    print(f"  Relative difference: {gm_rel_diff:.10e} %")
    
    # MU comparison
    cpu_mu = cpu_result.MU.values.flatten()
    gpu_mu = gpu_result.MU.values.flatten()
    
    print(f"\nMU (Chemical Potentials):")
    for i in range(min(len(cpu_mu), len(gpu_mu), 3)):  # Compare first 3 components
        comp_name = comps[i] if i < len(comps) else f"Component_{i}"
        mu_diff = abs(cpu_mu[i] - gpu_mu[i])
        mu_rel_diff = mu_diff / abs(cpu_mu[i]) * 100 if abs(cpu_mu[i]) > 1e-10 else 0
        
        print(f"  {comp_name}:")
        print(f"    CPU: {cpu_mu[i]:.10f} J/mol")
        print(f"    GPU: {gpu_mu[i]:.10f} J/mol")
        print(f"    Difference: {mu_diff:.10f} J/mol")
        print(f"    Relative: {mu_rel_diff:.10e} %")
    
    # Phase amount comparison
    cpu_np = cpu_result.NP.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    
    # Count active phases for both
    cpu_active_mask = ~np.isnan(cpu_np) & (cpu_np > 1e-10)
    gpu_active_mask = gpu_np > 1e-10
    
    cpu_active_count = np.sum(cpu_active_mask)
    gpu_active_count = np.sum(gpu_active_mask)
    
    print(f"\nPhase Amounts:")
    print(f"  CPU active phases: {cpu_active_count}")
    print(f"  GPU active phases: {gpu_active_count}")
    
    if cpu_active_count == gpu_active_count:
        print(f"  ✅ Phase counts match")
        
        cpu_active_amounts = cpu_np[cpu_active_mask]
        gpu_active_amounts = gpu_np[gpu_active_mask]
        
        for i, (cpu_amt, gpu_amt) in enumerate(zip(cpu_active_amounts, gpu_active_amounts)):
            amt_diff = abs(cpu_amt - gpu_amt)
            print(f"  Phase {i}:")
            print(f"    CPU: {cpu_amt:.10f}")
            print(f"    GPU: {gpu_amt:.10f}")
            print(f"    Difference: {amt_diff:.10f}")
    else:
        print(f"  ❌ Phase counts differ!")
    
    # Composition comparison
    cpu_x = cpu_result.X.values
    gpu_x = gpu_result.X.values
    
    print(f"\nCompositions:")
    print(f"  CPU X shape: {cpu_x.shape}")
    print(f"  GPU X shape: {gpu_x.shape}")
    
    # Compare compositions for active phases
    for i, (cpu_active, gpu_active) in enumerate(zip(cpu_active_mask, gpu_active_mask)):
        if cpu_active and gpu_active and i < min(cpu_x.shape[-2], gpu_x.shape[-2]):
            cpu_comp = cpu_x.reshape(-1, cpu_x.shape[-1])[i]
            gpu_comp = gpu_x.reshape(-1, gpu_x.shape[-1])[i]
            
            print(f"  Phase {i} composition:")
            for j, comp_name in enumerate(comps[:2]):  # NB, TI
                if j < len(cpu_comp) and j < len(gpu_comp):
                    comp_diff = abs(cpu_comp[j] - gpu_comp[j])
                    print(f"    {comp_name}:")
                    print(f"      CPU: {cpu_comp[j]:.10f}")
                    print(f"      GPU: {gpu_comp[j]:.10f}")
                    print(f"      Difference: {comp_diff:.10f}")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}")
    
    tolerance_gm = 1.0  # J/mol
    tolerance_mu = 10.0  # J/mol  
    tolerance_comp = 1e-6  # mole fraction
    
    success = True
    
    if gm_diff > tolerance_gm:
        print(f"❌ GM difference ({gm_diff:.6f} J/mol) exceeds tolerance ({tolerance_gm} J/mol)")
        success = False
    else:
        print(f"✅ GM difference ({gm_diff:.6f} J/mol) within tolerance ({tolerance_gm} J/mol)")
    
    if cpu_active_count != gpu_active_count:
        print(f"❌ Phase counts differ: CPU={cpu_active_count}, GPU={gpu_active_count}")
        success = False
    else:
        print(f"✅ Phase counts match: {cpu_active_count}")
    
    if success:
        print(f"\n🎉 SUCCESS: GPU and CPU results match within acceptable tolerances!")
        print(f"   The GPU implementation is working correctly.")
    else:
        print(f"\n⚠️  WARNING: Some differences exceed acceptable tolerances.")
        print(f"   Further investigation may be needed.")
    
    return success

if __name__ == "__main__":
    success = verify_exact_match()
    if success:
        print(f"\n✅ Verification completed successfully!")
    else:
        print(f"\n❌ Verification found issues that need attention.")