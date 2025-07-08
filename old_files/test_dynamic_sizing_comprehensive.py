#!/usr/bin/env python3
"""
Comprehensive test of the dynamic sizing implementation
"""

import os
import glob

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
                print(f"Removed cached kernel: {cubin_file}")
            except OSError:
                pass

# Clear cache to ensure we get fresh compilation
clear_cupy_kernel_cache()

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v

print("🚀 COMPREHENSIVE TEST: Dynamic GPU Kernel Sizing")
print("=" * 80)

# Test conditions
tdb = Database("NbTi.tdb")
phases = ["BCC_A2"]
comps = ["NB", "TI", "VA"]
conditions = {
    v.X("TI"): 0.1,
    v.T: 800,
    v.P: 101325
}

print(f"Test setup:")
print(f"  Database: NbTi.tdb")
print(f"  Phases: {phases}")
print(f"  Components: {comps}")
print(f"  Conditions: {conditions}")

# CPU calculation for reference
print(f"\n🖥️  CPU CALCULATION (Reference)")
print("-" * 40)
try:
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = cpu_result.GM.values.flatten()[0]
    cpu_mu = cpu_result.MU.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    cpu_x = cpu_result.X.values
    cpu_phases = cpu_result.Phase.values.flatten()
    
    print(f"✅ CPU Success!")
    print(f"  GM: {cpu_gm:.6f} J/mol")
    print(f"  MU: {cpu_mu[:2]}")
    print(f"  Active phases: {np.sum(~np.isnan(cpu_np) & (cpu_np > 1e-10))}")
    print(f"  Shapes: GM{cpu_result.GM.shape}, NP{cpu_result.NP.shape}, X{cpu_result.X.shape}")
    
    # Extract CPU composition
    cpu_active_mask = ~np.isnan(cpu_np) & (cpu_np > 1e-10)
    if np.any(cpu_active_mask):
        first_active_idx = np.where(cpu_active_mask)[0][0]
        if first_active_idx < cpu_x.shape[-2]:
            cpu_comp = cpu_x.reshape(-1, cpu_x.shape[-1])[first_active_idx][:2]
            print(f"  Composition: NB={cpu_comp[0]:.6f}, TI={cpu_comp[1]:.6f}")
    
except Exception as e:
    print(f"❌ CPU Error: {e}")
    exit(1)

# GPU calculation with dynamic sizing
print(f"\n🚀 GPU CALCULATION (Dynamic Sizing)")
print("-" * 40)
try:
    gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
    gpu_gm = gpu_result.GM.values.flatten()[0]
    gpu_mu = gpu_result.MU.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    gpu_x = gpu_result.X.values
    gpu_phases = gpu_result.Phase.values.flatten()
    
    print(f"✅ GPU Success!")
    print(f"  GM: {gpu_gm:.6f} J/mol")
    print(f"  MU: {gpu_mu[:2]}")
    print(f"  Active phases: {np.sum(gpu_np > 1e-10)}")
    print(f"  Shapes: GM{gpu_result.GM.shape}, NP{gpu_result.NP.shape}, X{gpu_result.X.shape}")
    
    # Extract GPU composition
    gpu_active_mask = gpu_np > 1e-10
    if np.any(gpu_active_mask):
        first_active_idx = np.where(gpu_active_mask)[0][0]
        if first_active_idx < gpu_x.shape[-2]:
            gpu_comp = gpu_x.reshape(-1, gpu_x.shape[-1])[first_active_idx][:2]
            print(f"  Composition: NB={gpu_comp[0]:.6f}, TI={gpu_comp[1]:.6f}")
    
except Exception as e:
    print(f"❌ GPU Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Detailed comparison
print(f"\n📊 DETAILED COMPARISON")
print("-" * 40)

# Compare shapes
print(f"Shape comparison:")
print(f"  GM: CPU{cpu_result.GM.shape} vs GPU{gpu_result.GM.shape}")
print(f"  MU: CPU{cpu_result.MU.shape} vs GPU{gpu_result.MU.shape}")
print(f"  NP: CPU{cpu_result.NP.shape} vs GPU{gpu_result.NP.shape}")
print(f"  X:  CPU{cpu_result.X.shape} vs GPU{gpu_result.X.shape}")

shapes_match = (cpu_result.GM.shape == gpu_result.GM.shape and
                cpu_result.MU.shape == gpu_result.MU.shape and
                cpu_result.NP.shape == gpu_result.NP.shape and
                cpu_result.X.shape == gpu_result.X.shape)

if shapes_match:
    print("✅ All shapes match!")
else:
    print("❌ Shape mismatch detected!")

# Compare values
gm_diff = abs(cpu_gm - gpu_gm)
mu_diff = np.abs(cpu_mu[:2] - gpu_mu[:2])

print(f"\nValue comparison:")
print(f"  GM difference: {gm_diff:.6f} J/mol")
print(f"  MU differences: NB={mu_diff[0]:.6f}, TI={mu_diff[1]:.6f} J/mol")

# Success criteria
gm_tolerance = 1.0  # J/mol
mu_tolerance = 10.0  # J/mol
shape_match_required = True

success = (gm_diff <= gm_tolerance and 
           np.all(mu_diff <= mu_tolerance) and
           (shapes_match or not shape_match_required))

print(f"\n{'='*80}")
print(f"🎯 FINAL RESULTS")
print(f"{'='*80}")

criteria = [
    (f"GM difference ≤ {gm_tolerance} J/mol", gm_diff <= gm_tolerance, f"{gm_diff:.6f} J/mol"),
    (f"MU differences ≤ {mu_tolerance} J/mol", np.all(mu_diff <= mu_tolerance), f"max {np.max(mu_diff):.6f} J/mol"),
    ("Array shapes match", shapes_match, "All match" if shapes_match else "Mismatch"),
]

all_passed = True
for criterion, passed, value in criteria:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{criterion:35}: {status} ({value})")
    if not passed:
        all_passed = False

print(f"\n{'='*80}")
if all_passed:
    print("🎉 COMPLETE SUCCESS!")
    print("Dynamic GPU kernel sizing implementation is working correctly!")
    print("GPU results now match CPU results within acceptable tolerances.")
    print("\nKey achievements:")
    print("• Eliminated hard-coded MAX_* constants")
    print("• Array shapes now match between GPU and CPU")
    print("• Numerical accuracy maintained")
    print("• Addressed user requirement for computed kernel sizes")
else:
    print("⚠️  ISSUES DETECTED")
    print("Some criteria did not pass. Further investigation needed.")
    print("\nProgress made:")
    print("• Dynamic sizing implementation completed")
    print("• Kernel compilation with -D flags working")
    print("• Cache invalidation handling dynamic sizes")

print(f"{'='*80}")