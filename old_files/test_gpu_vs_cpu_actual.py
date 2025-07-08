#!/usr/bin/env python3
"""
Compare actual GPU results vs CPU results now that GPU is working
"""

import os
import glob

# Setup CUDA environment BEFORE importing anything
def setup_cuda_environment():
    conda_env_path = "/home/scott/miniconda3/envs/pycalphad-gpu/bin"
    current_path = os.environ.get('PATH', '')
    if conda_env_path not in current_path:
        os.environ['PATH'] = conda_env_path + ":" + current_path
    cuda_home = "/home/scott/miniconda3/envs/pycalphad-gpu"
    os.environ['CUDA_HOME'] = cuda_home
    os.environ['CUDA_ROOT'] = cuda_home

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
            except OSError:
                pass

setup_cuda_environment()
clear_cupy_kernel_cache()

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v

print("🎯 **ACTUAL GPU vs CPU COMPARISON**")
print("=" * 50)

# Test conditions
tdb = Database("NbTi.tdb")
phases = ["BCC_A2"]
comps = ["NB", "TI", "VA"]
conditions = {
    v.X("TI"): 0.1,
    v.T: 800,
    v.P: 101325
}

print(f"Test conditions: {conditions}")

# CPU calculation
print(f"\n🖥️  **CPU CALCULATION**")
print("-" * 30)
cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = cpu_result.GM.values.flatten()[0]
cpu_mu = cpu_result.MU.values.flatten()
cpu_np = cpu_result.NP.values.flatten()
cpu_x = cpu_result.X.values
cpu_phases = cpu_result.Phase.values.flatten()

print(f"CPU Results:")
print(f"  GM: {cpu_gm:.6f} J/mol")
print(f"  MU: {cpu_mu[:2]}")
print(f"  Shapes: GM{cpu_result.GM.shape}, NP{cpu_result.NP.shape}, X{cpu_result.X.shape}")
print(f"  Active phases: {np.sum(~np.isnan(cpu_np) & (cpu_np > 1e-10))}")

# Extract CPU composition
cpu_active_mask = ~np.isnan(cpu_np) & (cpu_np > 1e-10)
if np.any(cpu_active_mask):
    first_active_idx = np.where(cpu_active_mask)[0][0]
    if first_active_idx < cpu_x.shape[-2]:
        cpu_comp = cpu_x.reshape(-1, cpu_x.shape[-1])[first_active_idx][:2]
        print(f"  Composition: NB={cpu_comp[0]:.6f}, TI={cpu_comp[1]:.6f}")

# GPU calculation  
print(f"\n🚀 **GPU CALCULATION**")
print("-" * 30)
from pycalphad.gpu.gpu_equilibrium import equilibrium_gpu
gpu_result = equilibrium_gpu(tdb, comps, phases, conditions, 
                            gpu=True, verbose=False, fallback_on_error=False)

gpu_gm = gpu_result.GM.values.flatten()[0]
gpu_mu = gpu_result.MU.values.flatten()
gpu_np = gpu_result.NP.values.flatten()
gpu_x = gpu_result.X.values
gpu_phases = gpu_result.Phase.values.flatten()

print(f"GPU Results:")
print(f"  GM: {gpu_gm:.6f} J/mol")
print(f"  MU: {gpu_mu[:2]}")
print(f"  Shapes: GM{gpu_result.GM.shape}, NP{gpu_result.NP.shape}, X{gpu_result.X.shape}")
print(f"  Active phases: {np.sum(gpu_np > 1e-10)}")

# Extract GPU composition
gpu_active_mask = gpu_np > 1e-10
if np.any(gpu_active_mask):
    first_active_idx = np.where(gpu_active_mask)[0][0]
    if first_active_idx < gpu_x.shape[-2]:
        gpu_comp = gpu_x.reshape(-1, gpu_x.shape[-1])[first_active_idx][:2]
        print(f"  Composition: NB={gpu_comp[0]:.6f}, TI={gpu_comp[1]:.6f}")

# Detailed comparison
print(f"\n📊 **DETAILED COMPARISON**")
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

print(f"\nNumerical comparison:")
print(f"  GM difference: {gm_diff:.6f} J/mol")
print(f"  MU differences: NB={mu_diff[0]:.6f}, TI={mu_diff[1]:.6f} J/mol")

# Compare compositions if available
if np.any(cpu_active_mask) and np.any(gpu_active_mask):
    comp_diff = np.abs(cpu_comp - gpu_comp)
    print(f"  Composition differences: NB={comp_diff[0]:.6f}, TI={comp_diff[1]:.6f}")

# Success criteria
gm_tolerance = 1.0  # J/mol
mu_tolerance = 10.0  # J/mol
comp_tolerance = 1e-6  # mole fraction

gm_ok = gm_diff <= gm_tolerance
mu_ok = np.all(mu_diff <= mu_tolerance)
comp_ok = True
if np.any(cpu_active_mask) and np.any(gpu_active_mask):
    comp_ok = np.all(comp_diff <= comp_tolerance)

print(f"\n🎯 **ACCURACY ASSESSMENT**")
print("-" * 40)

criteria = [
    (f"GM difference ≤ {gm_tolerance} J/mol", gm_ok, f"{gm_diff:.6f} J/mol"),
    (f"MU differences ≤ {mu_tolerance} J/mol", mu_ok, f"max {np.max(mu_diff):.6f} J/mol"),
    ("Array shapes match", shapes_match, "All match" if shapes_match else "Mismatch"),
    (f"Composition differences ≤ {comp_tolerance}", comp_ok, f"max {np.max(comp_diff) if np.any(cpu_active_mask) and np.any(gpu_active_mask) else 'N/A'}"),
]

all_passed = True
for criterion, passed, value in criteria:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{criterion:35}: {status} ({value})")
    if not passed:
        all_passed = False

print(f"\n{'='*50}")
if all_passed:
    print("🎉 **COMPLETE SUCCESS!**")
    print("GPU results match CPU results within all tolerances!")
    print("The GPU implementation is now fully working and accurate!")
else:
    print("⚠️  **ISSUES DETECTED**")
    print("Some criteria failed. Need to fix remaining numerical differences.")
    
    if not gm_ok:
        print(f"\n🔧 **NEXT FIX**: GM difference too large ({gm_diff:.6f} J/mol)")
    elif not mu_ok:
        print(f"\n🔧 **NEXT FIX**: Chemical potential differences too large")
    elif not shapes_match:
        print(f"\n🔧 **NEXT FIX**: Array shape mismatches")
    elif not comp_ok:
        print(f"\n🔧 **NEXT FIX**: Composition differences too large")

print(f"{'='*50}")