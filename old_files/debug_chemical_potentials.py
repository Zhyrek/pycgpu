#!/usr/bin/env python3
"""
Debug chemical potential differences between GPU and CPU
"""

import os
import glob

# Setup CUDA environment
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

print("🔍 **DEBUGGING CHEMICAL POTENTIAL DIFFERENCES**")
print("=" * 60)

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
print(f"Expected: Single BCC_A2 phase with NB=0.9, TI=0.1")

# CPU calculation with detailed output
print(f"\n📊 **CPU DETAILED ANALYSIS**")
print("-" * 40)

cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = cpu_result.GM.values.flatten()[0]
cpu_mu = cpu_result.MU.values.flatten()
cpu_np = cpu_result.NP.values.flatten()
cpu_x = cpu_result.X.values

print(f"CPU Results:")
print(f"  GM: {cpu_gm:.10f} J/mol")
print(f"  MU[NB]: {cpu_mu[0]:.10f} J/mol")
print(f"  MU[TI]: {cpu_mu[1]:.10f} J/mol")
print(f"  Active phases: {np.sum(~np.isnan(cpu_np) & (cpu_np > 1e-10))}")
print(f"  Phase amounts: {cpu_np[~np.isnan(cpu_np) & (cpu_np > 1e-10)]}")

# Extract CPU composition details
cpu_active_mask = ~np.isnan(cpu_np) & (cpu_np > 1e-10)
if np.any(cpu_active_mask):
    first_active_idx = np.where(cpu_active_mask)[0][0]
    if first_active_idx < cpu_x.shape[-2]:
        cpu_comp = cpu_x.reshape(-1, cpu_x.shape[-1])[first_active_idx]
        comp_str = ""
        for i, comp_name in enumerate(["NB", "TI", "VA"]):
            if i < len(cpu_comp):
                comp_str += f"{comp_name}={cpu_comp[i]:.10f}, "
        print(f"  Composition: {comp_str.rstrip(', ')}")

# GPU calculation with detailed output
print(f"\n🚀 **GPU DETAILED ANALYSIS**")
print("-" * 40)

from pycalphad.gpu.gpu_equilibrium import equilibrium_gpu
gpu_result = equilibrium_gpu(tdb, comps, phases, conditions, 
                            gpu=True, verbose=False, fallback_on_error=False)

gpu_gm = gpu_result.GM.values.flatten()[0]
gpu_mu = gpu_result.MU.values.flatten()
gpu_np = gpu_result.NP.values.flatten()
gpu_x = gpu_result.X.values

print(f"GPU Results:")
print(f"  GM: {gpu_gm:.10f} J/mol")
print(f"  MU[NB]: {gpu_mu[0]:.10f} J/mol")
print(f"  MU[TI]: {gpu_mu[1]:.10f} J/mol")
print(f"  Active phases: {np.sum(gpu_np > 1e-10)}")
print(f"  Phase amounts: {gpu_np[gpu_np > 1e-10]}")

# Extract GPU composition details
gpu_active_mask = gpu_np > 1e-10
if np.any(gpu_active_mask):
    first_active_idx = np.where(gpu_active_mask)[0][0]
    if first_active_idx < gpu_x.shape[-2]:
        gpu_comp = gpu_x.reshape(-1, gpu_x.shape[-1])[first_active_idx]
        comp_str = ""
        for i, comp_name in enumerate(["NB", "TI", "VA"]):
            if i < len(gpu_comp):
                comp_str += f"{comp_name}={gpu_comp[i]:.10f}, "
        print(f"  Composition: {comp_str.rstrip(', ')}")

# Detailed difference analysis
print(f"\n🔍 **DIFFERENCE ANALYSIS**")
print("-" * 40)

gm_diff = abs(cpu_gm - gpu_gm)
mu_diff_nb = abs(cpu_mu[0] - gpu_mu[0])
mu_diff_ti = abs(cpu_mu[1] - gpu_mu[1])

print(f"Absolute differences:")
print(f"  GM: {gm_diff:.6f} J/mol ({gm_diff/abs(cpu_gm)*100:.4f}% relative)")
print(f"  MU[NB]: {mu_diff_nb:.6f} J/mol ({mu_diff_nb/abs(cpu_mu[0])*100:.4f}% relative)")
print(f"  MU[TI]: {mu_diff_ti:.6f} J/mol ({mu_diff_ti/abs(cpu_mu[1])*100:.4f}% relative)")

# Check if compositions are identical
if np.any(cpu_active_mask) and np.any(gpu_active_mask):
    # Handle different composition array sizes
    min_len = min(len(cpu_comp), len(gpu_comp))
    comp_diff = np.abs(cpu_comp[:min_len] - gpu_comp[:min_len])
    print(f"Composition differences:")
    for i, comp_name in enumerate(["NB", "TI", "VA"]):
        if i < len(comp_diff):
            print(f"  {comp_name}: {comp_diff[i]:.2e}")
        elif i < len(cpu_comp):
            print(f"  {comp_name}: CPU only - {cpu_comp[i]:.2e}")
        elif i < len(gpu_comp):
            print(f"  {comp_name}: GPU only - {gpu_comp[i]:.2e}")

# Analysis of potential causes
print(f"\n🎯 **ROOT CAUSE ANALYSIS**")
print("-" * 40)

print(f"Observations:")
print(f"1. GM difference is small ({gm_diff:.3f} J/mol) - GPU energy calculation is accurate")
print(f"2. Compositions are identical - GPU composition handling is correct")
if mu_diff_ti > 50:
    print(f"3. Large TI chemical potential difference ({mu_diff_ti:.1f} J/mol) suggests:")
    print(f"   - GPU may be using different chemical potential calculation method")
    print(f"   - GPU kernel may not be converging to same equilibrium state")
    print(f"   - GPU may be using starting_point MU instead of converged MU")
else:
    print(f"3. Chemical potential differences are reasonable")

# Tolerance assessment
print(f"\n📋 **TOLERANCE ASSESSMENT**")
print("-" * 40)

tolerances = {
    "GM": (1.0, gm_diff),
    "MU_NB": (10.0, mu_diff_nb), 
    "MU_TI": (10.0, mu_diff_ti),
    "Composition": (1e-6, np.max(comp_diff) if np.any(cpu_active_mask) and np.any(gpu_active_mask) else 0)
}

all_pass = True
for param, (tolerance, actual) in tolerances.items():
    status = "✅ PASS" if actual <= tolerance else "❌ FAIL"
    print(f"{param:12}: {status} ({actual:.6f} ≤ {tolerance})")
    if actual > tolerance:
        all_pass = False

print(f"\n{'='*60}")
if all_pass:
    print("🎉 **ALL TOLERANCES MET** - GPU implementation is accurate!")
else:
    print("⚠️  **TOLERANCES EXCEEDED** - Need to investigate further")
    print("\n📝 **NEXT STEPS:**")
    if mu_diff_ti > 10:
        print("1. Check if GPU uses starting_point MU vs converged MU")
        print("2. Verify GPU equilibrium convergence criteria")
        print("3. Compare GPU kernel chemical potential calculation vs CPU")

print("=" * 60)