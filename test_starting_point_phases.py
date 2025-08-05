#!/usr/bin/env python
"""Test to capture starting point phase selection in CPU vs GPU paths."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Monkey patch to capture starting point results
import pycalphad.core.workspace as ws_module
import pycalphad.gpu.gpu_equilibrium as gpu_module

captured_starting_points = {'cpu': None, 'gpu': None}

# Patch CPU starting_point call
original_ws_init = ws_module.Workspace.__init__

def patched_ws_init(self, *args, **kwargs):
    original_ws_init(self, *args, **kwargs)
    # Override the original eq with a capturing version
    if hasattr(self, 'eq') and self.eq is not None:
        captured_starting_points['cpu'] = {
            'phases': self.eq.Phase.values if hasattr(self.eq.Phase, 'values') else self.eq.Phase,
            'NP': self.eq.NP.values if hasattr(self.eq.NP, 'values') else self.eq.NP,
            'GM': self.eq.GM.values if hasattr(self.eq.GM, 'values') else self.eq.GM,
            'MU': self.eq.MU.values if hasattr(self.eq.MU, 'values') else self.eq.MU
        }

ws_module.Workspace.__init__ = patched_ws_init

# Patch GPU _prepare_gpu_data to capture starting point
original_prepare_gpu_data = gpu_module._prepare_gpu_data

def patched_prepare_gpu_data(wks_obj, properties, dynamic_sizes=None, force_gpu=True):
    # Capture properties which contain starting point
    if properties is not None:
        captured_starting_points['gpu'] = {
            'phases': properties.Phase.values if hasattr(properties.Phase, 'values') else properties.Phase,
            'NP': properties.NP.values if hasattr(properties.NP, 'values') else properties.NP,
            'GM': properties.GM.values if hasattr(properties.GM, 'values') else properties.GM,
            'MU': properties.MU.values if hasattr(properties.MU, 'values') else properties.MU
        }
    return original_prepare_gpu_data(wks_obj, properties, dynamic_sizes, force_gpu)

gpu_module._prepare_gpu_data = patched_prepare_gpu_data

def test_starting_points():
    """Compare starting points between CPU and GPU."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("STARTING POINT COMPARISON")
    print("="*70)
    
    # Reset captures
    captured_starting_points['cpu'] = None
    captured_starting_points['gpu'] = None
    
    # CPU calculation
    print("\n--- CPU CALCULATION ---")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    
    if captured_starting_points['cpu'] is not None:
        print("\nCPU Starting Point:")
        cpu_phases = captured_starting_points['cpu']['phases']
        cpu_np = captured_starting_points['cpu']['NP']
        cpu_gm = captured_starting_points['cpu']['GM']
        
        # Find active phases in starting point
        if cpu_phases.ndim > 1:
            # Multi-dimensional, take first condition
            phases_1d = cpu_phases.flatten()[:10]  # First 10 phases
            np_1d = cpu_np.flatten()[:10]
        else:
            phases_1d = cpu_phases
            np_1d = cpu_np
            
        print(f"  GM: {cpu_gm.flatten()[0]:.6f}")
        print("  Active phases:")
        for phase, amount in zip(phases_1d, np_1d):
            if phase != '' and amount > 1e-8:
                print(f"    {phase}: {amount:.6f}")
    
    # GPU calculation
    print("\n--- GPU CALCULATION ---")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    
    if captured_starting_points['gpu'] is not None:
        print("\nGPU Starting Point:")
        gpu_phases = captured_starting_points['gpu']['phases']
        gpu_np = captured_starting_points['gpu']['NP']
        gpu_gm = captured_starting_points['gpu']['GM']
        
        # Find active phases in starting point
        if gpu_phases.ndim > 1:
            # Multi-dimensional, take first condition
            phases_1d = gpu_phases.flatten()[:10]  # First 10 phases
            np_1d = gpu_np.flatten()[:10]
        else:
            phases_1d = gpu_phases
            np_1d = gpu_np
            
        print(f"  GM: {gpu_gm.flatten()[0]:.6f}")
        print("  Active phases:")
        for phase, amount in zip(phases_1d, np_1d):
            if phase != '' and amount > 1e-8:
                print(f"    {phase}: {amount:.6f}")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    if captured_starting_points['cpu'] is not None and captured_starting_points['gpu'] is not None:
        cpu_active = set()
        gpu_active = set()
        
        # Extract active phases from CPU starting point
        cpu_phases_flat = captured_starting_points['cpu']['phases'].flatten()
        cpu_np_flat = captured_starting_points['cpu']['NP'].flatten()
        for phase, amount in zip(cpu_phases_flat[:10], cpu_np_flat[:10]):
            if phase != '' and amount > 1e-8:
                cpu_active.add(phase)
        
        # Extract active phases from GPU starting point  
        gpu_phases_flat = captured_starting_points['gpu']['phases'].flatten()
        gpu_np_flat = captured_starting_points['gpu']['NP'].flatten()
        for phase, amount in zip(gpu_phases_flat[:10], gpu_np_flat[:10]):
            if phase != '' and amount > 1e-8:
                gpu_active.add(phase)
        
        if cpu_active != gpu_active:
            print("❌ Different starting points!")
            print(f"  CPU phases: {cpu_active}")
            print(f"  GPU phases: {gpu_active}")
            print("\nThis explains why the equilibrium results differ.")
            print("The starting_point() function itself is identical,")
            print("so the difference must come from:")
            print("  1. The grid passed to starting_point()")
            print("  2. Numerical differences in the convex hull calculation")
            print("  3. Phase record factory differences")
        else:
            print("✅ Same starting points")

if __name__ == "__main__":
    test_starting_points()