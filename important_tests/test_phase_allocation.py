#!/usr/bin/env python
"""Test to verify MAX_PHASES allocation and phase detection in GPU code."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes
from pycalphad.core.workspace import Workspace
import warnings
warnings.filterwarnings("ignore")

def test_phase_allocation():
    """Test MAX_PHASES allocation for Al-Cu-Fe system."""
    
    # Load the Al-Cu-Fe database
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # All 21 phases from the Al-Cu-Fe system
    phases = ['AL13FE4', 'AL2FE', 'AL5FE2', 'AL5FE4', 'ALCU_DELTA', 'ALCU_EPSILON', 
              'ALCU_ETA', 'ALCU_PRIME', 'ALCU_THETA', 'ALCU_ZETA', 'BCC_A2', 'BCC_B2', 
              'FCC_A1', 'GAMMA_D83', 'GAMMA_H', 'L12', 'LIQUID', 'TS01T1', 'TS01T2', 
              'TS01T3', 'TS01TI']
    
    print("=" * 80)
    print("Testing MAX_PHASES allocation for Al-Cu-Fe system")
    print("=" * 80)
    print(f"\nTotal phases in system: {len(phases)}")
    print(f"Phase list: {', '.join(phases)}")
    
    # Create a workspace to compute dynamic sizes
    conditions = {
        v.X('AL'): 0.40,
        v.X('CU'): 0.25,
        v.T: 700,
        v.P: 101325
    }
    
    # Create workspace
    from pycalphad.core.equilibrium import _adjust_conditions
    from pycalphad.core.light_dataset import LightDataset
    from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    
    conds_adj = _adjust_conditions(conditions)
    
    # Create models
    from pycalphad.model import Model
    models = {}
    for phase in phases:
        models[phase] = Model(dbf, comps, phase)
    
    # Create phase record factory
    phase_record_factory = PhaseRecordFactory(dbf, comps, [v.T, v.P], models)
    
    # Create minimal workspace-like object
    class MinimalWorkspace:
        def __init__(self):
            self.components = comps
            self.phases = phases
            self.models = models
            self.phase_record_factory = phase_record_factory
    
    wks = MinimalWorkspace()
    
    # Compute dynamic kernel sizes
    dynamic_sizes = compute_dynamic_kernel_sizes(wks)
    
    print(f"\n--- Dynamic Kernel Sizes ---")
    print(f"MAX_COMPONENTS: {dynamic_sizes['MAX_COMPONENTS']} (actual: {len(comps)})")
    print(f"MAX_PHASES: {dynamic_sizes['MAX_PHASES']} (actual: {len(phases)})")
    print(f"MAX_STATEVARS: {dynamic_sizes['MAX_STATEVARS']}")
    print(f"MAX_DOF_PER_PHASE: {dynamic_sizes['MAX_DOF_PER_PHASE']}")
    print(f"MAX_INTERNAL_CONSTRAINTS: {dynamic_sizes['MAX_INTERNAL_CONSTRAINTS']}")
    print(f"MAX_FIXED_MOLE_FRACTION_CONDITIONS: {dynamic_sizes['MAX_FIXED_MOLE_FRACTION_CONDITIONS']}")
    
    # Check if MAX_PHASES is sufficient
    if dynamic_sizes['MAX_PHASES'] >= len(phases):
        print(f"\n✓ MAX_PHASES ({dynamic_sizes['MAX_PHASES']}) is SUFFICIENT for {len(phases)} phases")
    else:
        print(f"\n✗ MAX_PHASES ({dynamic_sizes['MAX_PHASES']}) is INSUFFICIENT for {len(phases)} phases!")
        print(f"  Need at least {len(phases)} but only have {dynamic_sizes['MAX_PHASES']}")
    
    # Now test a specific equilibrium calculation
    print("\n--- Testing Equilibrium Calculation ---")
    print("Condition: X(AL)=0.40, X(CU)=0.25, X(FE)=0.35, T=700K")
    
    # Run CPU calculation
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    
    # Get stable phases for CPU
    cpu_phases = []
    for phase in phases:
        if phase in cpu_result.Phase.values:
            idx = np.where(cpu_result.Phase.values == phase)[0]
            if len(idx) > 0:
                np_val = cpu_result.NP.values.flat[idx[0]]
                if np_val > 1e-6:
                    cpu_phases.append(phase)
    
    print(f"CPU stable phases: {', '.join(cpu_phases)}")
    
    # Check if AL13FE4 is in the stable phases
    if 'AL13FE4' in cpu_phases:
        print("✓ AL13FE4 is detected by CPU as stable")
        
        # Find its phase fraction
        idx = np.where(cpu_result.Phase.values == 'AL13FE4')[0]
        if len(idx) > 0:
            np_val = cpu_result.NP.values.flat[idx[0]]
            print(f"  AL13FE4 phase fraction: {np_val:.6f}")
    else:
        print("✗ AL13FE4 is NOT detected by CPU as stable")
    
    # Run GPU calculation
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    
    # Get stable phases for GPU
    gpu_phases = []
    for phase in phases:
        if phase in gpu_result.Phase.values:
            idx = np.where(gpu_result.Phase.values == phase)[0]
            if len(idx) > 0:
                np_val = gpu_result.NP.values.flat[idx[0]]
                if np_val > 1e-6:
                    gpu_phases.append(phase)
    
    print(f"GPU stable phases: {', '.join(gpu_phases)}")
    
    # Check if AL13FE4 is in the stable phases
    if 'AL13FE4' in gpu_phases:
        print("✓ AL13FE4 is detected by GPU as stable")
        
        # Find its phase fraction
        idx = np.where(gpu_result.Phase.values == 'AL13FE4')[0]
        if len(idx) > 0:
            np_val = gpu_result.NP.values.flat[idx[0]]
            print(f"  AL13FE4 phase fraction: {np_val:.6f}")
    else:
        print("✗ AL13FE4 is NOT detected by GPU as stable")
        print("  This is the issue - GPU is missing AL13FE4!")
    
    # Compare phase detection
    print("\n--- Phase Detection Comparison ---")
    missing_in_gpu = set(cpu_phases) - set(gpu_phases)
    extra_in_gpu = set(gpu_phases) - set(cpu_phases)
    
    if missing_in_gpu:
        print(f"✗ Phases missing in GPU: {', '.join(missing_in_gpu)}")
    if extra_in_gpu:
        print(f"✗ Extra phases in GPU: {', '.join(extra_in_gpu)}")
    if not missing_in_gpu and not extra_in_gpu:
        print("✓ Phase detection matches between CPU and GPU")

if __name__ == "__main__":
    test_phase_allocation()