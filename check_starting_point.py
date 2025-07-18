#!/usr/bin/env python
"""Check starting point differences between CPU and GPU for problematic condition."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
from pycalphad.core.equilibrium import lower_convex_hull
from pycalphad import calculate

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Problematic condition
conditions = {v.X('TI'): 0.5, v.T: 700, v.P: 101325}

print("CHECKING STARTING POINT FOR X(TI)=0.5, T=700K")
print("=" * 60)

# First, get the starting point using the same method as equilibrium()
print("1. Running calculate step (same for both CPU and GPU)...")

try:
    # This is the initial calculate step that both CPU and GPU use
    calculate_result = calculate(dbf, comps, phases, T=700, P=101325)
    
    print(f"   Calculate result shape: {calculate_result.GM.shape}")
    print(f"   Available phases: {[p.name for p in phases]}")
    
    # Run lower_convex_hull to get starting point
    print("2. Running lower_convex_hull...")
    
    # This creates the initial guess that both CPU and GPU start from  
    hull_result = lower_convex_hull(calculate_result, conditions)
    
    print(f"   Hull result type: {type(hull_result)}")
    if hasattr(hull_result, 'Phase'):
        phase_fractions = hull_result.Phase.values.flatten()
        print(f"   Initial phase fractions: {phase_fractions}")
        
        # Check which phases are active
        active_phases = []
        for i, frac in enumerate(phase_fractions):
            if frac > 1e-12:
                phase_name = phases[i].name if i < len(phases) else f"Phase_{i}"
                active_phases.append((i, phase_name, frac))
                print(f"     Phase {i} ({phase_name}): {frac:.6f}")
        
        print(f"   Number of active phases: {len(active_phases)}")
        
        # Check compositions
        if hasattr(hull_result, 'X'):
            compositions = hull_result.X.values
            print(f"   Composition shape: {compositions.shape}")
            print(f"   X(TI) values per phase:")
            for i, (phase_idx, phase_name, frac) in enumerate(active_phases):
                if compositions.size > phase_idx * len(comps) + 1:  # Check if TI composition exists
                    x_ti = compositions.flatten()[phase_idx * len(comps) + 1]  # TI is index 1
                    print(f"     Phase {phase_idx} ({phase_name}): X(TI)={x_ti:.6f}")
        
        # Check site fractions if available
        if hasattr(hull_result, 'Y'):
            site_fractions = hull_result.Y.values
            print(f"   Site fraction shape: {site_fractions.shape}")
            print(f"   Site fractions per phase:")
            for i, (phase_idx, phase_name, frac) in enumerate(active_phases):
                if site_fractions.size > phase_idx * 2:  # Assuming 2 site fractions per phase
                    y1 = site_fractions.flatten()[phase_idx * 2]
                    y2 = site_fractions.flatten()[phase_idx * 2 + 1] if site_fractions.size > phase_idx * 2 + 1 else 0
                    print(f"     Phase {phase_idx} ({phase_name}): Y=[{y1:.6f}, {y2:.6f}]")
                    
                    # Calculate X(TI) from site fractions for BCC_A2
                    if phase_name == 'BCC_A2':
                        x_ti_calc = y2  # For BCC_A2, Y(TI) = X(TI)
                        print(f"       Calculated X(TI) from site fractions: {x_ti_calc:.6f}")
    
    # Now check the difference between two active phases (if any)
    if len(active_phases) >= 2:
        print("\n3. Checking composition differences between phases...")
        phase1_idx, phase1_name, phase1_frac = active_phases[0]
        phase2_idx, phase2_name, phase2_frac = active_phases[1]
        
        if hasattr(hull_result, 'X'):
            comps_flat = hull_result.X.values.flatten()
            if len(comps_flat) > max(phase1_idx, phase2_idx) * len(comps) + 2:
                # Get X(TI) for both phases
                x_ti_phase1 = comps_flat[phase1_idx * len(comps) + 1]
                x_ti_phase2 = comps_flat[phase2_idx * len(comps) + 1]
                
                composition_diff = abs(x_ti_phase1 - x_ti_phase2)
                consolidation_threshold = 1e-4
                
                print(f"   Phase {phase1_idx} X(TI): {x_ti_phase1:.6f}")
                print(f"   Phase {phase2_idx} X(TI): {x_ti_phase2:.6f}")
                print(f"   Composition difference: {composition_diff:.6f}")
                print(f"   Consolidation threshold: {consolidation_threshold:.6f}")
                print(f"   Should consolidate: {'YES' if composition_diff < consolidation_threshold else 'NO'}")
                
                if composition_diff < consolidation_threshold:
                    print("   ⚠️  PROBLEM: Starting point already below consolidation threshold!")
                    print("   This means GPU will consolidate immediately in iteration 1")
                else:
                    print("   ✓ Starting point looks good, phases should remain separate initially")
    
    else:
        print(f"   Only {len(active_phases)} active phases found - no immiscibility gap in starting point")
    
except Exception as e:
    print(f"Error in starting point analysis: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)