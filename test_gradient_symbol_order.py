#!/usr/bin/env python
"""Test the gradient symbol ordering for ternary systems."""

from pycalphad import Database, Model, variables as v
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.workspace import Workspace
import numpy as np

def test_gradient_symbols():
    """Test the symbol ordering that determines gradient generation."""
    
    # Load Al-Cu-Fe database
    tdb = Database('Al-Cu-Fe.tdb')
    
    # Create conditions for ternary system
    conditions = {v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2}
    
    # Focus on LIQUID phase
    phase = tdb.phases['LIQUID']
    
    # Create Model for LIQUID phase
    model = Model(tdb, ['AL', 'CU', 'FE', 'VA'], 'LIQUID')
    
    print("TERNARY SYSTEM GRADIENT SYMBOL ANALYSIS")
    print("="*50)
    print(f"Phase: {model.phase_name}")
    print(f"Model state variables: {model.state_variables}")
    print(f"Model site fractions: {model.site_fractions}")
    
    # Count degrees of freedom
    site_fractions = model.site_fractions
    phase_dof = len(site_fractions)
    print(f"Phase DOF (site fractions): {phase_dof}")
    print(f"Site fractions: {[str(sf) for sf in site_fractions]}")
    
    # Create workspace to mimic GPU codegen environment
    components = ['AL', 'CU', 'FE', 'VA']
    wks_obj = Workspace(components=components, phases=['LIQUID'], conditions=conditions,
                       models={'LIQUID': model}, phase_record_factory=None,
                       verbose=True)
    
    # Now test the get_ordered_symbols_for_diff function
    import sys
    sys.path.append('/mnt/c/users/scott/Documents/pycalphad')
    from pycalphad.gpu.gpu_codegen import get_ordered_symbols_for_diff
    
    ordered_symbols = get_ordered_symbols_for_diff(model, wks_obj, verbose=True)
    
    print(f"\nOrdered symbols for differentiation:")
    for i, sym in enumerate(ordered_symbols):
        print(f"  [{i}]: {sym} (type: {type(sym).__name__})")
    
    print(f"\nTotal differentiation symbols: {len(ordered_symbols)}")
    
    print("\nEXPECTED vs ACTUAL:")
    print("Expected for ternary LIQUID phase:")
    print("  [0]: T (temperature)")
    print("  [1]: LIQUID0AL (site fraction AL)")
    print("  [2]: LIQUID0CU (site fraction CU)")  
    print("  [3]: LIQUID0FE (site fraction FE)")
    print("Expected total: 4 symbols (T + 3 site fractions)")
    
    print(f"\nActual symbols: {len(ordered_symbols)}")
    
    # Analyze the impact on gradient generation
    print("\n" + "="*60)
    print("GRADIENT GENERATION ANALYSIS:")
    print("="*60)
    
    print("For GPU gradient calculation:")
    print(f"  - formulagrad should output {len(ordered_symbols)} gradient values")
    print(f"  - These get mapped to grad[2], grad[3], grad[4], grad[5], ...")
    print(f"  - c_G calculation uses grad[num_statevars + j] for j in [0, phase_dof)")
    print(f"  - With num_statevars=3, phase_dof=3:")
    print(f"    c_G[0] uses grad[3]")
    print(f"    c_G[1] uses grad[4]") 
    print(f"    c_G[2] uses grad[5]")
    
    if len(ordered_symbols) < 4:
        print("\n*** POTENTIAL BUG DETECTED ***")
        print(f"Only {len(ordered_symbols)} symbols for differentiation!")
        print("This means formulagrad only outputs 3 values instead of 4.")
        print("grad[5] will be uninitialized when c_G[2] tries to use it!")
        print("This explains the wrong c_G values and 239 J/mol divergence.")
    else:
        print("\n✓ Correct number of differentiation symbols")
    
    # Also check if VA is being included/excluded properly
    non_va_fractions = [sf for sf in site_fractions if 'VA' not in str(sf)]
    print(f"\nNon-VA site fractions: {len(non_va_fractions)}")
    print(f"Site fractions with VA: {[str(sf) for sf in site_fractions if 'VA' in str(sf)]}")
    
    return ordered_symbols

if __name__ == "__main__":
    try:
        symbols = test_gradient_symbols()
        print(f"\nFinal analysis: {len(symbols)} differentiation symbols found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()