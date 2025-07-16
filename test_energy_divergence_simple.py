#!/usr/bin/env python
"""Simple test to find energy calculation divergence."""

from pycalphad import Database, calculate
from pycalphad.model import Model
import numpy as np

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Test conditions
T = 1000.0
P = 101325.0

print("="*80)
print("TESTING ENERGY CALCULATION AT SPECIFIC SITE FRACTIONS")
print("="*80)

# Test the site fractions from the divergent case
test_site_fractions = [
    (0.995150990801529, 0.004849009198471),  # Phase 0 initial
    (0.983050847457627, 0.016949152542373),  # Phase 1 initial
]

for i, (y_nb, y_ti) in enumerate(test_site_fractions):
    print(f"\nTest {i+1}: Y(NB)={y_nb:.15f}, Y(TI)={y_ti:.15f}")
    
    # Calculate energy using calculate function
    calc_result = calculate(tdb, comps, 'BCC_A2', 
                           T=T, P=P, 
                           Y={'BCC_A2': [[y_nb, y_ti]]},
                           output='GM')
    
    energy = float(calc_result.GM.values)
    print(f"  Energy from calculate: {energy:.15e} J/mol")
    
    # Also get it from the model directly
    mod = Model(tdb, comps, 'BCC_A2')
    # The model expects variables in a specific order
    # For BCC_A2, it's typically [T, Y(BCC_A2,0,NB), Y(BCC_A2,0,TI)]
    from pycalphad.variables import T as Tvar
    
    # Get the ordering of variables
    vars_in_order = [Tvar]
    phase_name = 'BCC_A2'
    for subl_idx in range(mod.site_fractions.shape[0]):
        for const in sorted(mod.constituents[subl_idx]):
            if const != 'VA':
                vars_in_order.append(mod.site_fractions[subl_idx, mod.constituents[subl_idx].index(const)])
    
    # Create DOF array
    dof = np.array([T, y_nb, y_ti])
    
    # Use the compiled function if available
    if hasattr(mod, '_obj_parameters') and mod._obj_parameters is not None:
        from pycalphad.core.compiled_model import CompiledModel
        cmod = CompiledModel(mod, comps)
        energy_compiled = cmod.GM(dof)
        print(f"  Energy from compiled model: {energy_compiled:.15e} J/mol")
    
    # Check symbolic evaluation
    import symengine as se
    subs_dict = {}
    for j, var in enumerate(vars_in_order):
        subs_dict[var] = dof[j]
    energy_symbolic = float(mod.GM.subs(subs_dict))
    print(f"  Energy from symbolic evaluation: {energy_symbolic:.15e} J/mol")

# Now let's add specific debug to trace where CPU and GPU might differ
print("\n" + "="*80)
print("ADDING DEBUG TRACES")
print("="*80)

# First, let's verify the energies match for the initial conditions
from pycalphad import equilibrium

conditions = {
    'T': 1000,
    'P': 101325,
    'X(TI)': 0.01
}

print("\nRunning equilibrium with debug output to trace divergence...")
print("\nKey values to watch:")
print("1. Initial phase energies")
print("2. Initial site fractions") 
print("3. Whether phases get consolidated")
print("4. Matrix entries and RHS values")

# We need to add more specific debug output to the actual calculation
# Let me check the specific consolidation logic