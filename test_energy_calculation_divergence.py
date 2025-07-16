#!/usr/bin/env python
"""Test energy calculation divergence between CPU and GPU for identical inputs."""

from pycalphad import Database, calculate
from pycalphad.model import Model
import numpy as np

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Set up conditions
T = 1000.0
P = 101325.0

# Test specific site fractions that appear in the divergent case
test_cases = [
    {'phase': 'BCC_A2', 'Y_NB': 0.995151, 'Y_TI': 0.004849},  # Phase 0 in divergent case
    {'phase': 'BCC_A2', 'Y_NB': 0.983051, 'Y_TI': 0.016949},  # Phase 1 in divergent case
    {'phase': 'BCC_A2', 'Y_NB': 0.990000, 'Y_TI': 0.010000},  # Target composition
]

print("="*80)
print("TESTING ENERGY CALCULATIONS WITH IDENTICAL INPUTS")
print("="*80)

for i, test in enumerate(test_cases):
    phase = test['phase']
    y_nb = test['Y_NB']
    y_ti = test['Y_TI']
    
    print(f"\nTest case {i+1}: {phase} with Y(NB)={y_nb:.6f}, Y(TI)={y_ti:.6f}")
    print("-" * 60)
    
    # Calculate using CPU
    conditions_cpu = {
        'T': T,
        'P': P,
        f'Y({phase},0,NB)': y_nb,
        f'Y({phase},0,TI)': y_ti
    }
    
    # CPU calculation
    calc_cpu = calculate(tdb, comps, phase, conditions=conditions_cpu, model=Model(tdb, comps, phase), output='GM')
    cpu_gm = calc_cpu.GM.values.item()
    print(f"CPU Energy: {cpu_gm:.15e} J/mol")
    
    # Now test if we can reproduce this with the Model directly
    mod = Model(tdb, comps, phase)
    
    # Get the energy using the model's energy function
    # The order is typically [T, Y1, Y2, ...] for the model
    dof = np.array([T, y_nb, y_ti])
    
    # Access the model's AST to evaluate energy
    from pycalphad.core.utils import unpack_components
    from symengine import symbols
    
    # Get the symbolic expression
    energy_expr = mod.GM
    
    # Get the variables in the correct order
    state_vars = [v.T]  # Temperature
    site_fracs = []
    for idx in range(mod.site_fractions.shape[0]):
        for comp in ['NB', 'TI']:
            v = mod.site_fractions[idx, mod.constituents[idx].index(comp)]
            site_fracs.append(v)
    
    # Create substitution dictionary
    subs_dict = {state_vars[0]: T}
    subs_dict[site_fracs[0]] = y_nb
    subs_dict[site_fracs[1]] = y_ti
    
    # Evaluate
    energy_val = float(energy_expr.subs(subs_dict))
    print(f"Model Direct Energy: {energy_val:.15e} J/mol")
    print(f"Difference: {abs(cpu_gm - energy_val):.15e}")
    
    # Now let's trace what the GPU would calculate
    # The GPU uses the same model, so the energy should be identical
    # if given the same inputs
    
print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)
print("If the energies match when calculated directly with the same inputs,")
print("then the divergence must come from:")
print("1. Different site fraction values being passed to the energy function")
print("2. Different handling of the DOF array (workspace vs model format)")
print("3. Numerical precision in the energy function evaluation")

# Let's also check the Hessian calculation
print("\n" + "="*80)
print("CHECKING HESSIAN CALCULATION")
print("="*80)

# Test Hessian for the first test case
from symengine import symbols, diff
y_nb_sym = symbols('Y_BCC_A2_0_NB')
y_ti_sym = symbols('Y_BCC_A2_0_TI')
t_sym = symbols('T')

# Get the model
mod = Model(tdb, comps, 'BCC_A2')
energy_expr = mod.GM

# Calculate Hessian symbolically
print("\nCalculating Hessian elements:")
h_nb_nb = diff(diff(energy_expr, y_nb_sym), y_nb_sym)
h_nb_ti = diff(diff(energy_expr, y_nb_sym), y_ti_sym)
h_ti_ti = diff(diff(energy_expr, y_ti_sym), y_ti_sym)

# Evaluate at test point
subs_dict = {t_sym: 1000.0, y_nb_sym: 0.995151, y_ti_sym: 0.004849}
h_nb_nb_val = float(h_nb_nb.subs(subs_dict))
h_nb_ti_val = float(h_nb_ti.subs(subs_dict))
h_ti_ti_val = float(h_ti_ti.subs(subs_dict))

print(f"H[NB,NB] = {h_nb_nb_val:.6e}")
print(f"H[NB,TI] = {h_nb_ti_val:.6e}")
print(f"H[TI,TI] = {h_ti_ti_val:.6e}")

print("\nThese should match the CPU values:")
print("  CPU Row 3: 8.355014e+03 1.304530e+04")
print("  CPU Row 4: 1.304530e+04 1.714680e+06")