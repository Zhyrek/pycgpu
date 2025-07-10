#!/usr/bin/env python3
"""Analyze GPU step size calculation"""

print("=== GPU Step Size Analysis ===")
print("\nFrom GPU debug output at iteration 0:")
print("- Initial phase amounts: [0.340986, 0.659014]")
print("- Equilibrium solution: [-51532.35, -47108.95, 154.95, -156.75]")
print("- Step size limiter: 0.00420414")

# The solution vector is [δμ₀, δμ₁, δNP₀, δNP₁]
delta_mu = [-51532.35, -47108.95]
delta_NP = [154.95, -156.75]

print(f"\nPhase amount changes (δNP): {delta_NP}")
print(f"Chemical potential changes (δμ): {delta_mu}")

# Apply step size
step_size = 0.00420414
phase_amt_initial = [0.340986, 0.659014]

print(f"\nWith step size {step_size}:")
for i in range(2):
    change = step_size * delta_NP[i]
    new_amt = phase_amt_initial[i] + change
    print(f"  Phase {i}: {phase_amt_initial[i]:.6f} + {step_size:.6f} * {delta_NP[i]:.2f} = {phase_amt_initial[i]:.6f} + {change:.6f} = {new_amt:.6f}")

print("\n=== The Problem ===")
print("The GPU debug shows phase 1 amount becomes 1.11e-16 after advance_state")
print("But our calculation shows it should be 0.0 (0.659 - 0.659)")
print("\nThe step size limiter is calculated to prevent phase amounts going negative")

# Check step size calculation
print("\n=== Step Size Limiter Calculation ===")
print("For phase going negative, step size = -phase_amt / delta_NP")
for i in range(2):
    if delta_NP[i] < 0:  # Would decrease
        max_step = -phase_amt_initial[i] / delta_NP[i]
        print(f"Phase {i}: max_step = -{phase_amt_initial[i]:.6f} / {delta_NP[i]:.2f} = {max_step:.6f}")

print("\nThe step size 0.00420414 ≈ 0.659014 / 156.75")
print("This is exactly the amount that makes phase 1 go to zero!")

print("\n=== Root Cause ===")
print("1. The large δNP values come from incorrect mass jacobian")
print("2. Step size is limited to prevent negative amounts")
print("3. Phase 1 goes to ~0 and gets removed")
print("4. With only 1 phase, system can't satisfy X(TI)=0.4 constraint")