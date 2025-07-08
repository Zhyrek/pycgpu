#!/usr/bin/env python3
"""Check if MU errors correlate with single-phase regions"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v

# Load database
tdb = pyc.Database("NbTi.tdb")
phases = ["LIQUID", "BCC_A2"]
comps = ["NB", "TI", "VA"]

# Test specific compositions that should show single vs two-phase
test_points = [
    # X(TI), expected phase configuration
    (0.00, "single"),  # Pure NB - BCC_A2
    (0.05, "single"),  # Near pure NB - BCC_A2  
    (0.10, "single"),  # Still single phase
    (0.20, "two"),     # Might be two-phase
    (0.50, "two"),     # Likely two-phase
    (0.80, "two"),     # Likely two-phase
    (0.90, "single"),  # Approaching pure TI
    (0.95, "single"),  # Near pure TI - BCC_A2
    (1.00, "single"),  # Pure TI - BCC_A2
]

print("Testing MU errors in single vs two-phase regions at T=500K")
print("=" * 70)
print(f"{'X(TI)':>6} | {'Expected':>10} | {'Actual Phases':>20} | {'MU Error':>12}")
print("-" * 70)

errors_single = []
errors_two = []

for x_ti, expected in test_points:
    # Run calculations
    conditions = {v.X("TI"): x_ti, v.T: 500}
    eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False, calc_opts={'pdens': 2})
    
    # Check phases
    phases_present = []
    for i in range(len(phases)):
        if eq_cpu.NP.values.flat[i] > 1e-6:
            phases_present.append(eq_cpu.Phase.values.flat[i])
    
    phase_config = "single" if len(phases_present) == 1 else "two"
    phase_str = "+".join(phases_present)
    
    # Calculate MU error
    cpu_mu = eq_cpu.MU.values.flatten()
    gpu_mu = eq_gpu.MU.values.flatten()
    mu_error = np.max(np.abs(cpu_mu - gpu_mu))
    
    # Track errors
    if phase_config == "single":
        errors_single.append(mu_error)
    else:
        errors_two.append(mu_error)
    
    # Print result
    check = "✓" if phase_config == expected else "✗"
    status = "ERROR" if mu_error > 1000 else "OK"
    print(f"{x_ti:6.2f} | {expected:>10} | {phase_str:>20} | {mu_error:12.2e} {status} {check}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY:")
print(f"Single-phase regions: {len(errors_single)} tested")
print(f"  Large errors (>1000): {sum(1 for e in errors_single if e > 1000)}")
print(f"  Max error: {max(errors_single):.2e}" if errors_single else "  No data")

print(f"\nTwo-phase regions: {len(errors_two)} tested")
print(f"  Large errors (>1000): {sum(1 for e in errors_two if e > 1000)}")
print(f"  Max error: {max(errors_two):.2e}" if errors_two else "  No data")

if errors_single and errors_two:
    single_has_errors = any(e > 1000 for e in errors_single)
    two_has_errors = any(e > 1000 for e in errors_two)
    
    if single_has_errors and not two_has_errors:
        print("\n✓ CONFIRMED: MU errors occur ONLY in single-phase regions!")
    elif single_has_errors and two_has_errors:
        print("\n⚠ MU errors occur in both single and two-phase regions")
    else:
        print("\n✗ No clear pattern found")