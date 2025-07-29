#\!/usr/bin/env python
"""Quick test to verify the GPU phase normalization fix."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = ''  # Disable all debug output

from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings('ignore')

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

# Test condition that previously showed 806 J/mol error
conditions = {
    v.T: 900,  
    v.P: 101325, 
    v.N: 1, 
    v.X('AL'): 0.6,
    v.X('CU'): 0.3
}

print(f"Testing AlCu system with phase normalization fix")
print(f"Condition: T=900K, X(AL)=0.6, X(CU)=0.3")
print(f"Expected result: GPU should now match CPU closely")
print("-" * 60)

# CPU calculation (silent)
cpu_result = equilibrium(dbf, comps, phases, conditions, 
                       calc_opts={'pdens': 100}, verbose=False)
cpu_gm = float(cpu_result.GM.values.item())

# GPU calculation (silent)
gpu_result = equilibrium(dbf, comps, phases, conditions, 
                       calc_opts={'pdens': 100}, verbose=False, gpu=True)
gpu_gm = float(gpu_result.GM.values.item())

# Compare
diff = abs(cpu_gm - gpu_gm)

print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Difference: {diff:.2f} J/mol")

if diff < 10:
    print(f"✓ EXCELLENT\! Fix successful - GPU matches CPU within 10 J/mol")
    print(f"  (Previous error was 806 J/mol)")
elif diff < 100:
    print(f"✓ Good improvement - error reduced to under 100 J/mol")
else:
    print(f"⚠️  Large error still remains")

# Check phase counts
cpu_phase_count = len([p for p, amt in zip(cpu_result.Phase.values.squeeze(), 
                                         cpu_result.NP.values.squeeze()) if amt > 0.01])
gpu_phase_count = len([p for p, amt in zip(gpu_result.Phase.values.squeeze(), 
                                         gpu_result.NP.values.squeeze()) if amt > 0.01])

print(f"\nPhase analysis:")
print(f"CPU phases with >1% amount: {cpu_phase_count}")
print(f"GPU phases with >1% amount: {gpu_phase_count}")

if cpu_phase_count == gpu_phase_count:
    print(f"✓ Both CPU and GPU predict same number of stable phases")
else:
    print(f"⚠️  Phase count mismatch - may indicate remaining issues")
EOF < /dev/null
