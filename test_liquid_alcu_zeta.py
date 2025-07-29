#\!/usr/bin/env python
"""Test GPU vs CPU with just LIQUID and ALCU_ZETA phases for Al-Cu-Fe system."""

from pycalphad import Database, equilibrium, variables as v
import warnings

warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']

# Test with just LIQUID (single sublattice) and ALCU_ZETA (multiple sublattices)
phases = ['LIQUID', 'ALCU_ZETA']

print(f"Testing GPU vs CPU with phases: {phases}")
print()

# Test conditions - focus on Al-Cu binary where ALCU_ZETA is stable
test_conditions = [
    {'desc': 'Al-Cu (X_CU=0.33)', 'T': 850, 'X_CU': 0.33, 'X_FE': 0.0},
    {'desc': 'Al-Cu (X_CU=0.17)', 'T': 800, 'X_CU': 0.17, 'X_FE': 0.0},
    {'desc': 'Al-Cu (X_CU=0.5)', 'T': 900, 'X_CU': 0.5, 'X_FE': 0.0},
]

pressure = 101325
passed = 0
failed = 0

for test in test_conditions:
    desc = test['desc']
    temp = test['T']
    x_cu = test['X_CU']
    x_fe = test['X_FE']
    x_al = 1.0 - x_cu - x_fe
    
    print(f"\nTest: {desc}")
    print(f"  X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, X(FE)={x_fe:.2f}, T={temp}K")
    
    conditions = {v.T: temp, v.P: pressure, v.N: 1}
    if x_cu < 0.999:
        conditions[v.X('CU')] = x_cu
    if x_fe < 0.999 and x_cu + x_fe < 0.999:
        conditions[v.X('FE')] = x_fe
    
    try:
        # CPU calculation
        print("  Running CPU...")
        cpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 10}, verbose=False)
        cpu_gm = float(cpu_result.GM.values)
        
        # Extract CPU phases
        cpu_phases = []
        for idx in range(cpu_result.dims['vertex']):
            np_val = float(cpu_result.NP.values[0,0,0,0,idx])
            if np_val > 1e-6:
                phase_name = str(cpu_result.Phase.values[0,0,0,0,idx])
                if phase_name and phase_name \!= '':
                    cpu_phases.append(f"{phase_name}({np_val:.3f})")
        
        # GPU calculation
        print("  Running GPU...")
        gpu_result = equilibrium(dbf, comps, phases, conditions, 
                               calc_opts={'pdens': 10}, verbose=False, gpu=True)
        gpu_gm = float(gpu_result.GM.values)
        
        # Extract GPU phases
        gpu_phases = []
        for idx in range(gpu_result.dims['vertex']):
            np_val = float(gpu_result.NP.values[0,0,0,0,idx])
            if np_val > 1e-6:
                phase_name = str(gpu_result.Phase.values[0,0,0,0,idx])
                if phase_name and phase_name \!= '':
                    gpu_phases.append(f"{phase_name}({np_val:.3f})")
        
        # Compare
        gm_diff = abs(gpu_gm - cpu_gm)
        status = "PASS" if gm_diff < 1.0 else "FAIL"
        
        print(f"  CPU: GM={cpu_gm:.2f} J/mol, phases={','.join(cpu_phases) if cpu_phases else 'NONE'}")
        print(f"  GPU: GM={gpu_gm:.2f} J/mol, phases={','.join(gpu_phases) if gpu_phases else 'NONE'}")
        print(f"  Difference: {gm_diff:.6f} J/mol - {status}")
        
        if status == "PASS":
            passed += 1
        else:
            failed += 1
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        failed += 1

print("\n" + "="*60)
print("SUMMARY:")
print(f"  Total tests: {len(test_conditions)}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Pass rate: {100*passed/len(test_conditions):.1f}%" if len(test_conditions) > 0 else "  No tests")
EOF < /dev/null
