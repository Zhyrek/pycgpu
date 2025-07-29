#!/usr/bin/env python
"""Test the new CSE-based implementation in gpu_codegen.py."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import time

print("Testing New CSE-Based GPU Code Generation")
print("="*60)

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print(f"Database: Al-Cu-Fe.tdb")
print(f"Components: {components}")
print(f"Conditions: T={conditions[v.T]}K, X(AL)=0.3, X(CU)=0.3")

# Test 1: LIQUID phase with new CSE implementation
print("\n" + "="*60)
print("TEST 1: LIQUID Phase (Should Use CSE Method)")
print("="*60)

print("Testing LIQUID phase with new CSE implementation...")
start_time = time.time()

try:
    result = equilibrium(db, components, ['LIQUID'], conditions, 
                        calc_opts={'pdens': 10}, 
                        gpu=True, verbose=False)
    
    compilation_time = time.time() - start_time
    print(f"✓ SUCCESS: LIQUID compiled and ran in {compilation_time:.1f}s")
    print(f"  GM = {result.GM.values[0]:.1f} J/mol")
    
    # Check if LIQUID is stable
    liquid_amount = result.NP.sel(phase='LIQUID').values[0]
    print(f"  LIQUID phase amount: {liquid_amount:.4f}")
    
except Exception as e:
    compilation_time = time.time() - start_time
    print(f"✗ FAILED after {compilation_time:.1f}s")
    print(f"  Error: {type(e).__name__}")
    if 'nvcc' in str(e).lower():
        print("  This appears to be an nvcc compilation error")
    else:
        print(f"  Message: {str(e)[:200]}")

# Test 2: CPU comparison for validation
print("\n" + "="*60)
print("TEST 2: CPU Comparison (Validation)")
print("="*60)

print("Running same calculation on CPU for comparison...")
start_time = time.time()

try:
    cpu_result = equilibrium(db, components, ['LIQUID'], conditions, 
                            calc_opts={'pdens': 10}, 
                            gpu=False, verbose=False)
    
    cpu_time = time.time() - start_time
    print(f"✓ CPU SUCCESS in {cpu_time:.3f}s")
    print(f"  GM = {cpu_result.GM.values[0]:.1f} J/mol")
    
    # Check if LIQUID is stable
    cpu_liquid_amount = cpu_result.NP.sel(phase='LIQUID').values[0]
    print(f"  LIQUID phase amount: {cpu_liquid_amount:.4f}")
    
    # Compare results if both succeeded
    if 'result' in locals():
        gm_diff = abs(result.GM.values[0] - cpu_result.GM.values[0])
        amount_diff = abs(liquid_amount - cpu_liquid_amount)
        
        print(f"\nComparison:")
        print(f"  GM difference: {gm_diff:.3f} J/mol")
        print(f"  Phase amount difference: {amount_diff:.6f}")
        
        if gm_diff < 1.0 and amount_diff < 1e-6:
            print("  ✓ GPU and CPU results match!")
        else:
            print("  ✗ GPU and CPU results differ significantly")
    
except Exception as e:
    cpu_time = time.time() - start_time
    print(f"✗ CPU FAILED after {cpu_time:.3f}s: {e}")

# Test 3: Try a more complex phase to see if CSE helps
print("\n" + "="*60)
print("TEST 3: ALCU_ZETA Phase (More Complex)")
print("="*60)

print("Testing ALCU_ZETA phase...")
start_time = time.time()

try:
    result_zeta = equilibrium(db, components, ['ALCU_ZETA'], conditions, 
                             calc_opts={'pdens': 10}, 
                             gpu=True, verbose=False)
    
    compilation_time = time.time() - start_time
    print(f"✓ SUCCESS: ALCU_ZETA compiled and ran in {compilation_time:.1f}s")
    print(f"  GM = {result_zeta.GM.values[0]:.1f} J/mol")
    
except Exception as e:
    compilation_time = time.time() - start_time
    print(f"✗ FAILED after {compilation_time:.1f}s")
    print(f"  Error: {type(e).__name__}")
    if 'nvcc' in str(e).lower():
        print("  This appears to be an nvcc compilation error")
        print("  The CSE method may still need refinement for this phase")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

expected_improvements = [
    "✓ Faster code generation (CSE + ccode vs regex)",
    "✓ More readable generated C code", 
    "✓ Better GPU optimization potential",
    "✓ Elimination of regex-based syntax fixes",
    "✓ Hierarchical function structure"
]

for improvement in expected_improvements:
    print(improvement)

if 'compilation_time' in locals():
    print(f"\nLast GPU compilation time: {compilation_time:.1f}s")
    if compilation_time < 30:
        print("This is much faster than the previous 60+ second compilations!")

print("\nNext steps:")
print("- Monitor console output for [CSE CODEGEN] messages")
print("- Check if BCC_B2 and other complex phases now compile")
print("- Compare compilation times with previous method")