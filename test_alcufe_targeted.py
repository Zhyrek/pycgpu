#!/usr/bin/env python
"""
Targeted test of Al-Cu-Fe system focusing on compositions where ALCU_ZETA might be stable
"""

import pycalphad as cp
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def test_alcufe_targeted():
    """Test specific Al-Cu-Fe compositions at 1000°C"""
    
    # Load database
    db = cp.Database('Al-Cu-Fe.tdb')
    components = ['AL', 'CU', 'FE', 'VA']
    
    # Get all phases
    all_phases = [phase for phase in db.phases.keys() if phase != 'VA']
    
    # Temperature at 1000°C
    temperature = 1273.15  # K
    
    # Test specific compositions - focusing on Al-Cu rich region where ALCU_ZETA might be stable
    test_points = [
        # (X_AL, X_CU) - X_FE is implicit
        (0.4, 0.5),  # 40% Al, 50% Cu, 10% Fe
        (0.45, 0.45), # 45% Al, 45% Cu, 10% Fe  
        (0.5, 0.4),  # 50% Al, 40% Cu, 10% Fe
        (0.6, 0.3),  # 60% Al, 30% Cu, 10% Fe
        (0.7, 0.2),  # 70% Al, 20% Cu, 10% Fe (original problematic case)
        (0.8, 0.1),  # 80% Al, 10% Cu, 10% Fe
        (0.3, 0.6),  # 30% Al, 60% Cu, 10% Fe
        (0.35, 0.55), # 35% Al, 55% Cu, 10% Fe
        (0.55, 0.35), # 55% Al, 35% Cu, 10% Fe
        (0.65, 0.25), # 65% Al, 25% Cu, 10% Fe
    ]
    
    print("Al-Cu-Fe Targeted Test at 1000°C")
    print("=" * 120)
    print(f"Temperature: {temperature} K ({temperature-273.15:.0f}°C)")
    print(f"Testing {len(test_points)} specific compositions")
    print("=" * 120)
    
    # Header
    print(f"\n{'X(AL)':<6} {'X(CU)':<6} {'X(FE)':<6} {'CPU_GM':<12} {'GPU_GM':<12} {'Error':<10} {'Status':<8} {'ALCU_ZETA?':<12}")
    print("-" * 80)
    
    results = []
    
    for x_al, x_cu in test_points:
        x_fe = 1.0 - x_al - x_cu
        
        # Set up conditions
        conditions = {
            cp.v.X('AL'): x_al,
            cp.v.X('CU'): x_cu,
            cp.v.T: temperature,
            cp.v.P: 101325
        }
        
        # CPU calculation
        cpu_success = False
        cpu_gm = None
        cpu_has_zeta = False
        
        try:
            eq_cpu = cp.equilibrium(db, components, all_phases, conditions, 
                                   verbose=False, debug=False)
            cpu_gm = float(eq_cpu.GM.values.flat[0])
            
            # Check if ALCU_ZETA is stable
            for i, phase in enumerate(eq_cpu.Phase.values.flat):
                if phase == 'ALCU_ZETA' and eq_cpu.NP.values.flat[i] > 1e-6:
                    cpu_has_zeta = True
                    break
                    
            cpu_success = True
            
        except Exception as e:
            pass
        
        # GPU calculation
        gpu_success = False
        gpu_gm = None
        gpu_has_zeta = False
        
        try:
            eq_gpu = cp.equilibrium(db, components, all_phases, conditions, 
                                   gpu=True, verbose=False, debug=False)
            gpu_gm = float(eq_gpu.GM.values.flat[0])
            
            # Check if ALCU_ZETA is stable
            for i, phase in enumerate(eq_gpu.Phase.values.flat):
                if phase == 'ALCU_ZETA' and eq_gpu.NP.values.flat[i] > 1e-6:
                    gpu_has_zeta = True
                    break
                    
            gpu_success = True
            
        except Exception as e:
            pass
        
        # Compare and output results
        if cpu_success and gpu_success:
            error = abs(cpu_gm - gpu_gm)
            status = "PASS" if error < 1.0 else "FAIL"
            zeta_status = "Both" if cpu_has_zeta and gpu_has_zeta else \
                         "CPU only" if cpu_has_zeta else \
                         "GPU only" if gpu_has_zeta else \
                         "Neither"
            
            print(f"{x_al:<6.2f} {x_cu:<6.2f} {x_fe:<6.2f} "
                  f"{cpu_gm:<12.1f} {gpu_gm:<12.1f} {error:<10.3f} "
                  f"{status:<8} {zeta_status:<12}")
                  
            results.append({
                'x_al': x_al, 'x_cu': x_cu, 'x_fe': x_fe,
                'cpu_gm': cpu_gm, 'gpu_gm': gpu_gm, 'error': error,
                'cpu_has_zeta': cpu_has_zeta, 'gpu_has_zeta': gpu_has_zeta
            })
            
        elif cpu_success:
            print(f"{x_al:<6.2f} {x_cu:<6.2f} {x_fe:<6.2f} "
                  f"{cpu_gm:<12.1f} {'GPU_FAIL':<12} {'N/A':<10} "
                  f"{'GPU_ERR':<8} {'N/A':<12}")
                  
        elif gpu_success:
            print(f"{x_al:<6.2f} {x_cu:<6.2f} {x_fe:<6.2f} "
                  f"{'CPU_FAIL':<12} {gpu_gm:<12.1f} {'N/A':<10} "
                  f"{'CPU_ERR':<8} {'N/A':<12}")
                  
        else:
            print(f"{x_al:<6.2f} {x_cu:<6.2f} {x_fe:<6.2f} "
                  f"{'BOTH_FAIL':<12} {'BOTH_FAIL':<12} {'N/A':<10} "
                  f"{'BOTH_ERR':<8} {'N/A':<12}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        errors = [r['error'] for r in results]
        print(f"\nSuccessful comparisons: {len(results)}")
        print(f"Mean error: {np.mean(errors):.3f} J/mol")
        print(f"Max error: {np.max(errors):.3f} J/mol")
        print(f"Min error: {np.min(errors):.3f} J/mol")
        
        # Check ALCU_ZETA stability
        zeta_results = [r for r in results if r['cpu_has_zeta'] or r['gpu_has_zeta']]
        if zeta_results:
            print(f"\nALCU_ZETA stable in {len(zeta_results)} conditions:")
            for r in zeta_results:
                print(f"  X(AL)={r['x_al']:.2f}, X(CU)={r['x_cu']:.2f}: ", end='')
                if r['cpu_has_zeta'] and r['gpu_has_zeta']:
                    print(f"Both (error={r['error']:.1f} J/mol)")
                elif r['cpu_has_zeta']:
                    print("CPU only")
                else:
                    print("GPU only")
                    
            # Check against 806 J/mol benchmark for ALCU_ZETA cases
            zeta_errors = [r['error'] for r in zeta_results if r['cpu_has_zeta'] and r['gpu_has_zeta']]
            if zeta_errors:
                print(f"\nALCU_ZETA error analysis (when stable in both):")
                print(f"  Mean: {np.mean(zeta_errors):.1f} J/mol")
                print(f"  Max: {np.max(zeta_errors):.1f} J/mol")
                if np.max(zeta_errors) > 806:
                    print(f"  Worse than 806 J/mol benchmark by: {np.max(zeta_errors)/806:.1f}x")
                else:
                    print(f"  Better than 806 J/mol benchmark by: {806/np.max(zeta_errors):.1f}x")

if __name__ == "__main__":
    test_alcufe_targeted()