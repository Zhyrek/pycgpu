#!/usr/bin/env python3
"""Time different phases of GPU execution"""

import time
import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v

class TimingContext:
    def __init__(self):
        self.times = {}
        
    def time_it(self, name, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self.times[name] = elapsed
        return result
    
    def report(self):
        total = sum(self.times.values())
        print(f"\n{'Phase':<30} {'Time (ms)':>10} {'Percent':>8}")
        print("-" * 50)
        for name, elapsed in self.times.items():
            print(f"{name:<30} {elapsed*1000:>10.1f} {elapsed/total*100:>7.1f}%")
        print("-" * 50)
        print(f"{'TOTAL':<30} {total*1000:>10.1f} {100.0:>7.1f}%")

def profile_gpu_phases():
    """Profile different phases of GPU calculation"""
    ctx = TimingContext()
    
    # Setup phase
    tdb = ctx.time_it("Database loading", pyc.Database, "NbTi.tdb")
    phases = ["LIQUID", "BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    # Test with different condition sizes
    for n_conditions in [1, 10, 100]:
        print(f"\n{'='*60}")
        print(f"Testing with {n_conditions} conditions")
        print(f"{'='*60}")
        
        ctx = TimingContext()
        
        if n_conditions == 1:
            conditions = {v.X("TI"): 0.5, v.T: 500}
        else:
            conditions = {v.X("TI"): np.linspace(0, 1, n_conditions), v.T: 500}
        
        # First run (includes compilation)
        if n_conditions == 1:
            eq_gpu = ctx.time_it("GPU first run (with compilation)", 
                               equilibrium, tdb, comps, phases, conditions, 
                               gpu=True, verbose=False)
        
        # Subsequent run (warmed up)
        eq_gpu = ctx.time_it(f"GPU equilibrium ({n_conditions} conditions)", 
                           equilibrium, tdb, comps, phases, conditions, 
                           gpu=True, verbose=False)
        
        # For comparison, run CPU
        eq_cpu = ctx.time_it(f"CPU equilibrium ({n_conditions} conditions)", 
                           equilibrium, tdb, comps, phases, conditions, 
                           gpu=False, verbose=False)
        
        ctx.report()
        
        # Check accuracy
        mu_error = np.max(np.abs(eq_cpu.MU.values - eq_gpu.MU.values))
        print(f"\nMax MU error: {mu_error:.2e}")
        print(f"GPU speedup: {ctx.times[f'CPU equilibrium ({n_conditions} conditions)']/ctx.times[f'GPU equilibrium ({n_conditions} conditions)']:.2f}x")

if __name__ == "__main__":
    print("GPU Phase Timing Analysis")
    profile_gpu_phases()