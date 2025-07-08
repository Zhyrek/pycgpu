#!/usr/bin/env python3
"""Simple profiling script for test_script.py"""

import cProfile
import pstats
import sys

# Profile the test script
if __name__ == "__main__":
    # Run with cProfile
    cProfile.run('exec(open("test_script.py").read())', 'profile_stats')
    
    # Load and print statistics
    p = pstats.Stats('profile_stats')
    
    print("\n" + "="*80)
    print("PROFILE RESULTS")
    print("="*80)
    
    # Sort by cumulative time and show top 30 functions
    print("\nTop 30 functions by cumulative time:")
    p.sort_stats('cumulative').print_stats(30)
    
    # Sort by total time spent in function
    print("\n\nTop 20 functions by time spent in function itself:")
    p.sort_stats('time').print_stats(20)
    
    # Show callers of the equilibrium function
    print("\n\nCalls to equilibrium function:")
    p.print_callers('equilibrium')
    
    # Clean up
    import os
    os.remove('profile_stats')