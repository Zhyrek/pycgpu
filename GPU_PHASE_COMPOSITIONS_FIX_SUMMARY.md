# GPU Phase Compositions Indexing Fix Summary

## Problem
The GPU was incorrectly reading Phase 1's X values as zeros even though the kernel correctly calculated them as [0.067797, 0.932203].

## Root Causes Found and Fixed

### 1. Phase Compositions Array Indexing (FIXED)
- **Issue**: GPU code was using `phase_compositions[idx * spec->num_components + c]` for indexing
- **Fix**: Changed all occurrences to use `phase_compositions[idx * MAX_COMPONENTS + c]`
- **Files Fixed**:
  - `/pycalphad/gpu/minimizer.h` - Fixed 11 occurrences in recompute() and other functions
  - `/pycalphad/gpu/gpu_codegen.py` - Fixed all generated kernel code to use MAX_COMPONENTS

### 2. Incorrect Phase Merger Logic (PARTIALLY FIXED) 
- **Issue**: Python post-processing was incorrectly merging two phases in a miscibility gap
- **Partial Fix**: Disabled the phase merger by setting `has_symmetric_constraint = False`
- **File**: `/pycalphad/gpu/gpu_equilibrium.py` lines 1458-1463
- **Note**: This merger logic should be completely removed as it incorrectly collapses miscibility gaps

## Current Status
- GPU kernel correctly calculates and stores phase compositions with proper MAX_COMPONENTS spacing
- Debug output shows correct values: Phase 0=[0.946395, 0.053605], Phase 1=[0.067797, 0.932203]
- Python still shows Phase 1 X values as zeros due to remaining issues in result extraction

## Remaining Issues
1. Both phases have the same phase_id (0 = BCC_A2), which may trigger unwanted merging
2. Result extraction logic needs to properly handle the packed stable phase layout from GPU
3. GM difference is still 1.248 J/mol (above 0.001 threshold)

## Test Command
```bash
python test_final_fix.py
```

## Key Insight
The code must ALWAYS use MAX_COMPONENTS for array indexing stride, even when looping over num_components. This ensures consistent memory layout between CPU and GPU code.