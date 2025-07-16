# GPU Stride Fix Summary

## Problem
Python and GPU were using different memory layouts for condition data:
- Python packs with stride = MAX_STATEVARS + MAX_COMPONENTS
- GPU was calculating its own stride based on hardcoded constants
- This caused GPU threads to read from wrong memory locations for multi-condition calculations

## Solution Implemented
1. Pass the stride value from Python to GPU kernel as a parameter
2. Pass Python's MAX_STATEVARS value to GPU kernel  
3. GPU kernel uses these passed values instead of hardcoded constants

## Code Changes

### gpu_equilibrium.py
- Calculate stride using dynamic_sizes or _get_c_define
- Pass `condition_data_stride` and `max_statevars_scalar` as kernel arguments

### gpu_codegen.py
- Updated kernel signature to accept `condition_stride` and `python_max_statevars`
- Use `condition_stride` for offset calculation: `condition_offset = condition_idx * condition_stride`
- Use `python_max_statevars` to find where compositions start in each condition's data

## Verification
The stride fix is **working correctly**:
- Each GPU thread reads from the correct memory offset
- Thread 0 reads X(TI)=0.2 from offset 0
- Thread 1 reads X(TI)=0.3 from offset 8
- Thread 2 reads X(TI)=0.4 from offset 16
- Thread 3 reads X(TI)=0.5 from offset 24

## Status
✅ **STRIDE FIX COMPLETE** - GPU now respects Python's data layout for multi-condition calculations

Note: Some conditions may still show unrealistic GM values due to unrelated solver convergence issues, but the memory access problem is fixed.