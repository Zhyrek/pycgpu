# GPU Parameter Handling Fix

## Problem Identified

The GPU code is not handling parameters correctly when calling phase record functions like `formulahess`. 

### CPU Behavior:
- The CPU phase record methods use `alloc_dof_with_parameters` which:
  1. Takes the first `num_statevars + phase_dof` elements from the DOF array
  2. Concatenates them with parameter values from `phase_record.parameters`
  3. Passes this concatenated array to the compiled functions

### GPU Issue:
- The GPU code passes `compset->dof` directly to functions without appending parameters
- This causes the compiled functions to read incorrect values if they expect parameters at the end

## Solution

Need to modify the GPU code to:
1. Store parameter values in the phase record or pass them to the kernel
2. Allocate a temporary array that concatenates DOF values with parameters
3. Pass this concatenated array to the phase record functions

## Code Changes Required

1. In `phase_rec.h`: Add parameter storage
2. In `minimizer.h`: Allocate and concatenate DOF with parameters before function calls
3. In GPU kernel initialization: Pass parameter values from workspace

## Implementation Plan

The fix involves modifying how the GPU code calls phase record functions to match the CPU's parameter handling.