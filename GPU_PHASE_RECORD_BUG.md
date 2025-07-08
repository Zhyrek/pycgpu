# GPU Phase Record Assignment Bug

## Issue
The GPU equilibrium solver has a fundamental design issue where multiple instances of the same phase type (e.g., two BCC_A2 phases with different compositions) are assigned the same phase_record pointer. This causes both phases to share the same memory and calculations, leading to incorrect results.

## Root Cause
The GPU code creates a mapping from phase names to unique indices:
```python
py_phase_name_to_unique_idx_map = {'BCC_A2': 0, 'LIQUID': 1}
```

When the lower_convex_hull result contains multiple instances of the same phase (e.g., two BCC_A2 phases at different compositions), both get mapped to the same index (0), causing them to use the same phase_record pointer.

## Where the Bug Occurs
1. In `gpu_codegen.py` line 1441: `py_phase_name_to_unique_idx_map[ph_name] = len(unique_py_models)`
2. In `gpu_equilibrium.py` line 568: `py_phase_name_to_unique_idx_map[phase_name]` 
3. In `eqsolver.h` line 367: `phase_record = &phase_data->phase_records_array[pr_idx]`

## Impact
- Only works correctly when each phase type appears at most once in equilibrium
- Fails when immiscibility gaps exist (multiple instances of same phase)
- Causes GPU threads to compute incorrect equilibria

## Proper Fix
The GPU architecture needs to be redesigned to support multiple instances of the same phase type:
1. Instead of mapping phase names to unique indices, track grid point indices
2. Create phase records for each grid point, not just unique phase types
3. Pass grid point indices from lower_convex_hull to the GPU kernel

## Temporary Workaround
Add a check to detect when multiple instances of the same phase are present and either:
1. Fall back to CPU calculation
2. Warn the user that GPU calculation may be incorrect
3. Implement phase instance tracking

## Test Case
```python
# This case triggers the bug:
conditions = {v.T: 1000, v.P: 101325, v.X('TI'): 0.3}
# Results in two BCC_A2 phases with different compositions
```