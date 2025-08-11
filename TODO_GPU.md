# GPU Code TODO List

## Batch Processing for Large Condition Sets

**Status**: Design complete, implementation needed

### Problem:
- GPU runs out of memory with >2000-3000 conditions
- Each thread needs ~600 KB workspace (due to 138x138 SVD matrices)
- 10,000 conditions = 5.8 GB just for work arrays

### Solution - Batch Processing with 4K Thread Cap:

Implement striding pattern where each thread processes multiple conditions:
- Cap at 4096 threads (64 blocks × 64 threads/block)
- Memory usage stays constant at 2.4 GB maximum
- Each thread processes conditions with stride of 4096

#### Implementation:
1. **Kernel modification** - Add stride loop:
```c
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;  // 4096

for (int cond_idx = tid; cond_idx < num_conditions; cond_idx += stride) {
    // Process condition cond_idx
}
```

2. **Memory allocation** - Cap at MAX_THREADS:
```python
MAX_THREADS = 4096
actual_threads = min(num_conditions, MAX_THREADS)
# Allocate arrays for actual_threads, not num_conditions
```

3. **Benefits**:
- 50,000 conditions: 2.4 GB instead of 29.8 GB (92% savings)
- 100,000+ conditions possible on 8GB GPU
- Simple implementation, good load balancing

## Multi-Component System Support (Ternary and Higher)

**Status**: Currently broken for systems with more than 2 non-VA components

### Issues Found:

1. **Hard-coded 2-component assumption in gpu_systemspec_array.py**:
   - Line 59: Code only processes ONE X() condition then breaks
   - For ternary systems like Al-Cu-Fe with X(CU) and X(FE), only X(CU) is processed
   - This causes incorrect SystemSpecification population for additional components

2. **Incorrect condition indexing for multi-composition systems**:
   - Current code assumes simple 2D grid (temperature × single composition)
   - Actual pycalphad creates N-dimensional grid for ternary: T × X_comp1 × X_comp2
   - Example: Al-Cu-Fe with T=[600,800,1000], X_CU=[0.1,0.2,0.3,0.4], X_FE=[0.1,0.2,0.3,0.4]
     creates 3×4×4=48 conditions, not handled correctly

3. **Pycalphad range syntax quirks**:
   - Range (start, stop, step) is NOT consistently inclusive of stop value
   - (600, 1200, 200) gives [600, 800, 1000] - missing 1200
   - (0.1, 0.5, 0.1) gives [0.1, 0.2, 0.3, 0.4] - missing 0.5
   - Need to account for this when calculating indices

### Required Fixes:

1. **Update gpu_systemspec_array.py**:
   - Process ALL X() conditions, not just the first one
   - Store all composition conditions in a dictionary
   - Update indexing logic to handle N-dimensional condition grids
   - Correct formula for ternary: 
     ```python
     temp_idx = condition_idx // (x_cu_len * x_fe_len)
     remainder = condition_idx % (x_cu_len * x_fe_len)  
     cu_idx = remainder // x_fe_len
     fe_idx = remainder % x_fe_len
     ```

2. **Update TempWorkspace class**:
   - Handle multiple composition indices properly
   - Map each X() condition to its correct index in the flattened array

3. **Update PropertiesSubset**:
   - May need to handle multiple composition indices for property extraction

4. **Verify _populate_system_specification**:
   - Ensure it processes ALL mole fraction constraints
   - Currently may only handle one X() constraint properly

### Test Case:
- Al-Cu-Fe system with LIQUID and FCC_A1 phases
- Conditions: X(CU)=(0.1,0.5,0.1), X(FE)=(0.1,0.4,0.1), T=(600,1200,200)
- Currently fails with 0% pass rate after attempted fix
- Shows garbage values for chemical potentials suggesting memory/data issues

### Notes:
- Binary systems (Au-Bi) work fine with current code
- Issue only manifests with 3+ component systems
- Modular compilation for many phases is a separate issue (already addressed)