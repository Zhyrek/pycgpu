# CPU vs GPU Comparison After Iteration 0

## Key Differences Found:

### 1. Chemical Potentials
- **CPU**: [-51167.0325761, -47806.28007426] (2 components)
- **GPU**: [-51167.03257610269, -47806.28007426182, 0.0] (3 components including VA)

The values match to high precision, but GPU includes VA component.

### 2. Phase Amounts (formula units)
- **CPU Phase 0**: 0.234568 (different from NP!)
- **CPU Phase 1**: 0.765432 (different from NP!)
- **GPU Phase 0**: 0.373665 (same as NP)
- **GPU Phase 1**: 0.626335 (same as NP)

**Critical Issue**: CPU phase_amt is different from NP, while GPU phase_amt equals NP.

### 3. Site Fractions
- **CPU Phase 0**: [0.70189102, 0.29810898]
- **CPU Phase 1**: [0.70189387, 0.29810613]
- **GPU Phase 0**: [0.291477, 1e-14] 
- **GPU Phase 1**: [0.305085, 1e-14]

**Critical Issue**: GPU site fractions are stored incorrectly!

### 4. State Variables (DOF array)
- **CPU**: [1.0, 1000.0] - [N, T]
- **GPU**: [1000.0, 0.708523] - Appears to be [T, X(NB)]

**Critical Issue**: GPU is storing DOF array incorrectly.

### 5. Phase Compositions  
- **CPU**: Not shown (MemoryView)
- **GPU Phase 0**: [0.708523, 0.291477, 0.0] - Correct X values
- **GPU Phase 1**: [0.0, 0.694915, 0.305085] - Wrong! Should be [0.694915, 0.305085, 0.0]

### 6. Convergence Changes
- **CPU**: 
  - largest_phase_amt_change: 0.139097 (significant change)
  - largest_y_change: 0.006979
- **GPU**: 
  - largest_phase_amt_change: 4.08e-15 (no change)
  - largest_y_change: 1e-14

## Root Causes:

1. **DOF Array Format Mismatch**: GPU stores DOF as [T, Y1, Y2] while CPU expects [N, P, T, Y1, Y2]
2. **Phase Amount Calculation**: CPU calculates phase_amt differently from NP (formula units vs mole fraction)
3. **Site Fraction Storage**: GPU is not extracting/storing site fractions correctly from DOF array
4. **Phase Composition Error**: GPU Phase 1 composition is shifted/wrong

These differences explain why CPU and GPU diverge significantly after just one iteration.