# Critical Finding: CPU vs GPU Matrix Size Mismatch

## Condition
- X(AL) = 0.60, X(CU) = 0.10, X(FE) = 0.30
- T = 600K, P = 101325 Pa
- Phases: LIQUID, FCC_A1, BCC_A2, BCC_B2, L12, ALCU_THETA, AL13FE4, AL5FE2

## Key Discovery
**CPU and GPU are solving different sized systems:**
- CPU: 6x6 matrix (2 phases active)
- GPU: 7x7 matrix (3 phases active)

This explains the 706 J/mol energy difference!

## Matrix Analysis

### CPU Matrix (6x6)
- 2 active phases (rows 0-1: energy equations)
- 2 composition constraints (rows 3-4: X(AL), X(CU))
- 1 system constraint (row 5: N=1)
- Total: 6 equations

### GPU Matrix (7x7)
- 3 active phases (rows 0-3: energy equations)
- 2 composition constraints (rows 4-5: X(AL), X(CU))
- 1 system constraint (row 6: N=1)
- Total: 7 equations

## What This Means
The GPU is considering an additional phase that the CPU is not, leading to:
1. Different equilibrium solutions
2. GPU finding lower energy (-49226 vs -48520 J/mol)
3. Different phase assemblages:
   - CPU: BCC_B2(0.47) + AL5FE2(0.53)
   - GPU: BCC_B2(0.86) + AL5FE2(0.14) + [third phase]

## Root Cause
The initial phase selection or convergence criteria differ between CPU and GPU:
- CPU converges with 2 phases
- GPU explores and retains 3 phases

## Implications
1. This is not a numerical precision issue
2. It's a fundamental difference in phase selection
3. GPU may be finding a more stable equilibrium (lower energy)
4. Need to investigate why CPU drops the third phase

## Next Steps
1. Check which phase GPU includes that CPU doesn't
2. Verify if GPU solution is thermodynamically correct
3. Investigate CPU's phase dropping criteria
4. Test if this pattern occurs in other failing conditions