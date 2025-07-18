# CPU vs GPU Key Difference Found!

## The Problem
The GPU solver fails to converge while CPU converges in 6 iterations.

## Root Cause
**Different handling of dependent site fractions!**

### CPU Approach:
- Tracks BOTH Y(NB) and Y(TI) as independent variables
- mass_jac[TI] = [0, 0, 0, 0, 1] 
- This means d(moles(TI))/dY(TI) = 1
- The constraint Y(NB) + Y(TI) = 1 is enforced separately

### GPU Approach:
- Only tracks Y(NB) as independent (Y(TI) = 1 - Y(NB))
- mass_jac[TI] = [0, 0, 0, -1, 0]
- This means d(moles(TI))/dY(NB) = -1
- The dependency is built into the gradients

## Why This Matters

When constructing the mole fraction constraint row:

**CPU**: Uses mass_jac[1,4] = 1.0 and c_G[1] = -0.22
- Gives reasonable constraint coefficients
- The constraint can be satisfied by adjusting Y(TI) directly

**GPU**: Uses mass_jac[1,3] = -1.0 and c_component is tiny (~3e-05)
- Results in tiny constraint coefficients
- Makes constraint enforcement ineffective
- The solver can't correct the composition error

## The Scaling Issue

The GPU's approach would work IF c_component wasn't so small. But:
1. The Hessian is large (~50,000)
2. Its inverse (e_matrix) is tiny (~3e-05)
3. c_component = mass_jac * e_matrix is also tiny
4. This makes the constraint row coefficients too small

## Why CPU Doesn't Have This Issue

The CPU avoids this by:
1. Tracking all site fractions as independent
2. Using a different gradient structure
3. Possibly having better scaled matrices
4. The constraint enforcement is more direct

## Solution

The GPU code needs to either:
1. Track all site fractions like CPU does, OR
2. Fix the scaling issue so c_component isn't so small, OR
3. Handle constraints differently to avoid the scaling problem