# Phase Amount Normalization Analysis

## Problem Summary
The GPU code shows incorrect phase amounts (NP values) for multi-sublattice phases compared to CPU code. For example:
- CPU: BCC_A2 NP = 0.1000
- GPU: BCC_A2 NP = 0.0050 (20x smaller)

## Root Cause Analysis

### 1. CPU Code Behavior

In the CPU code (`pycalphad/core/minimizer.pyx`):
```python
# In recompute() method:
for comp_idx in range(num_components):
    compset.phase_record.formulamole_obj(csst.masses[comp_idx, :], x, comp_idx)
    if self.phase_amt[idx] > 0:
        self.mole_fractions[comp_idx] += self.phase_amt[idx] * csst.masses[comp_idx, 0]
        self.system_amount += self.phase_amt[idx] * csst.masses[comp_idx, 0]
```

The `formulamole_obj` function calls `Model.moles()` with `per_formula_unit=False`, which means:
- The result is divided by a normalization factor
- For multi-sublattice phases, this normalization is the sum of (site_ratio * number_of_atoms)
- For BCC_A2 with site ratios [9, 11], the normalization factor is 20

### 2. Model.moles() Implementation

From `pycalphad/model.py`:
```python
def moles(self, species, per_formula_unit=False):
    # ... calculation of result and normalization ...
    if not per_formula_unit:
        return result / normalization  # This is the key normalization!
    else:
        return result
```

### 3. GPU Code Issue

The GPU code appears to be missing this normalization factor when:
1. Initializing phase amounts
2. Converting between phase amounts and system constraints
3. Handling the phase consolidation logic

## Solution

The GPU code needs to:

1. **Store the normalization factor** for each phase (sum of site_ratios * atoms_per_site)
2. **Apply this normalization** when:
   - Calculating masses from formula units
   - Converting phase amounts in constraint equations
   - During phase consolidation

## Impact on Constraints

The mole fraction constraint in the CPU code uses:
```python
out_row[...] += prefactor * (phase_amt[idx]/current_system_amount) * 
                (masses[component_idx, 0] - system_mole_fractions[component_idx] * moles_normalization)
```

Where `moles_normalization` is the sum of all masses for that phase, which accounts for the site ratio normalization.

## Verification

To verify this is the issue:
1. BCC_A2 has site ratios [9, 11] = total 20 atoms per formula unit
2. GPU shows NP = 0.0050, CPU shows NP = 0.1000
3. Ratio: 0.1000 / 0.0050 = 20, which matches the normalization factor!

This confirms the GPU code is missing the site ratio normalization for multi-sublattice phases.