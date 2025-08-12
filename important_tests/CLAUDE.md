# Equilibrium Matrix Debug Tool Documentation

## Overview

The `print_equilibrium_matrix.py` script is a debugging tool designed to print out the equilibrium matrices from both CPU and GPU calculations at the first solver iteration. This is essential for identifying divergences between CPU and GPU implementations of the equilibrium solver.

## Purpose

When debugging GPU/CPU divergences in pycalphad:
1. The first step is always to check if the equilibrium matrices are identical
2. The equilibrium matrix and RHS vector at iteration 0 should match exactly between CPU and GPU
3. Any differences in these matrices will propagate and cause divergent results

## Usage

### Basic Syntax

```bash
python print_equilibrium_matrix.py <database> <components> <phases> [options]
```

### Arguments

- `database`: Path to the thermodynamic database file (e.g., `Al-Cu-Fe.tdb`)
- `components`: Either:
  - Comma-separated list of components including VA (e.g., `AL,CU,FE,VA`)
  - `all` to use all components from the database
- `phases`: Either:
  - Comma-separated list of phases to consider (e.g., `LIQUID` or `LIQUID,FCC_A1`)
  - `all` to use all phases from the database

### Options

#### Thermodynamic Conditions
- `--T`: Temperature in Kelvin (default: 1000)
- `--P`: Pressure in Pascal (default: 101325)
- `--X_<ELEMENT>`: Mole fraction for specific elements (e.g., `--X_AL=0.5`)
  - Supported elements: AL, AU, BI, CU, FE, NB, TI
  - Add more as needed in the script

#### Output Control
- `--cpu-only`: Only run CPU calculation
- `--gpu-only`: Only run GPU calculation
- `--no-matrix`: Skip matrix output, only show GM values
- `--verbose`: Enable verbose GPU output (shows additional debug info)

## Examples

### 1. Binary System (Au-Bi)
```bash
python print_equilibrium_matrix.py AuBi-07Wan.tdb AU,BI,VA LIQUID --T=700 --P=101325 --X_BI=0.3
```

### 2. Ternary System (Al-Cu-Fe)
```bash
python print_equilibrium_matrix.py ../Al-Cu-Fe.tdb AL,CU,FE,VA LIQUID --T=973.15 --P=101325 --X_AL=0.5 --X_CU=0.2
```
Note: X(FE) is automatically calculated as 1.0 - 0.5 - 0.2 = 0.3

### 3. Multiple Phases
```bash
python print_equilibrium_matrix.py AuBi-07Wan.tdb AU,BI,VA FCC_A1,LIQUID --T=600 --P=101325 --X_BI=0.4
```

### 4. C15 Laves Phase Test
```bash
python print_equilibrium_matrix.py NbTi.tdb NB,TI,VA LIQUID,BCC_A2,HCP_A3,C15 --T=2000 --P=101325 --X_TI=0.5
```

### 5. Using "all" keyword for comprehensive testing
```bash
# Test with all phases in the database
python print_equilibrium_matrix.py Al-Cu-Fe.tdb AL,CU,FE,VA all --T=973.15 --P=101325 --X_AL=0.5 --X_CU=0.2

# Test with all components and all phases
python print_equilibrium_matrix.py Al-Cu-Fe.tdb all all --T=973.15 --P=101325 --X_AL=0.5 --X_CU=0.2
```

Note: Using "all" for phases may include phases that are unstable at the given conditions, which is useful for comprehensive testing but may result in longer computation times.

## Enabling Debug Output

### For CPU Matrix Output
The script automatically sets `PYCALPHAD_DEBUG_MODE=1` environment variable when running CPU calculations (unless `--no-matrix` is specified). This triggers the CPU code to print the equilibrium matrix at iteration 0.

### For GPU Matrix Output  
The script passes `verbose=True` to the GPU equilibrium call when matrix output is needed. This causes the GPU kernel to be compiled with `-DVERBOSE_DEBUG`, enabling matrix output.

### Manual Debug Mode
If running pycalphad directly (not through this script):

```python
# For CPU debugging
import os
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

# For GPU debugging
result = equilibrium(..., verbose=True, gpu=True)  # verbose=True enables VERBOSE_DEBUG
```

## Interpreting Output

### Matrix Structure

The equilibrium matrix for a ternary system with prescribed mole fractions typically has the following structure:

```
[CPU EQUILIBRIUM MATRIX] Iteration 0 (rows=6, cols=6):
  Row 0: [mass_AL] [mass_CU] [mass_FE] [1] [0] [0] | RHS: [energy gradient]
  Row 1: [mass_AL] [mass_CU] [mass_FE] [0] [1] [0] | RHS: [energy gradient]
  Row 2: [mass_AL] [mass_CU] [mass_FE] [0] [0] [1] | RHS: [energy gradient]
  Row 3: [constraint coefficients for X(AL)]         | RHS: [constraint residual]
  Row 4: [constraint coefficients for X(CU)]         | RHS: [constraint residual]
  Row 5: [0] [0] [0] [1] [1] [1]                    | RHS: [N=1 residual]
```

Where:
- Columns 0-2: Chemical potential variables (μ_AL, μ_CU, μ_FE)
- Columns 3-5: Phase amount variables (NP for each active phase)
- Rows 0-2: Mass balance for each phase
- Rows 3-4: Mole fraction constraints
- Row 5: System amount constraint (N=1)

### Common Issues to Check

1. **Matrix Dimensions**: CPU and GPU should have identical dimensions
   - If different, check `num_free_chemical_potentials` calculation
   - Verify VA is properly excluded from free chemical potentials

2. **Mass Values**: First few columns should contain component masses
   - All non-VA components should be present
   - Values should match between CPU and GPU

3. **Constraint Rows**: Mole fraction constraint coefficients
   - Should reflect the prescribed conditions
   - RHS should be the residual (target - current)

4. **System Amount Row**: Last row with [0,0,0,1,1,1] pattern
   - Ensures total phase amounts sum to prescribed system amount
   - RHS is prescribed_amount - current_amount

## Troubleshooting

### No Matrix Output

If you don't see matrix output:

1. Make sure you're not using `--no-matrix` flag
2. Check that debug output is properly enabled in the C/C++ code:
   - CPU: Look for `#ifdef DEBUG_MODE` in `minimizer.pyx`
   - GPU: Look for `#ifdef VERBOSE_DEBUG` in `minimizer.h`
3. The matrix is only printed at iteration 0, so ensure convergence doesn't happen too quickly

### Compilation Errors

If GPU fails to compile with debug flags:
- Check that the kernel size doesn't exceed limits with debug output
- May need to reduce `MAX_PHASES` or other constants when debugging

### Different Results

When CPU and GPU give different results:
1. First run this script to compare matrices
2. Look for differences in:
   - Matrix dimensions
   - Zero vs non-zero patterns
   - Magnitude of values (especially constraint RHS)
3. Trace back to the source of the first difference

## Adding Support for New Elements

To add support for new elements in composition specifications:

1. Edit `print_equilibrium_matrix.py`
2. Add argument in `parse_arguments()`:
   ```python
   parser.add_argument('--X_ZN', type=float, help='Mole fraction of ZN')
   ```
3. The `build_conditions()` function will automatically pick it up

## Notes

- The script captures stdout to extract matrix output, so direct print statements in the equilibrium calculation will be captured
- GM (Gibbs energy) values are always displayed for comparison
- The script works with any pycalphad-compatible database file
- Matrix output requires recompilation of Cython/CUDA code with debug flags