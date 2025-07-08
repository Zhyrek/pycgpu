#!/usr/bin/env python
"""Test GPU function with actual values"""

import numpy as np
import math

# The GPU function expects x array with:
# x[0] = N = 1.0
# x[1] = P = 101325
# x[2] = T = 1000
# x[3] = Y(BCC_A2,0,NB) = 0.610169491525422
# x[4] = Y(BCC_A2,0,TI) = 0.389830508474579

def gpu_obj_function(x):
    """Python version of the GPU obj function"""
    # Check for division by zero
    sum_y = x[3] + x[4]
    if sum_y == 0:
        return float('nan')
    
    # First term: NB reference energy
    if x[2] < 2750.0:
        nb_ref = -8519.353 + 142.045475*x[2] - 26.4711*x[2]*math.log(x[2]) + 93399.0*pow(x[2], -1.0) + 0.000203475*pow(x[2], 2.0) - 3.5012e-07*pow(x[2], 3.0)
    elif 2750.0 <= x[2]:
        nb_ref = -37669.3 + 271.720843*x[2] - 41.77*x[2]*math.log(x[2]) + 1.528238e+32*pow(x[2], -9.0)
    else:
        nb_ref = 0
    
    # Second term: TI reference energy
    if x[2] < 1155.0:
        ti_ref = -1272.064 + 134.71418*x[2] - 25.5768*x[2]*math.log(x[2]) + 7208.0*pow(x[2], -1.0) - 0.000663845*pow(x[2], 2.0) - 2.78803e-07*pow(x[2], 3.0)
    elif x[2] < 1941.0 and 1155.0 <= x[2]:
        ti_ref = 6667.385 + 105.366379*x[2] - 22.3771*x[2]*math.log(x[2]) - 2002750.0*pow(x[2], -1.0) + 0.00121707*pow(x[2], 2.0) - 8.4534e-07*pow(x[2], 3.0)
    elif 1941.0 <= x[2]:
        ti_ref = 26483.26 - 182.426471*x[2] + 19.0900905*x[2]*math.log(x[2]) + 1400501.0*pow(x[2], -1.0) - 0.02200832*pow(x[2], 2.0) + 1.228863e-06*pow(x[2], 3.0)
    else:
        ti_ref = 0
    
    # Ideal mixing term
    ideal_nb = x[3]*math.log(x[3]) if 1e-15 < x[3] else 0
    ideal_ti = x[4]*math.log(x[4]) if 1e-15 < x[4] else 0
    
    # Calculate result
    result = 1.0*(x[3]*nb_ref + x[4]*ti_ref)/sum_y + 8.3145*x[2]*(1.0*ideal_nb + 1.0*ideal_ti)/sum_y + 13045.3*x[3]*x[4]/sum_y
    
    return result

# Test with GPU values
x = np.array([1.0, 101325, 1000, 0.610169491525422, 0.389830508474579])
print(f"Testing with x = {x}")
print(f"Sum of site fractions: {x[3] + x[4]}")

result = gpu_obj_function(x)
print(f"Python GPU function result: {result}")

# Also check intermediate values
print("\nIntermediate values:")
print(f"  T = {x[2]} (should use T < 1155 branch for TI)")
print(f"  NB ref energy term: {-8519.353 + 142.045475*x[2] - 26.4711*x[2]*math.log(x[2]) + 93399.0*pow(x[2], -1.0) + 0.000203475*pow(x[2], 2.0) - 3.5012e-07*pow(x[2], 3.0)}")
print(f"  TI ref energy term: {-1272.064 + 134.71418*x[2] - 25.5768*x[2]*math.log(x[2]) + 7208.0*pow(x[2], -1.0) - 0.000663845*pow(x[2], 2.0) - 2.78803e-07*pow(x[2], 3.0)}")
print(f"  log(Y_NB) = {math.log(x[3])}")
print(f"  log(Y_TI) = {math.log(x[4])}")

# Expected CPU result
print(f"\nExpected CPU result: -49808.12492831601")