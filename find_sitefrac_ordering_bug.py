#!/usr/bin/env python3
"""Find the site fraction ordering bug in GPU code generation."""

# The bug is that the GPU code is generating:
# out[0] = 1.0*x[3]  for moles(NB)
# out[1] = 1.0*x[4]  for moles(TI)

# But x[3] should be Y_VA and x[4] should be Y_TI based on the ordering.
# So it's assigning moles(NB) = Y_VA, which is wrong!

# The issue must be in how the site fractions are ordered when building the x[] array.

print("SITE FRACTION ORDERING ANALYSIS")
print("="*50)
print()

print("CPU behavior:")
print("- Site fractions in model: [Y_NB, Y_TI, Y_VA]")
print("- But Y_NB is dependent (Y_NB = 1 - Y_TI - Y_VA)")
print("- So only Y_TI and Y_VA are independent variables")
print("- The CPU removes the first site fraction as dependent")
print()

print("GPU x[] array should be:")
print("- x[0] = N")
print("- x[1] = P") 
print("- x[2] = T")
print("- x[3] = Y_VA  (or Y_TI, depending on ordering)")
print("- x[4] = Y_TI  (or Y_VA, depending on ordering)")
print()

print("The generated code shows:")
print("- out[0] = 1.0*x[3]  <- This is moles(NB)")
print("- out[1] = 1.0*x[4]  <- This is moles(TI)")
print()

print("If x[3]=Y_VA and x[4]=Y_TI, then:")
print("- moles(NB) = x[3] = Y_VA  <- WRONG!")
print("- moles(TI) = x[4] = Y_TI  <- Correct")
print()

print("The problem is that the GPU code generator is not handling")
print("the dependent site fraction correctly. It's directly mapping")
print("Y_NB to some x[] value instead of expanding it to (1 - Y_TI - Y_VA).")
print()

print("SOLUTION:")
print("The GPU code generator needs to:")
print("1. Identify which site fraction is dependent (first one)")
print("2. Replace it with (1 - sum of other site fractions)")
print("3. Generate correct gradients based on this expansion")