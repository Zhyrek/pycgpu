#!/usr/bin/env python
"""Fix GPU solver tolerance issue that causes wrong direction after consolidation."""

print("FIXING GPU SOLVER TOLERANCE ISSUE")
print("=" * 60)

print("\nPROBLEM IDENTIFIED:")
print("1. GPU calls lstsq with tolerance=1e-16")
print("2. But SVD solver internally uses rcond=1e-10 for rank determination")
print("3. After consolidation, condition number is ~1e16")
print("4. This causes SVD to treat matrix as rank-deficient")
print("5. Result: tiny solution despite non-zero RHS")

print("\nSOLUTION:")
print("Change rcond in SVD solver from 1e-10 to match LAPACK's default of ~1e-15")
print("This will allow GPU to handle poorly conditioned systems like CPU does")

# The fix is in svd.c, lines 705 and 749:
# Change: rcond = 1e-10;
# To:     rcond = 1e-15;

print("\nEXPECTED RESULT:")
print("GPU should now produce larger corrections after consolidation")
print("Leading to X(TI) = 0.900000 instead of 0.902960")