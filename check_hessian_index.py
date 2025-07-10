#!/usr/bin/env python3
"""Check Hessian index mapping"""

# With 5 variables (N, P, T, Y_NB, Y_TI), the Hessian is 5x5
# The index mapping for a symmetric matrix stored as a 1D array is:
# out[k] = hess[i,j] where k = i*n + j

n = 5  # number of variables

print("Hessian index mapping:")
k = 0
for i in range(n):
    for j in range(n):
        print(f"out[{k}] = hess[{i},{j}]", end="")
        if i == 3 and j == 3:
            print(" <- This is Y_NB diagonal")
        elif i == 4 and j == 4:
            print(" <- This is Y_TI diagonal")
        else:
            print()
        k += 1

print(f"\nSo out[18] corresponds to hess[3,3] (Y_NB diagonal)")
print(f"And out[24] corresponds to hess[4,4] (Y_TI diagonal)")