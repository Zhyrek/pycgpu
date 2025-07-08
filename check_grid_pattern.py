# The test uses a grid of compositions
# Grid points are likely evenly spaced in composition
# With 20 points (0-19), and if we're varying X(TI):
# Point 0: X(TI) = 0.00
# Point 1: X(TI) = 0.05  <- This one fails\!
# Points 2-18: X(TI) = 0.10 to 0.90
# Point 19: X(TI) = 0.95  <- This one fails\!

print("Grid composition analysis:")
print("Point 1: X(TI) ≈ 0.05 - Near pure NB")
print("Point 19: X(TI) ≈ 0.95 - Near pure TI")
print("\nBoth failing points are near pure components\!")
print("This suggests the GPU solver has issues with:")
print("1. Near-pure component compositions")
print("2. Or phase stability calculations at composition extremes")
