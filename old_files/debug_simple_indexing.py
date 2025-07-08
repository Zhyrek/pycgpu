#!/usr/bin/env python
"""
Simple debug of array indexing.
"""

import numpy as np

def main():
    print("Array Indexing Debug")
    print("="*30)
    
    # Simulate the properties.NP array from the debug output
    # Shape: (1, 1, 3, 1, 3) with values shown in debug
    np_array = np.array([[[[[0.5, 0.5, 0.]],
                          [[0.5, 0.5, 0.]],
                          [[0.5, 0.5, 0.]]]]])
    
    print(f"Array shape: {np_array.shape}")
    print(f"Array values:\n{np_array}")
    
    print(f"\nTesting indexing:")
    
    # Test the indexing used in GPU code
    for cond_idx in range(3):
        multi_idx = np.unravel_index(cond_idx, (1, 1, 3, 1))
        print(f"\nCondition {cond_idx}: multi_idx = {multi_idx}")
        
        # This is what the GPU code does
        indexed_value = np_array[multi_idx]
        print(f"  Indexed value: {indexed_value}")
        print(f"  Indexed value shape: {indexed_value.shape}")
        
        # What we want is the phase amounts for this condition
        correct_value = np_array[0, 0, cond_idx, 0, :]
        print(f"  Correct value: {correct_value}")

if __name__ == "__main__":
    main()