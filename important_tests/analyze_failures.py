#!/usr/bin/env python
"""Analyze the pattern of failing conditions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Analyze failing conditions to find patterns."""
    
    # The failing conditions from the test
    failures = [
        (0.40, 0.40, 0.20, 600),
        (0.50, 0.40, 0.10, 600),
        (0.10, 0.50, 0.40, 900),
        (0.20, 0.50, 0.30, 900),
        (0.30, 0.50, 0.20, 900),
        (0.70, 0.20, 0.10, 900),
        (0.20, 0.40, 0.40, 1200),
        (0.30, 0.60, 0.10, 1200),
        (0.70, 0.10, 0.20, 1200),
    ]
    
    print("=" * 80)
    print("ANALYSIS OF FAILING CONDITIONS")
    print("=" * 80)
    
    print("\nFailing conditions:")
    print("X(AL) | X(CU) | X(FE) | T(K) | Sum | Pattern")
    print("------|-------|-------|------|-----|--------")
    
    for x_al, x_cu, x_fe, temp in failures:
        total = x_al + x_cu + x_fe
        pattern = []
        
        # Check for patterns
        if x_cu == 0.40:
            pattern.append("X_CU=0.4")
        if x_cu == 0.50:
            pattern.append("X_CU=0.5")
        if x_cu == 0.60:
            pattern.append("X_CU=0.6")
        if x_al == 0.70:
            pattern.append("X_AL=0.7")
        if x_cu >= 0.40:
            pattern.append("High Cu")
            
        print(f" {x_al:.2f} | {x_cu:.2f} | {x_fe:.2f} | {temp:4d} | {total:.2f} | {', '.join(pattern)}")
    
    # Analyze patterns
    print("\nPattern analysis:")
    
    cu_values = [x_cu for _, x_cu, _, _ in failures]
    al_values = [x_al for x_al, _, _, _ in failures]
    temps = [temp for _, _, _, temp in failures]
    
    print(f"  X(CU) values in failures: {sorted(set(cu_values))}")
    print(f"  X(AL) values in failures: {sorted(set(al_values))}")
    print(f"  Temperatures in failures: {sorted(set(temps))}")
    
    high_cu = sum(1 for x_cu in cu_values if x_cu >= 0.40)
    print(f"\n  Failures with X(CU) >= 0.40: {high_cu}/{len(failures)} ({high_cu/len(failures)*100:.0f}%)")
    
    # Check grid positions
    print("\nGrid position analysis:")
    print("(Assuming 9x9 grid for X_AL and X_CU)")
    
    # X_AL: 0.1 to 0.9 in 0.1 steps -> indices 0-8
    # X_CU: 0.1 to 0.9 in 0.1 steps -> indices 0-8
    
    for x_al, x_cu, x_fe, temp in failures:
        al_idx = int(round((x_al - 0.1) / 0.1))
        cu_idx = int(round((x_cu - 0.1) / 0.1))
        grid_pos = al_idx * 9 + cu_idx  # Row-major ordering
        
        print(f"  X_AL={x_al:.1f}, X_CU={x_cu:.1f} -> Grid indices ({al_idx},{cu_idx}) -> Linear position {grid_pos}")
    
    print("\n" + "=" * 80)
    print("OBSERVATIONS:")
    print("- Most failures have X(CU) >= 0.40 (high copper content)")
    print("- Failures occur across all temperatures")
    print("- May be related to phase stability at high Cu compositions")
    print("=" * 80)

if __name__ == "__main__":
    main()