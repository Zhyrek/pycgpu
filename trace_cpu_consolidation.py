#!/usr/bin/env python3
"""Trace CPU consolidation"""

print("=== CPU Phase Consolidation ===")

print("\nFrom the hull calculation, we have 2 BCC_A2 phases:")
print("Phase 0: Y(NB)=0.6122449, Y(TI)=0.3877551")
print("Phase 1: Y(NB)=0.59366427, Y(TI)=0.40633573")

print("\nAverage (what CPU uses after consolidation):")
y_nb_avg = (0.6122449 + 0.59366427) / 2
y_ti_avg = (0.3877551 + 0.40633573) / 2
print(f"Y(NB) = {y_nb_avg:.8f}")
print(f"Y(TI) = {y_ti_avg:.8f}")

print("\nComparing with CPU debug output:")
print("CPU shows: Y(NB)=0.60315742, Y(TI)=0.39684258")
print(f"Calculated: Y(NB)={y_nb_avg:.8f}, Y(TI)={y_ti_avg:.8f}")

print("\nThese are DIFFERENT! Let me check if it's weighted by phase amounts...")

# Phase amounts
np0 = 0.34098573
np1 = 0.65901427
total = np0 + np1

print(f"\nPhase amounts: NP[0]={np0}, NP[1]={np1}")

# Weighted average
y_nb_weighted = (0.6122449 * np0 + 0.59366427 * np1) / total
y_ti_weighted = (0.3877551 * np0 + 0.40633573 * np1) / total

print(f"\nWeighted average:")
print(f"Y(NB) = {y_nb_weighted:.8f}")
print(f"Y(TI) = {y_ti_weighted:.8f}")

print("\nBut the CPU actually shows something else!")
print("This suggests the CPU does some OTHER consolidation logic.")