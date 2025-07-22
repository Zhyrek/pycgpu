#!/usr/bin/env python
"""Trace when phase 0 is removed in CPU vs GPU."""

import re

with open('full_debug_output.txt', 'r') as f:
    cpu_content = f.read()

with open('full_debug_output2.txt', 'r') as f:
    gpu_content = f.read()

print("=== CPU PHASE 0 TRACKING ===")
# Find CPU phase 0 amounts by iteration
cpu_matches = re.findall(r'(?:CPU.*iteration (\d+).*?)phase_amt=([\d.e+-]+).*?system_amt', cpu_content, re.DOTALL)
for i, (iter_num, amt) in enumerate(cpu_matches[:10]):
    if float(amt) < 1e-9:
        print(f"Iteration {iter_num}: phase_amt = {amt}")

# Find when CPU has 3 rows (1 phase)
cpu_matrix = re.findall(r'CPU MATRIX.*iteration.*(\d+).*rows=(\d+)', cpu_content)
for iter_num, rows in cpu_matrix[:5]:
    print(f"Iteration {iter_num}: matrix rows = {rows}")

print("\n=== GPU PHASE 0 TRACKING ===")
# Find GPU phase 0 status
gpu_phase0 = re.findall(r'GPU.*Phase 0.*?phase_amt=([\d.e+-]+).*?iteration.*?(\d+)', gpu_content, re.DOTALL)
for amt, iter_num in gpu_phase0[:10]:
    if float(amt) < 1e-9 or amt == "0.000000":
        print(f"Iteration {iter_num}: phase_amt = {amt}")

# Find when phase 0 is marked for removal
removal = re.search(r'Phase 0 marked for removal.*?iteration.*?(\d+)', gpu_content, re.DOTALL)
if removal:
    print(f"Phase 0 marked for removal in iteration: {removal.group(1)}")
else:
    # Try different pattern
    removal2 = re.findall(r'(?:iteration|ITERATION) (\d+).*?Phase 0 marked for removal', gpu_content, re.DOTALL)
    if removal2:
        print(f"Phase 0 marked for removal in iteration: {removal2[0]}")