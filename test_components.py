from pycalphad import Database, equilibrium
from pycalphad.core.workspace import Workspace
import numpy as np

# Create workspace to check component handling
dbf = Database("Al-Cu-Fe.tdb")
wks = Workspace(dbf, ["AL", "CU", "FE", "VA"], ["LIQUID"], 
                {"T": 973.15, "P": 101325, "X(AL)": 0.5, "X(CU)": 0.2})

print("Components in workspace:")
for i, comp in enumerate(wks.components):
    print(f"  {i}: {comp}")

print(f"\nTotal components: {len(wks.components)}")
non_va = [c for c in wks.components if c != "VA"]
print(f"Non-VA components: {non_va}")
print(f"Number of non-VA: {len(non_va)}")