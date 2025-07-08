import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
tdb = pyc.Database("NbTi.tdb")
phases = ["LIQUID", "BCC_A2"]
comps = ["NB", "TI", "VA"]

print(f"Number of components: {len(comps)}")
print(f"Components: {comps}")

# Check what MAX_COMPONENTS is
from pycalphad.gpu.gpu_codegen import _get_c_define
max_components = _get_c_define("MAX_COMPONENTS")
print(f"\nMAX_COMPONENTS from _get_c_define: {max_components}")

# Also check the dynamic calculation
from pycalphad.core.workspace import Workspace
wks = Workspace(tdb, comps, phases, {v.X("TI"): 0.5, v.T: 500})
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes
dynamic_sizes = compute_dynamic_kernel_sizes(wks)
print(f"\nDynamic MAX_COMPONENTS: {dynamic_sizes.get('MAX_COMPONENTS', 'not set')}")

# The issue seems to be that we have 3 components but the kernel might be compiled with MAX_COMPONENTS=3
# while the Python code expects MAX_COMPONENTS=4