import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.utils import instantiate_models
from pycalphad.gpu.gpu_codegen import _nb_formulahess_from_model

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Create workspace 
wks = Workspace(database=dbf, components=comps, phases=phases, 
                conditions={v.T: 1800, v.P: 101325, v.X('TI'): 0.3})

# Get models
models = instantiate_models(dbf, comps, phases)
model = models['BCC_A2']

print("Testing GPU code generation for BCC_A2...")
print(f"Model state variables: {model.state_variables}")
print(f"Model site fractions: {model.site_fractions}")

# Test Hessian code generation
try:
    hess_code = _nb_formulahess_from_model(model, 0, wks, validate=True, verbose=False)
    print(f"\nHessian code generated successfully!")
    print(f"Code length: {len(hess_code)} characters")
    
    # Look for temperature usage in the code
    if 'x[2]' in hess_code and ('2750' in hess_code or '1800' in hess_code):
        print("✓ Generated code uses x[2] for temperature (correct workspace format)")
    elif 'x[0]' in hess_code and ('2750' in hess_code or '1800' in hess_code):
        print("✗ Generated code uses x[0] for temperature (incorrect model format)")
    else:
        print("? Cannot determine temperature variable usage")
        
    # Write code to file for inspection
    with open('test_hessian_code.c', 'w') as f:
        f.write(hess_code)
    print("Generated code written to test_hessian_code.c")
    
except Exception as e:
    print(f"Code generation failed: {e}")
    import traceback
    traceback.print_exc()