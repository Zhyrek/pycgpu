from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import instantiate_models
from pycalphad.core.workspace import Workspace
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory

# Setup
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
models = instantiate_models(dbf, comps, phases)

# Check what state variables are used
print("Model state variables:")
for phase, mod in models.items():
    print(f"  {phase}: {mod.state_variables}")

# Create workspace
wks = Workspace(database=dbf, components=comps, phases=phases, conditions={v.T: 1800, v.P: 101325, v.X('TI'): 0.3})

print("\nWorkspace phase_record_factory state variables:")
if wks.phase_record_factory:
    print(f"  {wks.phase_record_factory.state_variables}")
else:
    print("  phase_record_factory not set")

# Run equilibrium to trigger GPU code generation
print("\nRunning GPU equilibrium...")
eq = equilibrium(dbf, comps, phases, {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}, 
                 calc_opts={'pdens': 10}, gpu=True)