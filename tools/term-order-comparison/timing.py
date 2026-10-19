"""
CPU time of each stage on one large database, for comparing the cost of the fixes.

usage: python timing.py DATABASE stages|eq
  stages: load the Database, build every Model, then compile every PhaseRecord function by
          evaluating each once (compilation is lazy, so building PhaseRecords alone compiles nothing)
  eq:     load, then one equilibrium through Workspace, from a cold start
Prints one JSON object. Components and conditions are the MPEA (Al-C-Co-Cr-Fe-Mn-Ni-V) ones.
"""
import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
COMPS = ["AL", "C", "CO", "CR", "FE", "MN", "NI", "V", "VA"]

db_path, mode = sys.argv[1], sys.argv[2]
out = {}
t0 = time.process_time()
from pycalphad import Database, Model, Workspace, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.utils import filter_phases, unpack_species
t1 = time.process_time()
out["import"] = t1 - t0
dbf = Database(db_path)
t2 = time.process_time()
out["load"] = t2 - t1
phases = sorted(filter_phases(dbf, unpack_species(dbf, COMPS)))
if mode == "stages":
    models = {p: Model(dbf, COMPS, p) for p in phases}
    t3 = time.process_time()
    out["models"] = t3 - t2
    prf = PhaseRecordFactory(dbf, COMPS, {v.N, v.P, v.T}, models)
    for p in phases:
        prx = prf.get(p)
        nv = prx.num_statevars + prx.phase_dof
        dof = np.full(nv, 0.5)
        dof[:3] = [1.0, 101325.0, 1000.0]
        for f, sh in [(prx.obj, (1,)), (prx.formulaobj, (1,)), (prx.formulagrad, (nv,)), (prx.formulahess, (nv, nv)),
                      (prx.internal_cons_func, (prx.num_internal_cons,)), (prx.internal_cons_jac, (prx.num_internal_cons, nv)),
                      (prx.internal_cons_hess, (prx.num_internal_cons, nv, nv))]:
            f(np.zeros(sh), dof)
        for ci in range(len(prx.nonvacant_elements)):
            for f, sh in [(prx.mass_obj, (1,)), (prx.formulamole_obj, (1,)), (prx.formulamole_grad, (nv,)),
                          (prx.formulamole_hess, (nv, nv))]:
                f(np.zeros(sh), dof, ci)
    out["compile"] = time.process_time() - t3
    out["n_phases"] = len(phases)
elif mode == "eq":
    conds = {v.T: 1200, v.P: 101325, v.N: 1, v.X("CO"): 0.2, v.X("CR"): 0.2, v.X("FE"): 0.2, v.X("MN"): 0.2,
             v.X("AL"): 0.05, v.X("C"): 0.005, v.X("V"): 0.01}
    eq = Workspace(dbf, COMPS, phases, conds).eq
    out["equilibrium_total"] = time.process_time() - t2
    out["GM"] = float(np.squeeze(eq.GM))
    out["phases"] = sorted(str(p) for p in np.squeeze(eq.Phase) if str(p))
else:
    raise SystemExit(__doc__)
print(json.dumps(out))
