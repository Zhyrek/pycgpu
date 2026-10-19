"""
Bit-identity of phase records under scrambled term and parameter order.

For every phase of every database in pycalphad/tests/databases, compare everything a PhaseRecord
evaluates (obj, formulaobj/grad/hess, internal constraints and their derivatives, mass_obj and
formulamole_obj/grad/hess per component) at 10 reproducible points, between the database as read
and a copy with:
  - the FUNCTIONs in a shuffled order, the terms of each of their sums shuffled
  - the parameters in a shuffled order, the terms of each of their sums shuffled
  - with MODE=addmul, the factors of each product shuffled as well

The functions are compiled from scratch for each phase (build_functions caches cleared), as in a new
process. Shuffles are seeded by database name, so every checkout scrambles the same way.

usage: python scramble.py add|addmul OUT.jsonl
Run it with the checkout under test first on PYTHONPATH. One JSON line per phase.
"""
import copy
import glob
import json
import os
import random
import sys
import time
import warnings

import numpy as np
from symengine import Add, Basic, Mul, Piecewise

import pycalphad.tests.databases
from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.codegen.sympydiff_utils import build_constraint_functions, build_functions

warnings.filterwarnings("ignore")


def scramble(expr, rng, mode):
    if not isinstance(expr, Basic):
        return expr
    memo = {}

    def rebuild(e):
        if not e.args:
            return e
        if e in memo:
            return memo[e]
        args = [rebuild(arg) for arg in e.args]
        if isinstance(e, Add):
            rng.shuffle(args)
            out = Add(*args)
        elif isinstance(e, Mul) and mode == "addmul":
            rng.shuffle(args)
            out = Mul(*args)
        elif isinstance(e, Piecewise):
            out = Piecewise(*zip(args[0::2], args[1::2]))
        else:
            out = type(e)(*args)
        memo[e] = out
        return out

    return rebuild(expr)


def scrambled_database(dbf, seed, mode):
    rng = random.Random(seed)
    sdb = copy.deepcopy(dbf)
    sdb.symbols = {name: scramble(dbf.symbols[name], rng, mode)
                   for name in rng.sample(list(dbf.symbols), len(dbf.symbols))}
    params = [dict(param) for param in sdb._parameters.all()]
    rng.shuffle(params)
    for param in params:
        param["parameter"] = scramble(param["parameter"], rng, mode)
    sdb._parameters.truncate()
    sdb._parameters.insert_multiple(params)
    return sdb


def phase_record_outputs(dbf, comps, phase_name, num_points=10):
    """Same evaluation as _phase_record_outputs in Bocklund's test_model.py"""
    build_functions.cache_clear()
    build_constraint_functions.cache_clear()
    mod = Model(dbf, comps, phase_name)
    prx = PhaseRecordFactory(dbf, comps, {v.N, v.P, v.T}, {phase_name: mod}).get(phase_name)
    num_vars = prx.num_statevars + prx.phase_dof
    rng = np.random.default_rng(1769)
    sublattice_indices = np.array([sf.sublattice_index for sf in mod.site_fractions])
    outputs = []
    for _ in range(num_points):
        dof = rng.uniform(size=num_vars)
        dof[:3] = [1.0, 101325.0, 300.0 + 2000.0 * dof[2]]  # N, P, T
        for sublattice_index in np.unique(sublattice_indices):
            columns = 3 + np.nonzero(sublattice_indices == sublattice_index)[0]
            dof[columns] /= dof[columns].sum()
        for func, shape in [(prx.obj, (1,)), (prx.formulaobj, (1,)), (prx.formulagrad, (num_vars,)),
                            (prx.formulahess, (num_vars, num_vars)), (prx.internal_cons_func, (prx.num_internal_cons,)),
                            (prx.internal_cons_jac, (prx.num_internal_cons, num_vars)),
                            (prx.internal_cons_hess, (prx.num_internal_cons, num_vars, num_vars))]:
            out = np.zeros(shape)
            func(out, dof)
            outputs.append(out)
        for comp_idx in range(len(prx.nonvacant_elements)):
            for func, shape in [(prx.mass_obj, (1,)), (prx.formulamole_obj, (1,)), (prx.formulamole_grad, (num_vars,)),
                                (prx.formulamole_hess, (num_vars, num_vars))]:
                out = np.zeros(shape)
                func(out, dof, comp_idx)
                outputs.append(out)
    return outputs


def main(mode, out_path):
    db_dir = os.path.dirname(pycalphad.tests.databases.__file__)
    paths = sorted(glob.glob(os.path.join(db_dir, "*.tdb")) + glob.glob(os.path.join(db_dir, "*.dat")))
    with open(out_path, "w") as fo:
        for path in paths:
            db_name = os.path.basename(path)
            dbf = Database(path)
            sdb = scrambled_database(dbf, db_name, mode)
            comps = sorted(dbf.elements)
            for phase_name in sorted(dbf.phases):
                t0 = time.time()
                try:
                    expected = phase_record_outputs(dbf, comps, phase_name)
                    actual = phase_record_outputs(sdb, comps, phase_name)
                    # NaN == NaN here, as in np.testing.assert_array_equal (some mass_obj outputs are NaN)
                    num_differ = sum(not np.array_equal(x, y, equal_nan=True) for x, y in zip(expected, actual))
                    max_rel = max((np.nanmax(np.abs(x - y) / np.maximum(np.abs(x), 1e-300))
                                   for x, y in zip(expected, actual) if x.size), default=0.0)
                    rec = {"db": db_name, "phase": phase_name, "differ": num_differ > 0,
                           "n_out_differ": int(num_differ), "n_out": len(expected), "maxrel": float(max_rel)}
                except Exception as e:  # Models that cannot be built with every element as a component
                    rec = {"db": db_name, "phase": phase_name, "err": repr(e)[:200]}
                rec["t"] = round(time.time() - t0, 2)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
