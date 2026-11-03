"""
Mapping census: maps a fixed set of binary phase diagrams and ternary isothermal sections with the pycalphad on
sys.path and writes every ZPF line and node to a JSON file, for comparison with compare.py.

Run from (or with PYTHONPATH set to) the pycalphad checkout to measure, with a fixed hash seed (the mapping is only
deterministic with one):

    PYTHONHASHSEED=0 PYTHONPATH=/path/to/checkout python census.py out.json [--only Al-Fe,Cr-V]

Databases are taken from the databases directory next to this script (databases that are not in every checkout),
the directories given with --dbdir, or pycalphad/tests/databases of that checkout, in that order.
"""
import argparse
import json
import os
import sys
import time
import warnings
from importlib.resources import files

import numpy as np

import pycalphad
import pycalphad.tests.databases
from pycalphad import Database, variables as v
from pycalphad.core.utils import filter_phases, unpack_species
import pycalphad.mapping as mapping

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, database, components, phases (None: all), T range, x step) for binaries; the second component is the axis
BINARIES = [
    ("Al-Te", "AlTe-19Shi.tdb", ["AL", "TE"], (300, 1500), 0.02),
    ("Cr-V", "CrV-92Lee-LB.tdb", ["CR", "V"], (300, 4000), 0.02),
    ("Cu-Mg", "cumg.tdb", ["CU", "MG"], (300, 1500), 0.02),
    ("Ba-Ca", "BaCa-86Alc.tdb", ["BA", "CA"], (300, 1500), 0.02),
    ("Au-Cu", "AuCu-98Sun-LB.tdb", ["AU", "CU"], (300, 1500), 0.02),
    ("Ga-La", "GaLa-11Idb.tdb", ["GA", "LA"], (300, 2000), 0.02),
    ("Cr-Ta", "CrTa-93Dup-LB.tdb", ["CR", "TA"], (300, 3500), 0.02),
    ("Al-Fe", "alfe.tdb", ["AL", "FE"], (300, 1900), 0.02),
    ("Al-Ni", "alni_dupin_2001.tdb", ["AL", "NI"], (300, 2000), 0.02),
    ("Co-V", "CoV-20Wan.tdb", ["CO", "V"], (300, 2300), 0.02),
    ("Cr-Ti", "crtiv_ghosh.tdb", ["CR", "TI"], (300, 2300), 0.02),
    ("Fe-Nb", "CrFeNb_Jacob2016.tdb", ["FE", "NB"], (800, 2800), 0.02),
]

# (name, database, components (the last two are the axes), T, x step)
TERNARIES = [
    ("Al-Cr-Ni 1300K (alcocrni)", "alcocrni.tdb", ["AL", "CR", "NI"], 1300, 0.05),
    ("Al-Co-Ni 1233K (alcocrni)", "alcocrni.tdb", ["AL", "CO", "NI"], 1233.01, 0.05),
    ("Al-Co-Cr 1334K (alcocrni)", "alcocrni.tdb", ["AL", "CO", "CR"], 1334.04, 0.05),
    ("Al-Cr-Ni 1200K (alcrni)", "alcrni.tdb", ["AL", "CR", "NI"], 1200, 0.05),
    ("Al-Cu-Y 1774.65K", "Al-Cu-Y.tdb", ["AL", "CU", "Y"], 1774.65, 0.05),
]


DB_DIRS = [os.path.join(HERE, "databases")]


def load(db_name):
    for directory in DB_DIRS:
        path = os.path.join(directory, db_name)
        if os.path.exists(path):
            return Database(path)
    return Database(str(files(pycalphad.tests.databases).joinpath(db_name)))


def strategy_class(ternary):
    if hasattr(mapping, "TielineStrategy"):
        return mapping.TielineStrategy
    return mapping.TernaryStrategy if ternary else mapping.BinaryStrategy


def comp_set_record(cs):
    return {"phase": cs.phase_record.phase_name, "X": [float(x) for x in cs.X], "NP": float(cs.NP)}


def point_record(point, axis_vars):
    sets = sorted(point.stable_composition_sets, key=lambda cs: (cs.phase_record.phase_name, tuple(cs.X)))
    return {"T": float(np.squeeze(point.global_conditions[v.T])) if v.T in point.global_conditions else None,
            "sets": [comp_set_record(cs) for cs in sets],
            "fixed": sorted(cs.phase_record.phase_name for cs in point.fixed_composition_sets)}


def run_map(name, db_name, comps, conds, ternary, elements):
    dbf = load(db_name)
    comps = comps + ["VA"]
    phases = filter_phases(dbf, unpack_species(dbf, comps))
    t0 = time.time()
    strategy = strategy_class(ternary)(dbf, comps, phases, conds)
    strategy.do_map()
    elapsed = time.time() - t0
    lines = [{"phases": sorted(z.stable_phases), "fixed": sorted(z.fixed_phases), "status": z.status.name,
              "points": [point_record(p, strategy.axis_vars) for p in z.points]} for z in strategy.zpf_lines]
    nodes = [point_record(n, strategy.axis_vars) for n in strategy.node_queue.nodes if len(n.stable_composition_sets) >= 3]
    axis = {"T": {"lims": [float(x) for x in conds[v.T][:2]], "step": float(conds[v.T][2])} if isinstance(conds[v.T], tuple) else None,
            "x_step": float([val for key, val in conds.items() if isinstance(key, v.X)][0][2])}
    return {"name": name, "database": db_name, "elements": elements, "ternary": ternary, "elapsed": elapsed,
            "axis": axis, "lines": lines, "nodes": nodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--only", default=None, help="comma-separated names")
    parser.add_argument("--dbdir", action="append", default=[], help="another directory of databases")
    args = parser.parse_args()
    DB_DIRS.extend(args.dbdir)
    only = set(args.only.split(",")) if args.only else None
    warnings.simplefilter("ignore")
    results = {"pycalphad": os.path.dirname(pycalphad.__file__), "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "maps": []}
    for name, db_name, comps, T_lims, x_step in BINARIES:
        if only and name not in only:
            continue
        t0, t1 = T_lims
        conds = {v.T: (t0, t1, (t1 - t0) / 40), v.X(comps[1]): (0, 1, x_step), v.P: 101325, v.N: 1}
        results["maps"].append(run_map(name, db_name, comps, conds, False, comps))
        print(f"{name}: {len(results['maps'][-1]['lines'])} lines, {results['maps'][-1]['elapsed']:.0f} s", flush=True)
    for name, db_name, comps, T, x_step in TERNARIES:
        if only and name not in only:
            continue
        conds = {v.X(comps[1]): (0, 1, x_step), v.X(comps[2]): (0, 1, x_step), v.T: T, v.P: 101325, v.N: 1}
        results["maps"].append(run_map(name, db_name, comps, conds, True, comps))
        print(f"{name}: {len(results['maps'][-1]['lines'])} lines, {results['maps'][-1]['elapsed']:.0f} s", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    sys.exit(main())
