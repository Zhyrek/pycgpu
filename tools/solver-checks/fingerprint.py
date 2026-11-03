"""
Equilibrium and calculate outputs of a pycalphad checkout, for byte-for-byte comparison between branches.

    PYTHONHASHSEED=0 PYTHONPATH=<checkout> python fingerprint.py <out.npz> [--quick]
    python fingerprint.py --compare <a.npz> <b.npz>

The checkout must be built in place (python setup.py build_ext --inplace). Systems use databases from
pycalphad/tests/databases of that checkout.
"""
import os
import sys
import time

import numpy as np


def _arrays(prefix, ds, names):
    out = {}
    for name in names:
        if name in ds:
            out[f'{prefix}/{name}'] = np.asarray(ds[name].values)
    return out


def systems(quick=False):
    from pycalphad import variables as v
    T_binary = [1000, 1200, 1400, 1600, 1800] if not quick else [1000, 1600]
    x_step = 0.05 if not quick else 0.2
    yield ('alni_dupin', 'alni_dupin_2001.tdb', ['AL', 'NI', 'VA'], None,
           {v.T: T_binary, v.P: 101325, v.N: 1, v.X('NI'): (0.02, 0.99, x_step)})
    yield ('alfe_sundman', 'Al-Fe_sundman2009.tdb', ['AL', 'FE', 'VA'], None,
           {v.T: [800, 1100, 1400, 1700] if not quick else [1100], v.P: 101325, v.N: 1,
            v.X('AL'): (0.02, 0.99, x_step)})
    yield ('alcrni_1300', 'alcocrni.tdb', ['AL', 'CR', 'NI', 'VA'], None,
           {v.T: 1300, v.P: 101325, v.N: 1, v.X('AL'): (0.02, 0.9, 0.08 if not quick else 0.2),
            v.X('CR'): (0.02, 0.9, 0.08 if not quick else 0.2)})
    yield ('cost507_5comp_500K', 'COST507.tdb', ['AL', 'CU', 'MG', 'SI', 'ZN', 'VA'], None,
           {v.T: 500, v.P: 101325, v.N: 1, v.X('CU'): [0.02, 0.1], v.X('MG'): [0.05, 0.2],
            v.X('SI'): 0.05, v.X('ZN'): 0.05})
    yield ('kfni_mqmqa', 'Ocadiz-Flores.dat', ['K', 'F', 'NI'],
           ['F2(G)', 'KF_S1(S)', 'NIF2_S1(S)', 'NIK2F4_S1(S)', 'LIQUID2', 'NIKF3_S1(S)'],
           {v.T: [1300, 1450], v.P: 101325, v.N: 1, v.X('NI'): [0.1, 0.16667, 0.25], v.X('F'): [0.55, 0.58334]})


def load_database(name):
    from pycalphad import Database
    import pycalphad.tests.databases
    path = os.path.join(os.path.dirname(pycalphad.tests.databases.__file__), name)
    return Database(path)


def fingerprint(quick=False):
    import pycalphad
    from pycalphad import equilibrium, calculate
    from pycalphad.core.utils import filter_phases, unpack_species
    out = {}
    timings = {}
    for label, db_name, comps, phases, conds in systems(quick):
        dbf = load_database(db_name)
        if phases is None:
            phases = filter_phases(dbf, unpack_species(dbf, comps))
        t0 = time.perf_counter()
        eq = equilibrium(dbf, comps, phases, conds)
        timings[label] = time.perf_counter() - t0
        out.update(_arrays(f'eq/{label}', eq, ['GM', 'MU', 'NP', 'X', 'Y', 'Phase']))
        if label in ('alni_dupin', 'alcrni_1300', 'kfni_mqmqa'):
            calc = calculate(dbf, comps, phases, T=[1000, 1500], P=101325, N=1, pdens=200)
            out.update(_arrays(f'calc/{label}', calc, ['GM', 'X', 'Y', 'Phase']))
    out = {k: (val.astype(str) if val.dtype.kind in 'OU' else val) for k, val in out.items()}
    return out, timings, pycalphad.__file__


def compare(path_a, path_b):
    a, b = np.load(path_a), np.load(path_b)
    keys = sorted(set(a.files) | set(b.files))
    different = 0
    for key in keys:
        if key not in a.files or key not in b.files:
            print(f'{key}: only in {"a" if key in a.files else "b"}')
            different += 1
            continue
        va, vb = a[key], b[key]
        if va.shape != vb.shape or va.dtype != vb.dtype:
            print(f'{key}: shape/dtype {va.shape} {va.dtype} vs {vb.shape} {vb.dtype}')
            different += 1
        elif va.tobytes() != vb.tobytes():
            if va.dtype.kind == 'f':
                both = np.isfinite(va) & np.isfinite(vb)
                print(f'{key}: differs; {np.sum(va != vb)} of {va.size} entries, '
                      f'max |diff| {np.max(np.abs(va[both] - vb[both]), initial=0):.3g}')
            else:
                print(f'{key}: differs; {np.sum(va != vb)} of {va.size} entries')
            different += 1
    print(f'{len(keys)} arrays, {different} different')
    return different


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
    arrays, timings, location = fingerprint(quick='--quick' in sys.argv)
    np.savez(sys.argv[1], **arrays)
    print(location)
    print({k: round(t, 2) for k, t in timings.items()})
