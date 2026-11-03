"""
Condition sweep for the indefinite reduced Hessian fix (pycalphad gh-746): GM, convergence and a derivative
(HM.T) at each condition, to compare two checkouts.

    PYTHONHASHSEED=0 PYTHONPATH=<checkout> python hessian_sweep.py <out.npz> [<databases dir>]
    python hessian_sweep.py --compare <before.npz> <after.npz>

MQMQA liquids (Cu-Fe-C of gh-746, Cu-Ni and Cu-Mg-Ni test liquids, K-F-Ni), the ordered phases of Al-Ni
and Al-Fe (disordered configurations of ordered phases are saddle points below the ordering temperature),
and Al-Cr-Ni. CuFeC-Kang.dat is added by the fix, so pass the databases directory of the fixed checkout
when running the one before it. A condition counts as converged when GM is finite (eqsolver writes NaN
otherwise); conditions whose mole fractions sum above one are NaN in both.
"""
import os
import sys
import time
import warnings

import numpy as np


def systems():
    from pycalphad import variables as v
    yield ('cufec_kang', 'CuFeC-Kang.dat', ['C', 'CU', 'FE', 'VA'], ['BCC_A2', 'C_GRAPHITE(S)', 'FCC_A1', 'LIQUID'],
           {v.T: [1300, 1400, 1500, 1600, 1700], v.P: 101325, v.N: 1,
            v.X('C'): (0.05, 0.6, 0.1), v.X('FE'): (0.05, 0.6, 0.1)})
    yield ('mqmqa_cuni', 'MQMQA-tern-tests.dat', ['CU', 'NI', 'VA'], ['L_SUBG_0', 'L_SUBG_1', 'L_SUBG_2'],
           {v.T: [800, 1000, 1400], v.P: 101325, v.N: 1, v.X('NI'): (0.01, 1, 0.07)})
    yield ('mqmqa_cumgni', 'MQMQA-tern-tests.dat', ['CU', 'MG', 'NI', 'VA'], ['TERN_XS_123'],
           {v.T: 1000, v.P: 101325, v.N: 1, v.X('MG'): (0.01, 0.9, 0.1), v.X('NI'): (0.01, 0.9, 0.1)})
    yield ('kfnif2', 'Ocadiz-Flores.dat', ['K', 'F', 'NI'],
           ['F2(G)', 'KF_S1(S)', 'NIF2_S1(S)', 'NIK2F4_S1(S)', 'LIQUID2', 'NIKF3_S1(S)'],
           {v.T: [1100, 1300, 1450], v.P: 101325, v.N: 1, v.X('NI'): (0.02, 0.33, 0.03),
            v.X('F'): [0.52, 0.58334, 0.64]})
    yield ('alni_dupin', 'alni_dupin_2001.tdb', ['AL', 'NI', 'VA'], None,
           {v.T: [600, 900, 1200, 1500, 1800], v.P: 101325, v.N: 1, v.X('NI'): (0.02, 0.99, 0.04)})
    yield ('alfe_sundman', 'Al-Fe_sundman2009.tdb', ['AL', 'FE', 'VA'], None,
           {v.T: [600, 900, 1200, 1500], v.P: 101325, v.N: 1, v.X('AL'): (0.02, 0.99, 0.04)})
    yield ('alcrni_1300', 'alcocrni.tdb', ['AL', 'CR', 'NI', 'VA'], None,
           {v.T: 1300, v.P: 101325, v.N: 1, v.X('AL'): (0.02, 0.9, 0.08), v.X('CR'): (0.02, 0.9, 0.08)})


def run(db_dir=None):
    import pycalphad
    from pycalphad import Database, Workspace
    from pycalphad.core.utils import filter_phases, unpack_species
    import pycalphad.tests.databases
    db_dir = db_dir or os.path.dirname(pycalphad.tests.databases.__file__)
    out = {}
    for label, db_name, comps, phases, conds in systems():
        dbf = Database(os.path.join(db_dir, db_name))
        if phases is None:
            phases = filter_phases(dbf, unpack_species(dbf, comps))
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wks = Workspace(dbf, comps, phases, conds)
            gm, hmt = wks.get('GM', 'HM.T')
        eq = wks.eq
        out[f'{label}/GM'] = np.asarray(gm, dtype=float)
        out[f'{label}/HM.T'] = np.asarray(hmt, dtype=float)
        out[f'{label}/NP'] = np.asarray(eq.NP)
        out[f'{label}/Phase'] = np.asarray(eq.Phase).astype(str)
        print(f'{label}: {np.asarray(gm).size} conditions, {time.perf_counter() - t0:.1f} s', flush=True)
    print(pycalphad.__file__)
    return out


def compare(path_a, path_b, gm_tol=1e-6):
    a, b = np.load(path_a), np.load(path_b)
    labels = sorted({k.split('/')[0] for k in a.files})
    totals = dict(conditions=0, higher=0, lower=0, newly_converged=0, newly_unconverged=0, hmt_changed=0)
    for label in labels:
        gm_a, gm_b = a[f'{label}/GM'].ravel(), b[f'{label}/GM'].ravel()
        conv_a, conv_b = np.isfinite(gm_a), np.isfinite(gm_b)
        hmt_a, hmt_b = a[f'{label}/HM.T'].ravel(), b[f'{label}/HM.T'].ravel()
        diff = gm_b - gm_a
        finite = np.isfinite(diff)
        higher = finite & (diff > gm_tol)
        lower = finite & (diff < -gm_tol)
        same_gm = finite & (np.abs(diff) <= gm_tol)
        # derivative changed while the equilibrium (GM) is the same
        hmt_changed = same_gm & ~np.isclose(hmt_a, hmt_b, rtol=1e-6, atol=1e-6, equal_nan=True)
        row = dict(conditions=gm_a.size, higher=int(higher.sum()), lower=int(lower.sum()),
                   newly_converged=int((~conv_a & conv_b).sum()), newly_unconverged=int((conv_a & ~conv_b).sum()),
                   hmt_changed=int(hmt_changed.sum()))
        for key in totals:
            totals[key] += row[key]
        extra = ''
        if row['higher']:
            extra += f' max increase {diff[higher].max():.4g} J/mol'
        if row['lower']:
            extra += f' max decrease {-diff[lower].min():.4g} J/mol'
        print(f'{label}: {row}{extra}')
    print(f'total: {totals}')
    return totals


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        compare(sys.argv[2], sys.argv[3])
    else:
        np.savez(sys.argv[1], **run(sys.argv[2] if len(sys.argv) > 2 else None))
