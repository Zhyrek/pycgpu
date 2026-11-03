"""
Interleaved timings of pycalphad workloads in several checkouts.

    python bench.py --repeats 5 --workloads eq_alni,eq_alcrni label1=<checkout1> label2=<checkout2> ...

Each run is a fresh process (PYTHONHASHSEED=0, PYTHONPATH=<checkout>) that warms up (code generation, first
solve) and then times the workload; runs go round-robin over the checkouts, so that drifts in machine load
affect all of them alike. Prints median and range per workload and checkout.

    python bench.py --run <workload>    (one timing, in the current environment; prints seconds)
"""
import os
import subprocess
import sys
import time

import numpy as np


def _db(name):
    from pycalphad import Database
    import pycalphad.tests.databases
    return Database(os.path.join(os.path.dirname(pycalphad.tests.databases.__file__), name))


def _phases(dbf, comps):
    from pycalphad.core.utils import filter_phases, unpack_species
    return filter_phases(dbf, unpack_species(dbf, comps))


def _eq_workload(db_name, comps, conds, warm_conds, threads=0, chunk_key=None):
    from pycalphad import equilibrium
    dbf = _db(db_name)
    phases = _phases(dbf, comps)
    if threads:
        # One PhaseRecordFactory shared by the threads, with its functions built by the warm-up call. (With a
        # factory per equilibrium() call, PhaseRecordFactory.get's lru_cache, shared by all factories, can evict
        # the records of a factory still in use; looking one up again with a numpy.str_ phase name fails.)
        from concurrent.futures import ThreadPoolExecutor
        from pycalphad import variables as v
        from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
        from pycalphad.core.utils import instantiate_models
        models = instantiate_models(dbf, comps, phases)
        prf = PhaseRecordFactory(dbf, comps, [v.N, v.P, v.T], models)
        equilibrium(dbf, comps, phases, warm_conds, model=models, phase_records=prf)
        values = np.asarray(conds[chunk_key])
        chunks = np.array_split(values, 16)
        def solve(chunk):
            return equilibrium(dbf, comps, phases, {**conds, chunk_key: chunk}, model=models, phase_records=prf)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(solve, chunks))
        return time.perf_counter() - t0
    equilibrium(dbf, comps, phases, warm_conds)
    t0 = time.perf_counter()
    equilibrium(dbf, comps, phases, conds)
    return time.perf_counter() - t0


def workload(name):
    from pycalphad import variables as v
    base = {v.P: 101325, v.N: 1}
    if name == 'eq_alni':
        # Al-Ni (Dupin 2001), 16 T x 50 x
        return _eq_workload('alni_dupin_2001.tdb', ['AL', 'NI', 'VA'],
                            {**base, v.T: np.linspace(1000, 1750, 16), v.X('NI'): np.linspace(0.01, 0.99, 50)},
                            {**base, v.T: 1300, v.X('NI'): 0.4})
    if name == 'eq_alfe':
        # Al-Fe (Sundman 2009), 11 T x 30 x
        return _eq_workload('Al-Fe_sundman2009.tdb', ['AL', 'FE', 'VA'],
                            {**base, v.T: np.linspace(700, 1700, 11), v.X('AL'): np.linspace(0.01, 0.99, 30)},
                            {**base, v.T: 1300, v.X('AL'): 0.4})
    if name in ('eq_alcrni', 'eq_alcrni_threads4'):
        # Al-Cr-Ni at 1300 K, 80 x 80 grid (6400 conditions, about half with X(AL) + X(CR) > 1, skipped)
        x = np.linspace(0.01, 0.99, 80)
        return _eq_workload('alcocrni.tdb', ['AL', 'CR', 'NI', 'VA'],
                            {**base, v.T: 1300, v.X('AL'): x, v.X('CR'): x},
                            {**base, v.T: 1300, v.X('AL'): 0.3, v.X('CR'): 0.3},
                            threads=4 if name.endswith('threads4') else 0, chunk_key=v.X('AL'))
    if name in ('eq_alzn', 'eq_alzn_threads4'):
        # Al-Zn (Mey), 5 T x 1000 x
        return _eq_workload('alzn_mey.tdb', ['AL', 'ZN', 'VA'],
                            {**base, v.T: np.linspace(400, 800, 5), v.X('ZN'): np.linspace(0.001, 0.999, 1000)},
                            {**base, v.T: 600, v.X('ZN'): 0.3},
                            threads=4 if name.endswith('threads4') else 0, chunk_key=v.X('ZN'))
    if name == 'eq_cost507':
        # COST507, 5 components at 500 K
        return _eq_workload('COST507.tdb', ['AL', 'CU', 'MG', 'SI', 'ZN', 'VA'],
                            {**base, v.T: 500, v.X('CU'): np.linspace(0.01, 0.1, 4), v.X('MG'): np.linspace(0.01, 0.2, 4),
                             v.X('SI'): 0.05, v.X('ZN'): 0.05},
                            {**base, v.T: 500, v.X('CU'): 0.02, v.X('MG'): 0.05, v.X('SI'): 0.05, v.X('ZN'): 0.05})
    if name == 'tdb_parse':
        # Parse four databases
        from pycalphad import Database
        import pycalphad.tests.databases
        d = os.path.dirname(pycalphad.tests.databases.__file__)
        t0 = time.perf_counter()
        for db_name in ['COST507.tdb', 'alcocrni.tdb', 'Al-Fe_sundman2009.tdb', 'alni_dupin_2001.tdb']:
            Database(os.path.join(d, db_name))
        return time.perf_counter() - t0
    if name == 'model_build':
        # Models of every phase of COST507 with five components
        from pycalphad import Model
        dbf = _db('COST507.tdb')
        comps = ['AL', 'CU', 'MG', 'SI', 'ZN', 'VA']
        phases = _phases(dbf, comps)
        t0 = time.perf_counter()
        for phase in phases:
            Model(dbf, comps, phase)
        return time.perf_counter() - t0
    if name == 'calc_alcrni':
        # calculate (sampling and energies) of all Al-Cr-Ni phases, 10 temperatures
        from pycalphad import calculate
        from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
        dbf = _db('alcocrni.tdb')
        comps = ['AL', 'CR', 'NI', 'VA']
        phases = _phases(dbf, comps)
        calculate(dbf, comps, phases, T=1000, P=101325, N=1)
        t0 = time.perf_counter()
        for T in np.linspace(800, 1700, 10):
            calculate(dbf, comps, phases, T=T, P=101325, N=1)
        return time.perf_counter() - t0
    raise ValueError(name)


def main(argv):
    repeats = 5
    workloads = []
    checkouts = []
    i = 0
    while i < len(argv):
        if argv[i] == '--repeats':
            repeats = int(argv[i + 1]); i += 2
        elif argv[i] == '--workloads':
            workloads = argv[i + 1].split(','); i += 2
        else:
            label, path = argv[i].split('=', 1)
            checkouts.append((label, path)); i += 1
    results = {(w, label): [] for w in workloads for label, _ in checkouts}
    for rep in range(repeats):
        for w in workloads:
            order = checkouts if rep % 2 == 0 else checkouts[::-1]
            for label, path in order:
                env = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=path)
                out = subprocess.run([sys.executable, __file__, '--run', w], env=env, capture_output=True, text=True)
                if out.returncode != 0:
                    # e.g. the FastFunctionFactory cache race with threads crashes the process
                    print(f'{w} failed in {label} (exit {out.returncode}): {out.stderr[-500:]}', flush=True)
                    results[(w, label)].append(np.nan)
                    continue
                results[(w, label)].append(float(out.stdout.strip().splitlines()[-1]))
        print(f'repeat {rep + 1}/{repeats} done', flush=True)
    for w in workloads:
        ref = None
        for label, _ in checkouts:
            t = np.array(results[(w, label)])
            failed = int(np.isnan(t).sum())
            t = t[~np.isnan(t)]
            if t.size == 0:
                print(f'{w:22s} {label:12s} all {failed} runs failed')
                continue
            med = np.median(t)
            ref = med if ref is None else ref
            print(f'{w:22s} {label:12s} median {med:7.3f} s  range {t.min():7.3f}-{t.max():7.3f}  '
                  f'ratio to {checkouts[0][0]} {med / ref:5.3f}  runs {np.round(t, 3).tolist()}' +
                  (f'  failed runs {failed}' if failed else ''))


if __name__ == '__main__':
    if sys.argv[1] == '--run':
        import pycalphad
        assert os.environ.get('PYTHONPATH', '') == '' or pycalphad.__file__.startswith(os.environ['PYTHONPATH']), \
            pycalphad.__file__
        print(workload(sys.argv[2]))
    else:
        main(sys.argv[1:])
