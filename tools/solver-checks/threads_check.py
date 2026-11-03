"""
Threads sharing one PhaseRecordFactory: results must be bit-identical to a serial run.

    PYTHONHASHSEED=0 PYTHONPATH=<checkout> python threads_check.py [repeats] [threads]

For each repeat, a fresh PhaseRecordFactory (no functions built yet) is shared by all threads, so that
threads build functions of the same PhaseRecord concurrently (the FastFunctionFactory cache race), and
the solver runs concurrently (the minimizer and hull without the GIL). Also times serial and threaded runs.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def main(repeats=5, threads=4):
    import pycalphad
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
    from pycalphad.core.utils import filter_phases, unpack_species, instantiate_models
    import pycalphad.tests.databases
    db_dir = os.path.dirname(pycalphad.tests.databases.__file__)
    print(pycalphad.__file__)
    systems = [
        ('alcocrni.tdb', ['AL', 'CR', 'NI', 'VA'], {v.T: 1300, v.P: 101325, v.N: 1, v.X('CR'): np.linspace(0.02, 0.9, 12)},
         v.X('AL'), np.linspace(0.02, 0.9, 16)),
        ('alzn_mey.tdb', ['AL', 'ZN', 'VA'], {v.T: np.linspace(400, 800, 5), v.P: 101325, v.N: 1},
         v.X('ZN'), np.linspace(0.01, 0.99, 64)),
    ]
    failures = 0
    for db_name, comps, base_conds, key, values in systems:
        dbf = Database(os.path.join(db_dir, db_name))
        phases = filter_phases(dbf, unpack_species(dbf, comps))
        chunks = np.array_split(values, 16)

        def factory():
            models = instantiate_models(dbf, comps, phases)
            return models, PhaseRecordFactory(dbf, comps, [v.N, v.P, v.T], models)

        models, prf = factory()
        t0 = time.perf_counter()
        serial = [equilibrium(dbf, comps, phases, {**base_conds, key: chunk}, model=models, phase_records=prf)
                  for chunk in chunks]
        t_serial = time.perf_counter() - t0
        times = []
        for rep in range(repeats):
            models, prf = factory()
            def solve(chunk):
                return equilibrium(dbf, comps, phases, {**base_conds, key: chunk}, model=models, phase_records=prf)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(threads) as ex:
                threaded = list(ex.map(solve, chunks))
            times.append(time.perf_counter() - t0)
            for a, b in zip(serial, threaded):
                for name in ['GM', 'MU', 'NP', 'X', 'Y']:
                    if np.asarray(a[name].values).tobytes() != np.asarray(b[name].values).tobytes():
                        failures += 1
                        print(f'{db_name} repeat {rep}: {name} differs from serial')
                        break
        print(f'{db_name}: serial (first, includes building) {t_serial:.2f} s; {threads} threads (fresh factory each) '
              f'{np.median(times):.2f} s median, {min(times):.2f}-{max(times):.2f}', flush=True)
    print('failures:', failures)
    return failures


if __name__ == '__main__':
    args = [int(a) for a in sys.argv[1:]]
    sys.exit(1 if main(*args) else 0)
