"""
Structure of every phase's model contributions in every test database, for comparison between checkouts.

    PYTHONHASHSEED=0 PYTHONPATH=<checkout> python model_fingerprint.py <out.json>
    python model_fingerprint.py --compare <a.json> <b.json>

Each phase is built with the species of its own constituents. Records str() of each entry of Model.models
(the stored expression, so a different representative of equal subexpressions or a different term order shows),
or the exception if building fails.
"""
import glob
import hashlib
import json
import os
import sys
import warnings


def run():
    import pycalphad
    from pycalphad import Database, Model
    import pycalphad.tests.databases
    db_dir = os.path.dirname(pycalphad.tests.databases.__file__)
    out = {}
    for path in sorted(glob.glob(os.path.join(db_dir, '*.tdb')) + glob.glob(os.path.join(db_dir, '*.dat'))):
        name = os.path.basename(path)
        try:
            dbf = Database(path)
        except Exception as e:
            out[name] = f'database error: {type(e).__name__}: {e}'
            continue
        for phase_name in sorted(dbf.phases):
            phase = dbf.phases[phase_name]
            comps = sorted({str(sp) for subl in phase.constituents for sp in subl} | {'VA'})
            key = f'{name}/{phase_name}'
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mod = Model(dbf, comps, phase_name)
                out[key] = {k: hashlib.sha256(str(val).encode()).hexdigest() for k, val in mod.models.items()}
            except Exception as e:
                out[key] = f'error: {type(e).__name__}: {e}'
    print(pycalphad.__file__, len(out), 'entries')
    return out


def compare(path_a, path_b):
    a, b = json.load(open(path_a)), json.load(open(path_b))
    different = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    for k in different:
        print(k, a.get(k), b.get(k))
    print(f'{len(set(a) | set(b))} phases, {len(different)} different')
    return len(different)


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
    json.dump(run(), open(sys.argv[1], 'w'), indent=0, sort_keys=True)
