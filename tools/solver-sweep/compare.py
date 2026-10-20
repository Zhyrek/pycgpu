"""Compare sweep results against cg: right (within 0.1 J/mol of cg), wrong (above it), NaN.
usage: python compare.py [RESULTS_DIR]  (expects develop_set*.npz, grace_set*.npz, cg_set*.npz)"""
import os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
TOL = 0.1
totals = {s: dict(right=0, wrong=0, big=0, nan=0) for s in ('develop', 'grace')}
transitions = {}
print(f"{'system':18s} {'n':>5s}  develop right/wrong/NaN   grace right/wrong/NaN")
for suffix in ('set1', 'set2'):
    A = {k: np.load(os.path.join(d, f'{k}_{suffix}.npz')) for k in ('develop', 'grace', 'cg')}
    for key in [k for k in A['cg'].files if k.endswith('/GM')]:
        cg = A['cg'][key]
        feasible = np.isfinite(cg)  # cg returns NaN only at infeasible compositions in these grids
        cg = cg[feasible]
        states, line = {}, []
        for s in ('develop', 'grace'):
            x = A[s][key][feasible]
            nan = np.isnan(x)
            wrong = ~nan & (x - cg > TOL)
            states[s] = np.where(nan, 'NaN', np.where(wrong, 'wrong', 'right'))
            t = totals[s]
            t['right'] += int((~nan & ~wrong).sum()); t['wrong'] += int(wrong.sum())
            t['big'] += int((~nan & (x - cg > 100)).sum()); t['nan'] += int(nan.sum())
            line.append(f"{int((~nan & ~wrong).sum()):5d}/{int(wrong.sum()):3d}/{int(nan.sum()):3d}")
        for a, b in zip(states['develop'], states['grace']):
            if a != b:
                transitions[f'{a} -> {b}'] = transitions.get(f'{a} -> {b}', 0) + 1
        print(f"{key[:-3]:18s} {int(feasible.sum()):5d}  {line[0]:>22s}   {line[1]:>20s}")
for s, t in totals.items():
    print(f"{s}: right {t['right']}, wrong {t['wrong']} (>100 J/mol: {t['big']}), NaN {t['nan']}")
print('develop -> grace:', dict(sorted(transitions.items())))
