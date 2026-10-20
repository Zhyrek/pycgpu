# Solver sweep: nucleation grace period vs. develop, against cg

Equilibria over fixed condition grids with three solvers. cg is the reference, since it is never above develop by more than 1 J/mol on these grids.
- **develop:** upstream develop f11ec4c9; no solver changes since.
- **grace:** this branch, Bocklund's `nucleation-grace-period` (922e4c4a) on develop.
- **cg:** the column-generation backend from the fork's `dense-grid` branch.

```bash
PYTHONPATH=<develop checkout> python tools/solver-sweep/sweep.py develop results/develop_set1.npz
PYTHONPATH=<this checkout>    python tools/solver-sweep/sweep.py grace   results/grace_set1.npz
PYTHONPATH=<dense-grid>       python tools/solver-sweep/sweep.py cg      results/cg_set1.npz cg
SWEEP_SET=2 ...  # the same three runs, written to *_set2.npz
python tools/solver-sweep/compare.py
```

Set 1 is the 14 grids of the cg-vs-develop classification, plus Al-Cr-Ni at 1300 K and COST507 Al-Cu-Mg-Si-Zn at 500 K. Set 2 is multicomponent:
- COST507 Al-Cu-Mg-Si-Zn at 450–700 K, and a Si-Zn plane;
- COST507 with 10 components at 650 and 850 K;
- Al-Co-Cr-Ni at 1100–1500 K.

A condition counts as **right** within 0.1 J/mol of cg, and **wrong** above that. Conditions where cg has no answer are left out; these are the infeasible compositions.

| 1,957 conditions | right | wrong | of which > 100 J/mol | NaN |
|---|---|---|---|---|
| develop | 1769 | 156 | 26 | 32 |
| grace | 1777 | 147 | 21 | 33 |

The grace period changes the outcome at 36 conditions:

| change | conditions |
|---|---|
| NaN → right | 9 |
| wrong → right | 8 |
| wrong → NaN | 7 |
| right → NaN | 6 |
| NaN → wrong | 3 |
| right → wrong | 3 |

All the changes are in multicomponent COST507 grids, plus 5 conditions on the edges of ternary sections where one element is exactly zero. In the binaries and the Al-Co-Cr-Ni grids the results are identical. The grace period does not fix the Al-Cr-Ni L1₂ + L1₂ case either: the second set survives, but its composition drifts into the first and the two are merged.
