# Term-order fixes: #739 vs. bit-identical-phase-records

Compares two fixes for compiled functions depending on the order in which an expression's terms were built:

| | where it acts | what it orders |
|---|---|---|
| [pycalphad/pycalphad#739](https://github.com/pycalphad/pycalphad/pull/739) (ce030350) | `build_functions` / `build_constraint_functions`, before compiling | terms of sums and factors of products, by SymEngine hash |
| [bocklund/pycalphad `claude/bit-identical-phase-records`](https://github.com/bocklund/pycalphad/tree/claude/bit-identical-phase-records) (9043dd3a) | `Model`, as it is built | terms of sums in Database FUNCTIONs and parameters, by SymEngine hash; parameters (Redlich-Kister, Kohler-Toop, MQMQA); LNTHETA Einstein temperatures |

Both were measured on develop f11ec4c9, macOS arm64, Python 3.12.

## Reproducing

```bash
tools/term-order-comparison/setup_checkouts.sh /tmp/term-order
tools/term-order-comparison/run_all.sh /tmp/term-order path/to/mpea.tdb
```

`setup_checkouts.sh` makes three worktrees (`develop`, `739`, `bocklund`) and builds the extensions once. #739 predates #741, which edits the same decorator, so the script takes the PR's two files and restores #741's `lru_cache(maxsize=512)`. `run_all.sh` writes `results/` and prints `summarize.py`'s tables. Without the MPEA path, only the scramble test runs (about 2 minutes per branch and mode).

## Scrambled term and parameter order

`scramble.py` takes every phase of every database in `pycalphad/tests/databases`. For each one it compares everything a PhaseRecord evaluates, at 10 points, between the database as read and a copy with its FUNCTIONs and parameters in shuffled order and the terms of every sum shuffled (`add`). The `addmul` mode also shuffles the factors of every product. Comparison is bitwise, with NaN equal to NaN, as in `np.testing.assert_array_equal`. 16 phases cannot be built with every element as a component and are skipped, on all branches.

| phases that differ (of 811) | develop | #739 | Bocklund |
|---|---|---|---|
| sums shuffled | 655 | 5 | 0 |
| sums and products shuffled | 651 | 6 | 0 |

The phases that still differ with #739 are the `cfe_broshe` phases (LIQUID, BCC_A2, GRAPHITE, DIAMOND_A4, CEMENTITE_D011, M7C3_D101) and the `C_A_S_Fe_O_M` LIQUID. Relative differences go up to 3.9e-10, in the LIQUID derivatives. In GRAPHITE, the original and scrambled energies differ in a single constant: `-4997.7076666699995` vs. `-4997.707666670001`.

That constant appears in neither database. It is created while the Model substitutes FUNCTIONs into each other (e.g. `GA9C = +GHSERCC#`), which flattens nested sums. SymEngine adds their constants together as it flattens, in the order the terms are stored. By the time `build_functions` sees the expression there is one rounded constant, which reordering cannot undo. Ordering the FUNCTIONs and parameters before substitution, as the Model-side fix does, makes the addition order fixed.

## Bocklund's tests

The tests his branch adds to `test_model.py`, run on each checkout (copy his `test_model.py` into the other two):

| | develop | #739 | Bocklund |
|---|---|---|---|
| `test_parameter_order_does_not_change_phase_records` (3) | fail | pass | pass |
| `test_tdb_roundtrip_does_not_change_phase_records` (3) | fail | pass | pass |
| `test_phase_records_do_not_depend_on_hash_seed` | fail | pass | pass |

Both issue641 tests pass on every branch, in their original form (MU against Thermo-Calc, rtol=1e-5) and in his form (GM = Σ X·MU). Their worst MU error, as a fraction of rtol, is the same for develop and Bocklund on both tests, but higher with #739 on the second:

| | develop | #739 | Bocklund |
|---|---|---|---|
| `test_eq_issue641_trace_composition_conditions_converge` | 0.11 | 0.11 | 0.11 |
| `test_eq_issue641_unstable_compset_jitter_does_not_block_convergence` | 0.20 | 0.58 | 0.20 |

## Cost on a large database

`timing.py` uses the MPEA database: 74 phases for Al-C-Co-Cr-Fe-Mn-Ni-V. It reports CPU time, as the median of 4 runs interleaved across branches (3 for the equilibrium). `compile` evaluates every PhaseRecord function once, since compilation is lazy. `equilibrium_total` is one `Workspace` equilibrium from a cold start, at 1200 K, x(Co, Cr, Fe, Mn) = 0.2, x(Al) = 0.05, x(V) = 0.01, x(C) = 0.005.

| | develop | #739 | Bocklund |
|---|---|---|---|
| load Database | 1.162 s | +0.3% | +0.3% |
| build Models | 1.086 s (1.080–1.087) | +0.2% (1.078–1.102) | +5.3% (1.140–1.147) |
| compile | 15.08 s | −0.3% | −0.1% |
| one equilibrium, cold | 15.92 s (15.90–16.05) | +0.2% (15.94–16.05) | +0.9% (16.05–16.17) |

All three find FCC_A1 + M23C6_D84, with GM equal to the fourth decimal place: −79456.818224 (develop), −79456.818186 (#739), −79456.818126 (Bocklund). Each branch gives the same value on every run.
