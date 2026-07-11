# Cross-process nondeterminism in pycalphad results via `PYTHONHASHSEED`

## Summary

pycalphad produces **bitwise-different numerical results in different Python
processes for identical inputs**. The differences originate in the compiled
energy callables (the `calculate()` → codegen → symengine `lambdify` path),
are on the order of 1 ulp (~1e-12 relative), and are controlled by CPython's
hash randomization: setting `PYTHONHASHSEED` to a fixed value makes results
bit-reproducible across processes, and repeated calls **within one process are
always bit-identical**.

One ulp sounds harmless, but equilibrium solving amplifies it. For conditions
near degenerate phase boundaries or fragile convergence basins, the jitter
changes which local minimum the solver lands in — or whether it converges at
all. We observe a single stock `equilibrium()` call that returns **three
different outcomes** across identical invocations: two different converged
Gibbs energies **4232 J/mol apart**, and outright convergence failure (NaN).

This affects reproducibility of published results, makes CI equilibrium tests
intrinsically flaky at eps-degenerate conditions, and silently confounds any
A/B comparison of code changes performed across separate processes.

## Minimal reproducers

Both use the Al-Cu-Fe database (21 phases; smaller databases such as
`alzn_mey.tdb` did *not* reproduce, see "Localization" below).

### 1. Bitwise jitter in `calculate()`

```python
# repro_calculate.py — run twice, compare the .npy files
import sys
import numpy as np
from pycalphad import Database, calculate

dbf = Database('Al-Cu-Fe.tdb')
res = calculate(dbf, ['AL', 'CU', 'FE', 'VA'], sorted(dbf.phases.keys()),
                T=700, P=101325, N=1)
np.save(sys.argv[1], np.ascontiguousarray(res.GM.values))
```

```text
$ python repro_calculate.py a.npy && python repro_calculate.py b.npy
$ python -c "import numpy as np; a, b = np.load('a.npy'), np.load('b.npy'); \
             d = np.abs(a - b); print((d > 0).sum(), 'points differ, max', d.max())"
101 points differ, max 7.28e-12
```

The sampled compositions (`res.X.values`) are bit-identical between the two
runs — only the energies differ. With `PYTHONHASHSEED=0` exported for both
runs, the energies are bit-identical too.

### 2. Three-outcome flaky `equilibrium()`

```python
# repro_equilibrium.py
import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('Al-Cu-Fe.tdb')
conds = {v.X('AL'): 0.5999999999999999, v.X('CU'): 0.35,
         v.T: 700, v.P: 101325, v.N: 1}
r = equilibrium(dbf, ['AL', 'CU', 'FE', 'VA'], sorted(dbf.phases.keys()), conds)
print(repr(float(r.GM.values.flat[0])))
```

Ten identical invocations:

```text
nan  nan  nan  nan  nan  nan  nan  nan  -45333.20492145921  -41101.25820234221
```

Three distinct outcomes for the same physical problem: convergence failure
(8/10), an ALCU_ETA + TS01T2 two-phase minimum (1/10), and an
ALCU_THETA + L12 local minimum 4232 J/mol higher (1/10). (The deeper
ALCU_ETA + TS01T2 basin has been independently verified at this condition:
evaluating the model energies directly at converged site fractions from that
basin reproduces its GM with exact mass balance, ~4230 J/mol below the
ALCU_THETA + L12 answer — so the flakiness selects between a correct
equilibrium, an incorrect local minimum, and failure.)

With `PYTHONHASHSEED` fixed, all ten invocations return the same outcome
(for seed 0 that outcome happens to be the convergence failure — pinning the
seed buys *reproducibility*, not *correctness*; the knife-edge fragility
itself is a separate, now-deterministic solver issue).

Note this composition sits on a known convergence knife edge — that is the
point: the ulp-level jitter only becomes *visible* at such conditions, but it
is present everywhere (reproducer 1), and dense property mappings will
routinely contain a few such conditions. On a 245-condition Al-Cu-Fe grid we
see the set of converged conditions itself change (239 vs 240) between
identical runs of the same script.

## Localization

Evidence chain, each item compared bitwise across two fresh processes:

| Stage | Cross-process deterministic? |
|---|---|
| `Model(dbf, comps, phase).GM` printed expression string | **Yes** (string-identical, all phases tested) |
| `calculate()` sampled site fractions / compositions (`res.X`) | **Yes** |
| `calculate()` energies (`res.GM`) | **No** (1 ulp, 101/~10k points) |
| `starting_point()` chemical potentials | **No** (1 ulp, inherited) |
| `equilibrium()` GM / convergence | **No** (up to 4232 J/mol at knife edges) |
| Any of the above with `PYTHONHASHSEED` fixed | **Yes** (bit-identical) |
| Any of the above repeated within one process | **Yes** (bit-identical) |

Ruled out:

- **BLAS/LAPACK threading:** running with
  `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1` still jitters.
  Only `PYTHONHASHSEED` controls it.
- **Sampling randomness:** the sampled points are bit-identical; `calculate`'s
  point generation is already seeded/deterministic.
- **Symbol/argument ordering bugs:** a wrong argument order would produce
  large errors, not 1 ulp. The observed differences are pure
  floating-point-associativity-sized.

The divergence therefore enters between the symbolic model (whose *printed
form* is stable) and the numeric output of the compiled callables — i.e. in
`pycalphad.codegen.sympydiff_utils.build_functions` → symengine `lambdify`
(LLVM backend, CSE enabled). The most plausible mechanism: hash-randomized
`set`/`dict` iteration order somewhere in expression assembly or
lowering changes the *internal tree shape* of sum/product nodes (associativity
grouping) or the CSE ordering, producing a compiled function that evaluates
the same mathematical expression with a different floating-point operation
order. A printed expression string can be identical while the underlying DAG
and the emitted instruction order differ. The dependence on database size
(Al-Cu-Fe with large multi-sublattice phases reproduces; the 3-phase Al-Zn
database does not) is consistent with this: larger expressions have
exponentially more equivalent groupings.

## Why this matters

1. **Reproducibility.** Two researchers running the same script on the same
   machine can publish different phase fractions at degenerate conditions, and
   the same researcher cannot reproduce their own NaN/non-NaN pattern without
   pinning an undocumented environment variable.
2. **Test flakiness.** Equilibrium assertions at (unknowingly) knife-edge
   conditions will flake in CI at some low rate, and the failures are
   unreproducible under a debugger by construction.
3. **Development methodology.** Any bit-exactness A/B comparison of a code
   change made across separate processes is confounded. We initially
   attributed a 3.5e-6 GM shift to a refactor that was in fact exactly
   equivalence-preserving — the shift was this jitter. (Pinning
   `PYTHONHASHSEED` and re-running showed old and new code bit-identical.)
4. **Solver robustness masking.** Flaky convergence at knife-edge conditions
   looks like random solver failure, hiding the deterministic underlying cause
   and inflating perceived solver fragility.

## Workarounds available today

- `export PYTHONHASHSEED=0` (any fixed value) before launching Python:
  everything becomes bit-reproducible across processes, at zero performance
  cost. (Caveat: this must be set *before* interpreter start; setting it in
  `os.environ` at runtime has no effect.)
- Keep comparative computations within a single process.

## Suggested fix directions

The clean fix is to make the expression-assembly and lowering pipeline
iteration-order independent:

1. Audit `Model` contribution assembly, parameter/species handling, and
   `build_functions` for iteration over `set`/unordered `dict` where elements
   are floats/expressions that get summed or CSE'd; replace with
   deterministically sorted iteration (e.g. by canonical sort key of the
   symbol/species name).
2. Alternatively (or additionally), canonicalize the expression immediately
   before `lambdify` (a deterministic total ordering of `Add`/`Mul` args)
   so upstream ordering cannot influence the emitted operation order.
3. Add a regression test: build the same `PhaseRecord` in two subprocesses
   with different `PYTHONHASHSEED` values and assert bitwise-equal outputs on
   a fixed dof vector. This is cheap and pins the property permanently.

A one-line documentation note recommending `PYTHONHASHSEED` for strict
reproducibility would be a useful stopgap until the ordering audit lands.

## Environment

- pycalphad 0.11.1.dev (develop branch), Python 3.12.2, numpy 1.26.4,
  symengine 0.13.0 (LLVM backend, `cse=True`, `opt_level=0`)
- Linux x86-64 (WSL2); also expected on any platform since the mechanism is
  CPython hash randomization (enabled by default since Python 3.3)
- Database: commercial-style Al-Cu-Fe TDB (21 phases, multi-sublattice
  intermetallics). Not reproducible with the small `alzn_mey.tdb` test
  database — reviewers should use a database with large phase expressions.
