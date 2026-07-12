# Accelerated pycalphad backends + ESPEI integration — collaborator brief

*Status: working branches on my forks, both CI-green on all platforms; not yet
submitted as PRs. Everything below is measured, not projected.*

## One paragraph

I have a working pair of branches that add two optional, runtime-selectable
compute backends to pycalphad (`set_backend('c++')` / `set_backend('gpu')`)
and a batched ZPF likelihood for ESPEI built on top of them. The backends
generate C/CUDA code from the symbolic `Model` energies at runtime and
disk-cache the compiled kernels; anything they can't handle exactly falls
back silently to the reference solver, so enabling them globally never
changes results for unsupported problems. Measured: ~10x for pycalphad
equilibrium batches on one CPU core, ~13x per ESPEI ZPF likelihood, ~40x on
a real emcee run, and ~10 ms per chain per step (flat to 1000+ chains) for
GPU ensembles. The full pycalphad test suite passes under both backends on
current develop (319/318/318 across default/c++/gpu), and the full ESPEI
suite passes both with and without the accelerated pycalphad installed.

## Design principles (the two-sentence version each)

- **Reference solver untouched.** The reference tree is byte-for-byte
  upstream develop; the only reference-file changes are a ~40-line
  capability gate + dispatch in `core/equilibrium.py` and one evaluator
  hook in `core/calculate.py`. The reference remains the ground truth the
  backends are validated against.
- **Capability gate with silent fallback.** A single readable function
  decides whether a problem shape is supported; everything else routes to
  the reference path *including its validation errors*. New upstream model
  features either work generically (never-disorder did, at 1e-15) or fall
  back until implemented.
- **Bit-parity discipline.** Components are validated at the tightest level
  that's meaningful: the compiled starting-point hull is bit-identical to
  the Cython `hyperplane()` (~2,200 cases including fixed-chempot sets);
  end-to-end batches are bit-identical or eps-class with identical phase
  sets. Same-seed emcee runs give consistent posteriors and acceptance.
- **ESPEI owns its domain logic.** pycalphad exposes generic hooks (a
  point-list batch solver and walker-ensemble machinery); the ZPF-specific
  code lives in an ESPEI module that imports cleanly against stock
  pycalphad and activates only when the accelerated build is present —
  ESPEI's CI (which installs stock pycalphad) passes on the branch today.

## What is accelerated — pycalphad

| supported (accelerated) | falls back to reference |
|---|---|
| fully-determined N=1 / P / T / X condition grids | phase-local conditions |
| `calculate()` property outputs (GM, HM, SM, CPM, `_MIX`/`_FORM`; ~2x the reference callables) | non-symbolic outputs |
| `equilibrium()` output properties at converged states (NP-weighted, reference semantics) | phase-qualified / dotted-derivative outputs |
| grid-method phase diagrams: `binplot/ternplot(method='grid')` (Al-Ni 27.1 s → 2.9 s; boundaries = tie-line endpoints, 1e-15 vs reference) | ZPF-follower mapping stays reference |
| pip-only GPU installs (NVRTC/hipRTC fallback when nvcc/hipcc is absent; suite green under both compilers) | hipRTC untested in CI |
| MU (chemical potential) conditions, scalar AND array axes | — |
| W() mass-fraction and linear-combination conditions (incl. ratio form) — eps-class GM (<=1e-6) with matching phase sets; the backend converges on every W-grid condition the reference fails | — |
| plain `Model` incl. magnetic, order/disorder, never-disorder | custom Model subclasses (e.g. MQMQA) |
| runtime parameter overrides (scalar) | under/overdetermined inputs (reference errors preserved) |
| grids from ~1 to 1,000,000+ conditions (chunked) | dilute compositions X < 1e-9 (reference clamping) |

Measured speedups vs upstream 0.11.1 (laptop RTX 4070, one CPU core):
~7x at 250 conditions, ~10x at 10k–1M conditions (c++ and gpu comparable at
scale on this hardware; GPU FP64 is 1:64 on GeForce — datacenter cards are
the headroom).

## What is accelerated — ESPEI (by data type, honestly)

| residual | status today | path to acceleration |
|---|---|---|
| **ZPF** | **batched: ~13x/likelihood (c++), ~40x real emcee, ~165–200x GPU ensembles at 64–1024 chains** | done for binaries; ternary+ vertices fall back per-vertex (extension is mechanical) |
| Activity | **batched** — per-dataset subsystem machinery built once, all sample equilibria in one launch; residuals identical to stock (Cu-Mg fixture, with and without fit params); speedup scales with points/dataset (~1.2x at 10 points, approaching the point-count ratio for larger sets) | done; boundary-composition points and multi-pressure datasets fall back per-point |
| Non-eq. thermochemical (HM_MIX etc.) | **batched: ~100x/likelihood** — per-sample dof rows precomputed once, property evaluated in one compiled call (incl. reference-state shifts and excluded contributions); likelihoods identical to stock on all Cu-Mg fixtures | done; custom Model subclasses fall back per-group |
| Eq. thermochemical | **batched** — one point-solver launch per dataset + property at converged states; residuals within 1e-10 of stock, 6.6x at 7 points (scales with points) | done; all four residual types now batch |

Opt-in is one call — `enable_accelerated_backend('c++')` — which swaps the
ZPF residual in the pluggable registry; input YAML, `run_espei`, dask, all
unchanged. `sample_df='reference'` mode reproduces the stock log-likelihood
**exactly** on Cu-Mg (all 7 ESPEI-datasets ZPF sets).

## Validation evidence (highlights)

- pycalphad suite on current develop: default 319 / c++ 318 / gpu 318
  passed (the delta is one deliberately backend-aware skip). CI green on
  ubuntu/macos/windows × 3.11/3.12/3.13.
- ESPEI suite: 289 passed against the accelerated pycalphad; ESPEI's own
  CI (stock pycalphad release + develop) green on the branch.
- AlCuFe (37 phases, 245 conditions, fallback-guarded census): 240/240
  mutually converged with max |dGM| 8.8e-6 J/mol and matching phase
  sets; the backend also converges on all 5 conditions the reference
  fails, with zero backend-only failures. AuBi 1M conditions: GM
  bit-identical between compiled-hull and reference-hull paths.
- emcee acceptance: 16 walkers × 30 steps, stock vs batched — acceptance
  0.61 vs 0.63, consistent posteriors, 40.4x wall.

## Known limitations / open questions I'd ask a maintainer

1. **Which data types matter most?** ZPF dominated the fits I profiled, but
   if activity or thermochemical data dominate real-world ESPEI use, those
   are the next batching targets — is that where the demand is?
2. **MQMQA roadmap.** The backends fall back for MQMQA models (correct
   results, stock speed). Is MQMQA-heavy fitting a near-term priority that
   would justify porting its model/constraint structure?
3. **API stability assumptions.** The ESPEI module consumes `zpf_data` /
   `PhaseRegion` / `RegionVertex` and the residual registry; pycalphad-side
   code consumes the public Workspace attributes. Any planned refactors
   (emcee 3 migration, Workspace evolution, zpf_data schema) I should track?
4. **Ensemble strategy.** The GPU path makes 1000-chain ensembles cost
   ~12 s/step — does "many more chains, fewer steps" fit how you think
   about ESPEI's MCMC, and is a `vectorize`-style hook in ESPEI's mcmc
   driver something you'd take upstream?
5. **Packaging.** `pycalphad[cpp]` / `pycalphad[gpu]` extras exist; the gpu
   extra needs nvcc beyond the CuPy wheel (documented). Preferences on
   conda-forge story / CI coverage for compiled backends?

## Pointers

- pycalphad branch: `Zhyrek/pycgpu` @ `gpu-port-sync` (see `BACKENDS.md`,
  `PR_DRAFT.md` at the root)
- ESPEI branch: `Zhyrek/espei` @ `accelerated-backends` (see `PR_DRAFT.md`;
  the module is `espei/error_functions/batched_zpf_error.py`)
