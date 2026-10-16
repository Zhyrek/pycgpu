# Dense equilibrium grids

`pycalphad.densegrid` computes equilibrium grids too large for one
`equilibrium()` call. The grid is split into tiles, and each tile's results
are written to a Zarr store as soon as it is solved. A run can be stopped
and resumed at any point, and many worker processes, on one machine or on
every machine of a cluster that shares a filesystem, can work on it
together.

```bash
pip install "pycalphad[densegrid]"        # adds zarr
python -m pycalphad.densegrid create spec.json /shared/runs/myrun
python -m pycalphad.densegrid run /shared/runs/myrun --workers 64
python -m pycalphad.densegrid status /shared/runs/myrun
```

```python
from pycalphad.densegrid import open_dataset
ds = open_dataset('/shared/runs/myrun')   # lazy xarray Dataset
```

## The grid spec

A grid is the Cartesian product of its axes at fixed values of the other
conditions:

```json
{"database": "COST507.tdb",
 "components": ["AL", "MG", "SI", "CU", "ZN", "FE", "MN", "NI", "CR", "TI", "VA"],
 "axes": {"X(CU)": {"start": 0.005, "stop": 0.06, "num": 12},
          "X(MG)": {"start": 0.005, "stop": 0.06, "num": 12},
          "X(SI)": {"start": 0.005, "stop": 0.06, "num": 12},
          "X(ZN)": {"start": 0.005, "stop": 0.06, "num": 12}},
 "fixed": {"T": 750, "X(FE)": 0.005, "X(MN)": 0.005, "X(NI)": 0.005, "X(CR)": 0.005, "X(TI)": 0.001},
 "outputs": ["GM", "MU", "NP", "Phase", "X"],
 "backend": "cg"}
```

- **Axis values:** a list, `{"start", "stop", "num"}` (both ends included),
  or `{"start", "stop", "step"}`.
- **Conditions:** any condition string pycalphad understands, such as `T`,
  `P`, `X(CU)`, `W(CU)` or `MU(CU)`. N defaults to 1 and P to 101325 Pa.
- **`outputs`:** any of `GM`, `MU`, `NP`, `Phase`, `X` and `Y`. `Y`, the
  site fractions, is the largest by far.
- **`backend`:** any pycalphad backend. `cg` is the default and the fastest
  at many elements.
- **`threads`:** solver threads per worker process. The default is 1; see
  Reproducibility below.
- **`tile`:** points per tile along each axis. When omitted, a tile gets one
  value of each state variable and about 8,192 points over the other axes.
  Choose tiles that take at least tens of seconds. The fixed cost is about
  0.1 s per tile, and the number of files in the store grows with the
  number of tiles.

## Running on a cluster

Start `run` once per machine. All machines must see the run directory.
Workers coordinate only through files in that directory:
- a worker claims a tile by creating a file exclusively;
- while it works, a heartbeat keeps the claim fresh;
- a claim that hasn't been refreshed for `--lease` seconds (default 300) is
  taken over, so the tiles of a crashed worker or node are redone by the
  others.

```bash
#SBATCH --nodes=16 --ntasks-per-node=1 --cpus-per-task=64
srun python -m pycalphad.densegrid run /shared/runs/myrun --workers 64
```

Each worker first solves its own share of the tiles. Under SLURM, the share
comes from `SLURM_NODEID` and the node count; elsewhere, pass `--node` and
`--nodes`. When its share is done, a worker takes whatever tiles are left.

After an interruption, run the same command again; finished tiles are
skipped. A failed tile records its traceback in `tiles/failed/` and is not
retried unless you pass `--retry-failed`.

## The run directory

| path | holds |
|---|---|
| `spec.json`, `meta.json` | the spec; element and phase order, sizes, pycalphad version and git commit |
| `database.tdb` | the copy of the database every worker reads |
| `grid.zarr/` | the results, one chunk per tile |
| `tiles/{claims,done,failed}/` | tile bookkeeping (each `done` file holds that tile's timing and counts) |
| `logs/` | one log per worker process |

**Arrays:**
- `GM` over the axes;
- `MU` with `component`;
- `NP` and `Phase` with `vertex`;
- `X` with `vertex` and `component`;
- `Y` with `vertex` and `internal_dof`;
- `status` over the axes.

**Phase** holds indices into `ds.attrs['phases']`, with -1 meaning no
phase.

**status** values:

| value | meaning |
|---|---|
| -1 | not computed yet |
| 0 | solved |
| 1 | not solved (NaN) |
| 2 | not a valid composition (mole fractions summing to more than one) |

## Reproducibility

Each tile is solved from scratch, with fresh seeding and no state carried
over from the previous tile. So with one solver thread per worker, a tile's
results depend only on the spec. They don't depend on which worker,
machine, worker count or order solved it, or on whether the run was
interrupted.

The launcher sets `PYTHONHASHSEED=0` for its workers, unless you set it
yourself, because pycalphad sums symbolic terms in hash order. With that,
runs are bit-for-bit identical. With `threads` above 1, cg splits a tile
between threads, and results then agree within the solver tolerance
(1e-6 RT in energy).

## Sizing

All figures are for COST507 with 10 elements at 750 K, on a 4-D composition
grid in which most points have 7 to 10 phases.

| | |
|---|---|
| cg, one thread | about 1.8–2.5 CPU-ms per point |
| cpp backend, all 10 cores | 11 ms per point |
| reference solver | about 40 ms per point |
| worker start-up | about 10 s (models, phase records, first compile) |
| worker memory | about 1.3 GB |
| store size, default outputs | about 270 bytes per point after compression |
| storage share | `X` is most of it; adding `Y` multiplies the size several times |

So 10⁹ points cost about 500–700 CPU-core hours with cg, and about 270 GB
of storage with the default outputs.

## Solvers for tiles

`pycalphad.densegrid.TileSolver` turns a tile's conditions into its result
arrays through `equilibrium()`. A different engine for whole tiles, such as
a GPU solver, would plug in at this interface:
`solve(slices) -> (arrays, info)`.
