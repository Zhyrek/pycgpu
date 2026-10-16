"""Dense equilibrium grids too large for one equilibrium() call: tiled, streamed to disk, restartable, and worked on
by many processes (on one or several machines sharing a filesystem).

    from pycalphad.densegrid import GridSpec, create, run, open_dataset
    spec = GridSpec(database='COST507.tdb', components=[...], axes={...}, fixed={...})
    create(spec, 'myrun')          # lays out the run directory
    run('myrun', workers=8)        # solves the pending tiles; rerun after an interruption to continue
    ds = open_dataset('myrun')     # lazy xarray Dataset

or from the shell: python -m pycalphad.densegrid create|run|status ...
"""
from .spec import GridSpec
from .store import Run, open_dataset
from .worker import run, TileSolver


def create(spec, path, overwrite=False):
    """Lay out a run directory for `spec` (a GridSpec or the path of its JSON). Returns the Run."""
    return Run.create(spec, path, overwrite=overwrite)


def status(path):
    return Run(path).status()


__all__ = ['GridSpec', 'Run', 'TileSolver', 'create', 'run', 'status', 'open_dataset']
