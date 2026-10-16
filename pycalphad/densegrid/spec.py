"""What a dense grid computes, as plain JSON shared by every worker of a run.

A grid is the Cartesian product of its axes (each a condition with a list of values), at fixed values of the other
conditions. Example::

    {"database": "COST507.tdb",
     "components": ["AL", "CU", "MG", "SI", "ZN", "VA"],
     "axes": {"T": [700, 750, 800],
              "X(CU)": {"start": 0.005, "stop": 0.06, "num": 12},
              "X(MG)": {"start": 0.005, "stop": 0.06, "num": 12}},
     "fixed": {"P": 101325, "X(SI)": 0.01, "X(ZN)": 0.02},
     "outputs": ["GM", "MU", "NP", "Phase", "X"]}

Axis values are a list, or {"start", "stop", "num"} (endpoints included), or {"start", "stop", "step"}. N defaults to
1 and P to 101325 Pa. "tile" (points per tile along each axis) is chosen when omitted: one value of each state
variable axis, and about TARGET_TILE_POINTS points spread over the other axes.
"""
import json
import math
from dataclasses import dataclass, field, asdict

import numpy as np

from pycalphad import variables as v
from pycalphad.property_framework import as_property

ALL_OUTPUTS = ('GM', 'MU', 'NP', 'Phase', 'X', 'Y')
DEFAULT_OUTPUTS = ('GM', 'MU', 'NP', 'Phase', 'X')
# Points per tile when the spec does not say. At ~2 ms per point (10 elements, one thread) a tile takes ~15 s, and
# the fixed cost per tile (~0.1 s with prebuilt models, including seeding) stays below 1 %.
TARGET_TILE_POINTS = 8192


def axis_values(spec_value):
    """The values of one axis from its spec entry."""
    if isinstance(spec_value, dict):
        start, stop = float(spec_value['start']), float(spec_value['stop'])
        if 'num' in spec_value:
            return np.linspace(start, stop, int(spec_value['num']))
        step = float(spec_value['step'])
        n = int(math.floor((stop - start) / step + 1e-9)) + 1
        return start + step * np.arange(n)
    return np.asarray(spec_value, dtype=float).ravel()


def dim_name(condition):
    """The dimension name pycalphad uses for a condition string ('X(CU)' -> 'X_CU')."""
    return str(as_property(condition))


def is_state_variable(condition):
    return as_property(condition) in (v.T, v.P, v.N)


@dataclass
class GridSpec:
    database: str
    components: list
    axes: dict
    fixed: dict = field(default_factory=dict)
    phases: list = None
    tile: dict = None
    outputs: list = DEFAULT_OUTPUTS
    backend: str = 'cg'
    threads: int = 1   # solver threads per worker process (1: results do not depend on the machine's thread count)

    def __post_init__(self):
        self.components = [str(c).upper() for c in self.components]
        self.outputs = list(self.outputs)
        bad = [o for o in self.outputs if o not in ALL_OUTPUTS]
        if bad:
            raise ValueError(f'unknown outputs {bad}; available: {ALL_OUTPUTS}')
        if not self.axes:
            raise ValueError('a grid needs at least one axis')
        for cond in list(self.axes) + list(self.fixed):
            as_property(cond)   # raises for conditions pycalphad does not know
        both = {dim_name(a) for a in self.axes} & {dim_name(f) for f in self.fixed}
        if both:
            raise ValueError(f'conditions both on an axis and fixed: {sorted(both)}')
        fixed_dims = {dim_name(f) for f in self.fixed} | {dim_name(a) for a in self.axes}
        self.fixed = dict(self.fixed)
        if 'N' not in fixed_dims:
            self.fixed['N'] = 1.0
        if 'P' not in fixed_dims:
            self.fixed['P'] = 101325.0
        for a in self.axes:
            if len(axis_values(self.axes[a])) == 0:
                raise ValueError(f'axis {a} has no values')
        if self.tile is None:
            self.tile = default_tile(self.axes)
        else:
            unknown = set(self.tile) - set(self.axes)
            if unknown:
                raise ValueError(f'tile sizes for conditions that are not axes: {sorted(unknown)}')
            self.tile = {a: max(1, min(int(self.tile.get(a, 1)), len(axis_values(self.axes[a])))) for a in self.axes}

    @property
    def axis_names(self):
        return list(self.axes)

    @property
    def dims(self):
        return [dim_name(a) for a in self.axes]

    @property
    def shape(self):
        return tuple(len(axis_values(self.axes[a])) for a in self.axes)

    @property
    def tile_shape(self):
        return tuple(self.tile[a] for a in self.axes)

    def values(self, axis):
        return axis_values(self.axes[axis])

    def to_json(self):
        d = asdict(self)
        d['axes'] = {a: (val if isinstance(val, dict) else [float(x) for x in np.ravel(val)])
                     for a, val in self.axes.items()}
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_json(f.read())


def default_tile(axes):
    """Points per tile along each axis: one value of each state-variable axis; the others grown in turn (the axis
    with the fewest points per tile first) while the tile stays within TARGET_TILE_POINTS."""
    n = {a: len(axis_values(val)) for a, val in axes.items()}
    tile = {a: 1 for a in axes}
    free = [a for a in axes if not is_state_variable(a)] or list(axes)
    while True:
        grow = [a for a in free if tile[a] < n[a]]
        if not grow:
            break
        a = min(grow, key=lambda a: (tile[a], -n[a]))
        size = math.prod(tile.values()) // tile[a] * (tile[a] + 1)
        if size > TARGET_TILE_POINTS:
            break
        tile[a] += 1
    return tile
