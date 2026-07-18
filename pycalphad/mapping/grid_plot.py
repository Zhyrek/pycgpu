"""
Grid-based phase diagram plotting: dense equilibrium grids instead of
ZPF-line following.

A batch equilibrium over the full condition grid (fast under the accelerated
backends; see ``pycalphad.set_backend``) yields, at every grid point, the
stable phase assemblage AND the equilibrium composition of each stable phase.
Points in multi-phase fields therefore contribute EXACT phase-boundary
compositions (the tie-line endpoints), not grid-quantized cell edges — a
binary boundary from a two-phase point is at solver accuracy regardless of
grid resolution. Grid resolution only controls how densely the boundaries
are sampled.

This is the classic "step/map everywhere" approach; it trades the ZPF
follower's adaptive node solving for brute-force parallelism, which the
accelerated backends make cheap.
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from pycalphad import variables as v
from pycalphad.plot.utils import phase_legend


_COMP_EDGE = 1e-3  # composition grids stay off the exact 0/1 endpoints


def _inset_composition_ranges(conditions):
    """Nudge composition condition ranges inside (0, 1).

    Exact 0/1 endpoints imply dilute counter-components, which route to the
    (much slower) reference solver via the capability gate and add nothing to
    a phase diagram; boundary positions come from tie-line endpoints, so no
    information is lost.
    """
    from pycalphad.core.utils import unpack_condition
    fixed = {}
    for cond, value in conditions.items():
        if isinstance(cond, v.MoleFraction) and getattr(cond, 'phase_name', None) is None:
            value = np.atleast_1d(np.asarray(unpack_condition(value), dtype=np.float64))
            if value.size >= 2:
                lo, hi = value.min(), value.max()
                lo2, hi2 = max(lo, _COMP_EDGE), min(hi, 1.0 - _COMP_EDGE)
                value = np.linspace(lo2, hi2, value.size)
            fixed[cond] = value
        else:
            fixed[cond] = value
    return fixed


def _grid_equilibrium(database, components, phases, conditions, **eq_kwargs):
    from pycalphad import equilibrium
    return equilibrium(database, components, phases,
                       _inset_composition_ranges(conditions), **eq_kwargs)


def _stable_sets(eq):
    """(n_conditions, ...) arrays of phase names / NP / X flattened over conditions."""
    n_vert = eq.Phase.shape[-1]
    names = eq.Phase.values.reshape(-1, n_vert)
    np_arr = eq.NP.values.reshape(-1, n_vert)
    x_arr = eq.X.values.reshape(-1, n_vert, eq.X.shape[-1])
    return names, np_arr, x_arr


def binplot_grid(database, components, phases, conditions, x=None, y=None,
                 ax=None, eq_kwargs=None, plot_kwargs=None):
    """Binary T-x phase diagram from a dense equilibrium grid.

    Every grid point in a two-phase field contributes its two tie-line
    endpoint compositions at that temperature (solver-accurate boundary
    points); three-phase points mark invariant reactions.
    """
    eq_kwargs = eq_kwargs or {}
    # ZPF-binplot compatibility: the axes may arrive via plot_kwargs['ax'];
    # the remaining plot_kwargs are forwarded to the scatter calls.
    plot_kwargs = dict(plot_kwargs or {})
    ax = plot_kwargs.pop('ax', ax)
    comp_conds = [c for c in conditions
                  if isinstance(c, v.MoleFraction) and getattr(c, 'phase_name', None) is None]
    if len(comp_conds) != 1:
        raise ValueError("binplot(method='grid') needs exactly one composition condition")
    x_cond = comp_conds[0] if x is None else x
    comp_name = str(x_cond)[2:]

    eq = _grid_equilibrium(database, components, phases, conditions, **eq_kwargs)
    comp_axis = list(eq.coords['component'].values).index(comp_name)
    t_values = np.asarray(eq.coords['T'])

    names, np_arr, x_arr = _stable_sets(eq)
    # Broadcast T per flattened condition (T varies slower than X in the grid)
    n_cond = names.shape[0]
    t_per_cond = np.broadcast_to(
        t_values[:, None], (t_values.size, n_cond // t_values.size)).reshape(-1)

    per_phase = defaultdict(list)  # phase -> [(x_boundary, T), ...]
    invariant_ts = set()
    for i in range(n_cond):
        stable = [(p, k) for k, p in enumerate(names[i])
                  if p and not np.isnan(np_arr[i, k]) and np_arr[i, k] > 1e-6]
        if len(stable) == 2:
            for p, k in stable:
                per_phase[p].append((x_arr[i, k, comp_axis], t_per_cond[i]))
        elif len(stable) >= 3:
            invariant_ts.add(round(float(t_per_cond[i]), 9))

    if ax is None:
        _, ax = plt.subplots()
    handles, colors = phase_legend(sorted(per_phase))
    for phase_name, pts in sorted(per_phase.items()):
        arr = np.asarray(pts)
        ax.scatter(arr[:, 0], arr[:, 1], s=3, color=colors[phase_name],
                   **plot_kwargs)
    for t_inv in sorted(invariant_ts):
        ax.axhline(t_inv, color=(1, 0, 0, 0.4), lw=0.8)
    ax.set_xlabel(f"X({comp_name})")
    ax.set_ylabel("Temperature (K)")
    ax.set_xlim(0, 1)
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    return ax


def ternplot_grid(database, components, phases, conditions, x=None, y=None,
                  ax=None, eq_kwargs=None, plot_kwargs=None, tielines=True):
    """Isothermal ternary section from a dense equilibrium grid.

    Two-phase points contribute tie-lines (exact endpoint compositions);
    three-phase points contribute tie-triangles.
    """
    eq_kwargs = eq_kwargs or {}
    # ZPF-ternplot compatibility: the axes may arrive via plot_kwargs['ax'].
    plot_kwargs = dict(plot_kwargs or {})
    ax = plot_kwargs.pop('ax', ax)
    comp_conds = sorted((c for c in conditions
                         if isinstance(c, v.MoleFraction) and getattr(c, 'phase_name', None) is None),
                        key=str)
    if len(comp_conds) != 2:
        raise ValueError("ternplot(method='grid') needs exactly two composition conditions")
    x_cond = comp_conds[0] if x is None else x
    y_cond = comp_conds[1] if y is None else y
    x_name, y_name = str(x_cond)[2:], str(y_cond)[2:]

    eq = _grid_equilibrium(database, components, phases, conditions, **eq_kwargs)
    comp_names = list(eq.coords['component'].values)
    xi, yi = comp_names.index(x_name), comp_names.index(y_name)

    names, np_arr, x_arr = _stable_sets(eq)
    if ax is None:
        _, ax = plt.subplots()
    all_names = sorted({p for row in names for p in row if p})
    handles, colors = phase_legend(all_names)

    for i in range(names.shape[0]):
        stable = [(p, k) for k, p in enumerate(names[i])
                  if p and not np.isnan(np_arr[i, k]) and np_arr[i, k] > 1e-6]
        if len(stable) == 2 and tielines:
            (pa, ka), (pb, kb) = stable
            xa, ya = x_arr[i, ka, xi], x_arr[i, ka, yi]
            xb, yb = x_arr[i, kb, xi], x_arr[i, kb, yi]
            ax.plot([xa, xb], [ya, yb], color=(0, 1, 0, 0.25), lw=0.5, zorder=1)
            ax.scatter([xa, xb], [ya, yb], s=3,
                       color=[colors[pa], colors[pb]], zorder=2, **plot_kwargs)
        elif len(stable) == 3:
            pts = np.array([[x_arr[i, k, xi], x_arr[i, k, yi]] for _, k in stable])
            ax.fill(pts[:, 0], pts[:, 1], color=(1, 0, 0, 0.15), zorder=0)
    ax.set_xlabel(f"X({x_name})")
    ax.set_ylabel(f"X({y_name})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    return ax
