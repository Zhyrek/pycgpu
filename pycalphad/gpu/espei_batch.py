"""Batched ZPF driving-force evaluation for ESPEI on the accelerated backends.

Replaces ESPEI's per-vertex serial Workspace solves (``estimate_hyperplane`` +
``driving_force_to_hyperplane`` in ``espei.error_functions.zpf_error``) with a
small number of batched ``equilibrium()`` calls on the c++/CUDA backends:

- all hyperplane vertices with full composition conditions become ONE
  all-phase batch per (X-component) group;
- all "isolated phase" driving-force vertices become ONE single-phase batch
  per (phase, X-component) group;
- vertices the batch path cannot serve exactly (missing composition
  conditions -> sampling estimate, disordered configurations, near-pure-edge
  compositions) fall back to ESPEI's own serial code paths.

The batches are cartesian (unique T x unique X per group) and results are
extracted per vertex, so duplicate vertex conditions are solved once. The
driving-force/weight assembly reproduces ``calculate_zpf_driving_forces``
semantics exactly (vertex ordering, NaN-hyperplane -> zero driving force,
underdetermined-vertex exclusion from the hyperplane average).

This module intentionally does not import ESPEI at module level; it is
designed to be driven from an ESPEI residual class (the registry is
pluggable) or a test harness that already has ``zpf_data``.
"""
from collections import defaultdict

import numpy as np

# Compositions outside this window go to the serial fallback: the accelerated
# capability gate routes dilute conditions to the reference path anyway, and
# pure-edge vertices have dedicated clamping there.
_X_MIN = 1e-6
_X_MAX = 1.0 - 1e-6


def _vertex_x_condition(vertex):
    """The single independent (X(comp), value) pair of a binary vertex."""
    items = list(vertex.comp_conds.items())
    if len(items) != 1:
        return None  # ternary+ vertex grouping not implemented yet
    return items[0]


def _batchable(vertex):
    if vertex.has_missing_comp_cond or vertex.is_disordered:
        return False
    xc = _vertex_x_condition(vertex)
    if xc is None:
        return False
    return _X_MIN <= float(xc[1]) <= _X_MAX


class BatchedZPFCalculator:
    """Batched drop-in for ``calculate_zpf_driving_forces``.

    Parameters
    ----------
    zpf_data : output of ``espei.error_functions.zpf_error.get_zpf_data``
    param_names : sorted fit-parameter symbol names (str order must match
        ``extract_parameters``, e.g. ['VV0000', 'VV0001'])
    backend : 'c++' or 'gpu' (accelerated pycalphad backend name)
    """

    def __init__(self, zpf_data, param_names, backend='c++'):
        self.zpf_data = zpf_data
        self.param_names = [str(p) for p in param_names]
        self.backend = backend
        self._plan()

    # ------------------------------------------------------------------ plan
    def _plan(self):
        """Enumerate vertex solves and group them into cartesian batches."""
        # job records: (dg, pr, kind, index-within-kind, vertex)
        self.hyp_jobs = {}      # (dg, pr, hv) -> ('batch', group_key, T, x) | ('serial',)
        self.df_jobs = {}       # (dg, pr, vi) -> ('batch', group_key, T, x) | ('serial',)
        hyp_groups = defaultdict(set)   # (x_comp,) -> set of (P, T, xval)
        iso_groups = defaultdict(set)   # (phase, x_comp) -> set of (P, T, xval)

        for dg, data in enumerate(self.zpf_data):
            for pr, region in enumerate(data['phase_regions']):
                pot = {str(k): float(vv) for k, vv in region.potential_conds.items()}
                T, P = pot['T'], pot.get('P', 101325.0)
                for hv, vtx in enumerate(region.hyperplane_vertices):
                    if vtx.has_missing_comp_cond:
                        continue  # contributes nothing to the hyperplane
                    if _batchable(vtx):
                        xkey, xval = _vertex_x_condition(vtx)
                        gk = (str(xkey),)
                        hyp_groups[gk].add((P, T, float(xval)))
                        self.hyp_jobs[(dg, pr, hv)] = ('batch', gk, P, T, float(xval))
                    else:
                        self.hyp_jobs[(dg, pr, hv)] = ('serial',)
                for vi, vtx in enumerate(region.vertices):
                    if _batchable(vtx):
                        xkey, xval = _vertex_x_condition(vtx)
                        gk = (vtx.phase_name, str(xkey))
                        iso_groups[gk].add((P, T, float(xval)))
                        self.df_jobs[(dg, pr, vi)] = ('batch', gk, P, T, float(xval))
                    else:
                        self.df_jobs[(dg, pr, vi)] = ('serial',)

        def _axes(points):
            Ps = np.array(sorted({p for p, _, _ in points}))
            Ts = np.array(sorted({t for _, t, _ in points}))
            Xs = np.array(sorted({x for _, _, x in points}))
            return Ps, Ts, Xs

        self.hyp_axes = {gk: _axes(pts) for gk, pts in hyp_groups.items()}
        self.iso_axes = {gk: _axes(pts) for gk, pts in iso_groups.items()}

        # Shared model/species/phase context (identical across zpf_data by
        # construction in get_zpf_data for one system).
        region0 = self.zpf_data[0]['phase_regions'][0]
        self.species = region0.species
        self.phases = list(region0.phases)
        self.models = region0.models
        self.dbf = self.zpf_data[0]['dbf']
        self.nonvacant = sorted(
            {el.upper() for sp in self.species for el in sp.constituents} - {'VA'})

    # ------------------------------------------------------------ batch runs
    def _run_batches(self, params_dict):
        """Run all cartesian batches; return per-group result objects."""
        import pycalphad
        from pycalphad import equilibrium, variables as v

        hyp_results, iso_results = {}, {}
        with pycalphad.backend(self.backend):
            for gk, (Ps, Ts, Xs) in self.hyp_axes.items():
                (xname,) = gk
                conds = {v.N: 1, v.P: Ps.tolist(), v.T: Ts.tolist(),
                         v.X(xname[2:]): Xs.tolist()}
                hyp_results[gk] = equilibrium(
                    self.dbf, [str(s) for s in self.species], self.phases,
                    conds, model=self.models, parameters=params_dict)
            for gk, (Ps, Ts, Xs) in self.iso_axes.items():
                phase, xname = gk
                conds = {v.N: 1, v.P: Ps.tolist(), v.T: Ts.tolist(),
                         v.X(xname[2:]): Xs.tolist()}
                iso_results[gk] = equilibrium(
                    self.dbf, [str(s) for s in self.species], [phase],
                    conds, model=self.models, parameters=params_dict)
        return hyp_results, iso_results

    @staticmethod
    def _grid_index(axes, P, T, x):
        Ps, Ts, Xs = axes
        return (int(np.searchsorted(Ps, P)), int(np.searchsorted(Ts, T)),
                int(np.searchsorted(Xs, x)))

    def _extract_mu(self, res, axes, P, T, x):
        """Per-vertex MU with the reference underdetermined-vertex rule."""
        pi, ti, xi = self._grid_index(axes, P, T, x)
        # result dims: N, P, T, X -> squeeze N
        phase_arr = res.Phase.values[0, pi, ti, xi]
        Y_arr = res.Y.values[0, pi, ti, xi]
        mu = np.array([float(res.MU.sel(component=el).values[0, pi, ti, xi])
                       for el in self.nonvacant])
        num_phases = int(np.sum(phase_arr != ''))
        no_internal_dof = bool(np.all(np.isclose(Y_arr, 1.0) | np.isnan(Y_arr)))
        if num_phases == 1 and no_internal_dof:
            return np.full_like(mu, np.nan)
        return mu

    # --------------------------------------------------------- serial pieces
    def _serial_hyperplane_mu(self, region, vertex, params_dict):
        """Reference per-vertex MU (Workspace), for non-batchable vertices."""
        from pycalphad import Workspace, variables as v
        cond_dict = {**vertex.comp_conds, **region.potential_conds}
        wks = Workspace(database=self.dbf, components=self.species,
                        phases=self.phases, models=self.models,
                        phase_record_factory=vertex.phase_record_factory,
                        conditions=cond_dict, parameters=params_dict)
        mu = np.array([float(wks.get(v.MU(el))) for el in self.nonvacant])
        num_phases = int(np.sum(wks.eq.Phase.squeeze() != ''))
        Y_values = wks.eq.Y.squeeze()
        no_internal_dof = np.all(np.isclose(Y_values, 1.0) | np.isnan(Y_values))
        if num_phases == 1 and no_internal_dof:
            return np.full_like(mu, np.nan)
        return mu

    def _serial_driving_force(self, data, region, vertex, target, parameters):
        """Reference driving_force_to_hyperplane for non-batchable vertices."""
        from espei.error_functions.zpf_error import driving_force_to_hyperplane
        return driving_force_to_hyperplane(
            target, region, data['dbf'], data['parameter_dict'], vertex,
            parameters)

    # ------------------------------------------------------------------ main
    def driving_forces(self, parameters):
        """Batched equivalent of ``calculate_zpf_driving_forces``."""
        parameters = np.asarray(parameters, dtype=np.float64)
        params_dict = dict(zip(self.param_names, parameters))
        hyp_results, iso_results = self._run_batches(params_dict)

        driving_forces, weights = [], []
        for dg, data in enumerate(self.zpf_data):
            data_dfs, data_wts = [], []
            weight = data['weight']
            for pr, region in enumerate(data['phase_regions']):
                # 1. target hyperplane = nanmean of vertex MU rows
                rows = []
                for hv, vtx in enumerate(region.hyperplane_vertices):
                    job = self.hyp_jobs.get((dg, pr, hv))
                    if job is None:
                        continue
                    if job[0] == 'batch':
                        _, gk, P, T, x = job
                        rows.append(self._extract_mu(
                            hyp_results[gk], self.hyp_axes[gk], P, T, x))
                    else:
                        rows.append(self._serial_hyperplane_mu(
                            region, region.hyperplane_vertices[hv], params_dict))
                if rows:
                    with np.errstate(invalid='ignore'):
                        target = np.nanmean(np.asarray(rows), axis=0)
                else:
                    target = np.full(len(self.nonvacant), np.nan)

                if np.any(np.isnan(target)):
                    data_dfs.extend([0] * len(region.vertices))
                    data_wts.extend([weight] * len(region.vertices))
                    continue

                # 2. per-vertex driving forces
                for vi, vtx in enumerate(region.vertices):
                    job = self.df_jobs[(dg, pr, vi)]
                    if job[0] == 'batch':
                        _, gk, P, T, x = job
                        res = iso_results[gk]
                        pi, ti, xi = self._grid_index(self.iso_axes[gk], P, T, x)
                        gm = float(res.GM.values[0, pi, ti, xi])
                        df = float(np.dot(target, vtx.composition) - gm)
                    else:
                        df = self._serial_driving_force(
                            data, region, vtx, target, parameters)
                    data_dfs.append(df)
                    data_wts.append(weight)
            driving_forces.append(data_dfs)
            weights.append(data_wts)
        return driving_forces, weights

    def likelihood(self, parameters, data_weight=1.0):
        """Batched equivalent of ``calculate_zpf_error``."""
        from scipy.stats import norm
        dfs, wts = self.driving_forces(parameters)
        dfs = np.concatenate([np.asarray(d, dtype=np.float64) for d in dfs])
        wts = np.concatenate([np.asarray(w, dtype=np.float64) for w in wts])
        if np.any(np.isinf(dfs) | np.isnan(dfs)):
            return -np.inf
        return float(np.sum(norm.logpdf(dfs, loc=0, scale=1000 / data_weight / wts)))
