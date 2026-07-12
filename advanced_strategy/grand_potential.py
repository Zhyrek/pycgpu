"""
Grand Potential Phase Diagram Solver

Computes phase equilibria by working in chemical potential (mu) space
rather than composition (x) space.

In mu-space, finding the stable phase at each point is a simple minimum
comparison rather than a convex hull construction.

For each phase phi and chemical potential vector mu:
    Omega_phi(mu) = min_y [G_phi(T,P,y) - sum_i mu_i * x_i(y)]

The stable phase is: argmin_phi Omega_phi(mu)
Phase boundaries are where Omega curves of different phases cross.

Key insight: the Legendre transform converts the global optimization problem
(convex hull in composition space) into many independent local optimizations
(one per phase per mu-point). This is embarrassingly parallel and ideal for GPU.

Sublattice model handling:
    For sublattice models with internal degrees of freedom, the inner
    minimization min_y [G(y) - mu*x(y)] is over site fractions y subject
    to sublattice sum constraints. At boundaries of site fraction space
    (y -> 0 or 1), the ideal mixing term RT*y*ln(y) dominates the gradient,
    causing all gradient vectors to align (degenerate). The non-degenerate
    manifold where non-ideal energetics matter is lower-dimensional than
    the full site fraction space. The optimizer naturally finds this manifold
    by starting from endmember corners and converging inward.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from collections import OrderedDict
import itertools
import time

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory

MIN_SITE_FRACTION = 1e-12


class SublatticeOptimizer:
    """
    Handles the inner minimization for a single phase:
        min_y [G(y) - mu*x(y)]  subject to sublattice constraints

    Builds a mapping between free variables (reduced DOFs) and full
    site fraction arrays, accounting for sublattice sum constraints.

    For phases with internal degrees of freedom (ordered sublattice models),
    the extreme values of site fractions produce degenerate partial
    derivatives -- the RT*ln(y) ideal mixing terms dominate and all gradient
    vectors align. The interesting optimization manifold (where non-ideal
    terms and chemical potential coupling matter) is lower-dimensional.
    We handle this by starting from multiple endmember corners and letting
    the optimizer converge onto the non-degenerate manifold naturally.
    """

    def __init__(self, phase_record, model):
        self.phase_record = phase_record
        self.model = model
        self.phase_name = phase_record.phase_name
        self.num_statevars = phase_record.num_statevars
        self.phase_dof = phase_record.phase_dof
        self.num_components = len(phase_record.nonvacant_elements)

        # Map state variable names to their index in the dof array
        # so we can correctly set T, P, N etc.
        self.state_variable_names = [str(sv) for sv in phase_record.state_variables]

        # Parse sublattice structure from model
        self.constituents = model.constituents
        self.site_ratios = list(model.site_ratios)

        # Build sublattice info: for each sublattice, which species (sorted)
        # and how many. Also track which sublattices are VA-only.
        self.sublattice_species = []
        self.sublattice_sizes = []
        self.va_only_sublattices = []
        for subl_idx, subl in enumerate(self.constituents):
            species = sorted(subl, key=str)
            self.sublattice_species.append(species)
            n = len(species)
            self.sublattice_sizes.append(n)
            # Check if this sublattice only has VA
            va_only = all(
                (sp.name if hasattr(sp, 'name') else str(sp)) == 'VA'
                for sp in species
            )
            self.va_only_sublattices.append(va_only)

        # Free DOFs: for each sublattice with n species, n-1 free variables
        # (the last is determined by the sum=1 constraint)
        # VA-only sublattices have 0 free DOFs
        self.free_dof = 0
        self._free_dof_per_sublattice = []
        for subl_idx, n in enumerate(self.sublattice_sizes):
            if self.va_only_sublattices[subl_idx]:
                self._free_dof_per_sublattice.append(0)
            else:
                ndof = n - 1
                self._free_dof_per_sublattice.append(ndof)
                self.free_dof += ndof

        # Bounds for free variables
        self.bounds = [(MIN_SITE_FRACTION, 1.0 - MIN_SITE_FRACTION)] * self.free_dof

        # Pre-allocate work arrays
        self._dof = np.zeros(self.num_statevars + self.phase_dof)
        self._out_G = np.zeros(1)
        self._out_x = np.zeros(1)

    def _build_dof(self, T, P, site_fracs):
        """
        Build the DOF array with correct state variable ordering.

        PhaseRecord expects dof = [sv_0, sv_1, ..., y_0, y_1, ...]
        where state variables are in the order given by phase_record.state_variables.
        This might be [T] or [P, T] or [N, P, T] depending on the model.
        """
        dof = np.zeros(self.num_statevars + self.phase_dof)
        for i, sv_name in enumerate(self.state_variable_names):
            if sv_name == 'T':
                dof[i] = T
            elif sv_name == 'P':
                dof[i] = P
            else:
                dof[i] = 0.0  # Unknown state variable
        dof[self.num_statevars:] = site_fracs
        return dof

    def free_to_full_site_fracs(self, free_vars):
        """
        Convert free variables to full site fraction array.

        For each sublattice with n species, the first n-1 site fractions
        are free variables, and the last is 1 - sum(others).
        VA-only sublattices are always set to 1.0.
        """
        site_fracs = np.zeros(self.phase_dof)
        free_idx = 0
        sf_idx = 0
        for subl_idx, n in enumerate(self.sublattice_sizes):
            if self.va_only_sublattices[subl_idx]:
                # VA-only: all site fracs = 1/n (or 1.0 if n=1)
                for j in range(n):
                    site_fracs[sf_idx] = 1.0 / n
                    sf_idx += 1
            elif n == 1:
                site_fracs[sf_idx] = 1.0
                sf_idx += 1
            else:
                remaining = 1.0
                for j in range(n - 1):
                    val = np.clip(free_vars[free_idx], MIN_SITE_FRACTION,
                                  1.0 - MIN_SITE_FRACTION)
                    site_fracs[sf_idx] = val
                    remaining -= val
                    sf_idx += 1
                    free_idx += 1
                site_fracs[sf_idx] = max(MIN_SITE_FRACTION, remaining)
                sf_idx += 1
        return site_fracs

    def full_to_free_site_fracs(self, site_fracs):
        """Convert full site fractions to free variables."""
        free_vars = []
        sf_idx = 0
        for subl_idx, n in enumerate(self.sublattice_sizes):
            if self.va_only_sublattices[subl_idx]:
                sf_idx += n
            elif n == 1:
                sf_idx += 1
            else:
                for j in range(n - 1):
                    free_vars.append(site_fracs[sf_idx])
                    sf_idx += 1
                sf_idx += 1  # skip dependent
        return np.array(free_vars) if free_vars else np.array([])

    def eval_grand_potential(self, T, P, mu_vector, site_fracs):
        """
        Evaluate Omega = G(y) - sum_i mu_i * x_i(y) at given site fractions.

        Parameters
        ----------
        T, P : float
        mu_vector : array, shape (num_components,)
        site_fracs : array, shape (phase_dof,)

        Returns
        -------
        omega : float
        """
        dof = self._build_dof(T, P, site_fracs)

        # G per mole of atoms
        out_G = np.zeros(1)
        self.phase_record.obj(out_G, dof)
        G = out_G[0]

        # mu dot x
        mu_dot_x = 0.0
        out_x = np.zeros(1)
        for comp_idx in range(self.num_components):
            self.phase_record.mass_obj(out_x, dof, comp_idx)
            mu_dot_x += mu_vector[comp_idx] * out_x[0]

        return G - mu_dot_x

    def minimize_grand_potential(self, T, P, mu_vector, y_init=None):
        """
        Find min_y [G(y) - mu*x(y)] for this phase.

        Parameters
        ----------
        T, P : float
        mu_vector : array, shape (num_components,)
        y_init : array or None
            Initial guess for site fractions (continuation from previous mu)

        Returns
        -------
        omega : float
            Minimum grand potential value
        y_opt : array
            Optimal full site fractions
        x_opt : array
            Corresponding mole fractions
        converged : bool
        """
        def objective(free_vars):
            sf = self.free_to_full_site_fracs(free_vars)
            return self.eval_grand_potential(T, P, mu_vector, sf)

        best_omega = np.inf
        best_sf = None
        best_converged = False

        starting_points = self._get_starting_points(y_init)

        for y_start in starting_points:
            free_start = self.full_to_free_site_fracs(y_start)

            if self.free_dof == 0:
                # Stoichiometric or VA-only: no optimization needed
                omega = self.eval_grand_potential(T, P, mu_vector, y_start)
                if omega < best_omega:
                    best_omega = omega
                    best_sf = y_start.copy()
                    best_converged = True

            elif self.free_dof == 1:
                # 1D optimization -- use bounded scalar minimizer
                res = minimize_scalar(
                    lambda x: objective(np.array([x])),
                    bounds=(MIN_SITE_FRACTION, 1.0 - MIN_SITE_FRACTION),
                    method='bounded',
                    options={'xatol': 1e-12}
                )
                if res.fun < best_omega:
                    best_omega = res.fun
                    best_sf = self.free_to_full_site_fracs(np.array([res.x]))
                    best_converged = True

            else:
                # Multi-dimensional: L-BFGS-B with bounds
                # For sublattices with 3+ species, we also need the sum
                # constraint. Use a penalty approach for simplicity.
                def penalized_objective(fv):
                    sf = self.free_to_full_site_fracs(fv)
                    omega = self.eval_grand_potential(T, P, mu_vector, sf)
                    # Penalty for dependent site fracs going below MIN
                    sf_idx = 0
                    for subl_idx, n in enumerate(self.sublattice_sizes):
                        if self.va_only_sublattices[subl_idx] or n <= 1:
                            sf_idx += n
                            continue
                        dep = sf[sf_idx + n - 1]
                        if dep < MIN_SITE_FRACTION:
                            omega += 1e10 * (MIN_SITE_FRACTION - dep) ** 2
                        sf_idx += n
                    return omega

                res = minimize(
                    penalized_objective,
                    free_start,
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500}
                )
                if res.fun < best_omega:
                    best_omega = res.fun
                    best_sf = self.free_to_full_site_fracs(res.x)
                    best_converged = res.success

        if best_sf is None:
            return np.inf, np.zeros(self.phase_dof), np.zeros(self.num_components), False

        # Get mole fractions at optimum
        dof = self._build_dof(T, P, best_sf)

        x_opt = np.zeros(self.num_components)
        for comp_idx in range(self.num_components):
            out_x = np.zeros(1)
            self.phase_record.mass_obj(out_x, dof, comp_idx)
            x_opt[comp_idx] = out_x[0]

        return best_omega, best_sf, x_opt, best_converged

    def _get_starting_points(self, y_init=None):
        """
        Generate starting points for the inner minimization.

        Strategy:
        1. If a previous solution is provided (continuation), use it
        2. Midpoint of each sublattice (equal mixing)
        3. Endmember corners (each species dominant in turn)

        The endmember corners are important for finding the non-degenerate
        manifold: from each corner, the optimizer follows the steep ideal-mixing
        gradient into the interior, converging onto the manifold where the
        actual minimum lies.
        """
        points = []

        # Previous solution (continuation)
        if y_init is not None:
            points.append(y_init.copy())

        # Midpoint
        midpoint = np.zeros(self.phase_dof)
        sf_idx = 0
        for subl_idx, n in enumerate(self.sublattice_sizes):
            if self.va_only_sublattices[subl_idx]:
                for j in range(n):
                    midpoint[sf_idx] = 1.0 / n
                    sf_idx += 1
            else:
                for j in range(n):
                    midpoint[sf_idx] = 1.0 / n
                    sf_idx += 1
        points.append(midpoint)

        # Endmember corners: for each sublattice, try each species as dominant
        endmember_options = []
        for subl_idx, n in enumerate(self.sublattice_sizes):
            if self.va_only_sublattices[subl_idx] or n == 1:
                # Fixed sublattice
                y = np.zeros(n)
                y[:] = 1.0 / n if self.va_only_sublattices[subl_idx] else 1.0
                endmember_options.append([y])
            else:
                subl_opts = []
                for dominant in range(n):
                    y = np.full(n, MIN_SITE_FRACTION * 100)
                    y[dominant] = 1.0 - (n - 1) * MIN_SITE_FRACTION * 100
                    subl_opts.append(y)
                endmember_options.append(subl_opts)

        # All combinations of endmember choices across sublattices
        # (limit to avoid combinatorial explosion with many sublattices)
        max_combos = 50
        combos = list(itertools.product(*endmember_options))
        if len(combos) > max_combos:
            # Sample a subset
            rng = np.random.RandomState(42)
            indices = rng.choice(len(combos), max_combos, replace=False)
            combos = [combos[i] for i in indices]

        for combo in combos:
            y = np.concatenate(combo)
            points.append(y)

        return points


def _estimate_mu_range(optimizers, phase_records, T, P, nonvacant):
    """
    Estimate the range of chemical potential differences to grid.

    For a binary A-B with reference component A (index 0):
        delta_mu = mu_B - mu_A
    ranges from approximately G'(x_B -> 0) to G'(x_B -> 1) for each phase.

    We evaluate G at several compositions and numerically estimate dG/dx
    to find the range. The ideal mixing term contributes RT*ln(x/(1-x))
    which diverges at the boundaries, so we use a practical minimum
    composition of 10^-6 for range estimation.
    """
    num_comps = len(nonvacant)
    mu_values = []

    for phase_name, optimizer in optimizers.items():
        pr = phase_records[phase_name]

        # Test compositions: near boundaries and intermediate
        x_tests = [1e-6, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1-1e-4, 1-1e-6]

        for x_test in x_tests:
            if num_comps != 2:
                continue  # Only binary range estimation for now

            # Build site fractions for this composition
            sf = _build_site_fracs_for_composition(
                optimizer, nonvacant, {nonvacant[1]: x_test}
            )
            if sf is None:
                continue

            dof = optimizer._build_dof(T, P, sf)

            out_G = np.zeros(1)
            pr.obj(out_G, dof)
            G0 = out_G[0]

            out_x = np.zeros(1)
            pr.mass_obj(out_x, dof, 1)
            x0 = out_x[0]

            dx = 1e-8
            if x_test + dx > 1.0 - 1e-6:
                dx = -1e-8

            sf2 = _build_site_fracs_for_composition(
                optimizer, nonvacant, {nonvacant[1]: x_test + dx}
            )
            if sf2 is None:
                continue

            dof2 = optimizer._build_dof(T, P, sf2)

            out_G2 = np.zeros(1)
            pr.obj(out_G2, dof2)

            pr.mass_obj(out_x, dof2, 1)
            x1 = out_x[0]

            if abs(x1 - x0) > 1e-15:
                dGdx = (out_G2[0] - G0) / (x1 - x0)
                mu_values.append(dGdx)

    if mu_values:
        mu_min = min(mu_values)
        mu_max = max(mu_values)
        # Add some padding
        margin = 0.1 * (mu_max - mu_min)
        mu_min -= margin
        mu_max += margin
        # Ensure minimum range
        if mu_max - mu_min < 1000:
            center = (mu_max + mu_min) / 2
            mu_min = center - 500
            mu_max = center + 500
    else:
        # Fallback based on ideal mixing range
        RT = 8.314 * T
        mu_min = -12 * 2.303 * RT
        mu_max = 12 * 2.303 * RT

    return mu_min, mu_max


def _build_site_fracs_for_composition(optimizer, nonvacant, target_x):
    """
    Build site fractions that correspond to a target composition.

    For simple substitutional phases (1 sublattice), site fracs = mole fracs.
    For sublattice models, this is approximate -- set mixing sublattices
    to match target composition proportionally.

    Parameters
    ----------
    optimizer : SublatticeOptimizer
    nonvacant : list of str
        Sorted non-vacancy component names
    target_x : dict
        {component_name: mole_fraction} for independent components

    Returns
    -------
    site_fracs : ndarray or None
    """
    sf = np.zeros(optimizer.phase_dof)
    sf_idx = 0

    for subl_idx, species_list in enumerate(optimizer.sublattice_species):
        n = len(species_list)

        if optimizer.va_only_sublattices[subl_idx]:
            for j in range(n):
                sf[sf_idx] = 1.0 / n
                sf_idx += 1
            continue

        # Map species to component names
        sp_names = []
        for sp in species_list:
            name = sp.name if hasattr(sp, 'name') else str(sp)
            sp_names.append(name)

        # Check if this sublattice has VA mixed with other species
        has_va = 'VA' in sp_names
        mixing_species = [name for name in sp_names if name != 'VA']

        if len(mixing_species) == 0:
            # Pure VA sublattice handled above
            for j in range(n):
                sf[sf_idx] = 1.0
                sf_idx += 1
            continue

        # Assign site fractions based on target composition
        for j, name in enumerate(sp_names):
            if name == 'VA':
                sf[sf_idx] = MIN_SITE_FRACTION
            elif name in target_x:
                sf[sf_idx] = np.clip(target_x[name], MIN_SITE_FRACTION,
                                     1.0 - MIN_SITE_FRACTION)
            elif name in nonvacant:
                # Dependent component: 1 - sum(others)
                others = sum(target_x.get(c, 0) for c in nonvacant if c != name)
                sf[sf_idx] = np.clip(1.0 - others, MIN_SITE_FRACTION,
                                     1.0 - MIN_SITE_FRACTION)
            else:
                sf[sf_idx] = MIN_SITE_FRACTION
            sf_idx += 1

        # Normalize this sublattice to sum to 1
        subl_start = sf_idx - n
        subl_sum = np.sum(sf[subl_start:sf_idx])
        if subl_sum > 0:
            sf[subl_start:sf_idx] /= subl_sum

    return sf


def _refine_boundary(optimizer_a, optimizer_b, T, P, mu_left, mu_right,
                     mu_vector_func, tol=1e-2):
    """
    Refine a phase boundary location in mu-space using bisection.

    Finds mu* where Omega_a(mu*) = Omega_b(mu*).

    Parameters
    ----------
    optimizer_a, optimizer_b : SublatticeOptimizer
    T, P : float
    mu_left, mu_right : float
        Bracket: Omega_a < Omega_b at mu_left, Omega_a > Omega_b at mu_right
    mu_vector_func : callable
        Maps scalar delta_mu to full mu_vector
    tol : float
        Tolerance on mu

    Returns
    -------
    mu_boundary : float
    x_a : array, composition of phase a at boundary
    x_b : array, composition of phase b at boundary
    """
    def diff(delta_mu):
        mu_vec = mu_vector_func(delta_mu)
        omega_a, _, x_a, _ = optimizer_a.minimize_grand_potential(T, P, mu_vec)
        omega_b, _, x_b, _ = optimizer_b.minimize_grand_potential(T, P, mu_vec)
        return omega_a - omega_b

    try:
        mu_star = brentq(diff, mu_left, mu_right, xtol=tol, maxiter=50)
    except ValueError:
        # Bisection failed (no sign change), use midpoint
        mu_star = 0.5 * (mu_left + mu_right)

    mu_vec = mu_vector_func(mu_star)
    _, _, x_a, _ = optimizer_a.minimize_grand_potential(T, P, mu_vec)
    _, _, x_b, _ = optimizer_b.minimize_grand_potential(T, P, mu_vec)

    return mu_star, x_a, x_b


def _vectorized_binary_omega(optimizer, phase_record, T, P, mu_grid, x_resolution=2000):
    """
    Fast vectorized computation of Omega(mu) for a binary substitutional phase.

    Instead of calling scipy.optimize for each mu point, we:
    1. Evaluate G(x) on a fine composition grid
    2. Compute Omega(x, mu) = G(x) - mu*x for all (x, mu) pairs via broadcasting
    3. For each mu, find the x that minimizes Omega

    This is O(N_mu * N_x) with pure numpy, avoiding Python-level loops.

    Parameters
    ----------
    optimizer : SublatticeOptimizer
    phase_record : PhaseRecord
    T, P : float
    mu_grid : array of delta_mu values
    x_resolution : int
        Number of composition grid points

    Returns
    -------
    omega_min : array, shape (N_mu,) - minimum Omega at each mu
    x_opt : array, shape (N_mu, 2) - composition at minimum for each mu
    """
    num_mu = len(mu_grid)

    # Fine grid of compositions for the second component
    # Use log-spaced points near boundaries for better resolution
    x_lin = np.linspace(MIN_SITE_FRACTION, 1.0 - MIN_SITE_FRACTION, x_resolution)

    # Evaluate G and x for each grid point
    G_grid = np.zeros(x_resolution)
    x_grid = np.zeros((x_resolution, 2))

    out_G = np.zeros(1)
    out_x = np.zeros(1)

    for i, x_val in enumerate(x_lin):
        # Build site fractions (for simple substitutional: sf = [1-x, x])
        sf = np.zeros(optimizer.phase_dof)
        sf_idx = 0
        for subl_idx, species_list in enumerate(optimizer.sublattice_species):
            n = len(species_list)
            if optimizer.va_only_sublattices[subl_idx]:
                for j in range(n):
                    sf[sf_idx] = 1.0 / n
                    sf_idx += 1
            elif n == 2:
                # Binary on this sublattice
                sf[sf_idx] = 1.0 - x_val
                sf[sf_idx + 1] = x_val
                sf_idx += 2
            elif n == 1:
                sf[sf_idx] = 1.0
                sf_idx += 1
            else:
                # Equal distribution (not exact for ternary+, but we only
                # use this fast path for binary systems)
                for j in range(n):
                    sf[sf_idx] = 1.0 / n
                    sf_idx += 1

        dof = optimizer._build_dof(T, P, sf)
        phase_record.obj(out_G, dof)
        G_grid[i] = out_G[0]

        for comp_idx in range(2):
            phase_record.mass_obj(out_x, dof, comp_idx)
            x_grid[i, comp_idx] = out_x[0]

    # Vectorized Omega computation:
    # Omega[i, j] = G[j] - mu_grid[i] * x_grid[j, 1]
    # where i indexes mu, j indexes composition
    # Shape: (N_mu, N_x)
    omega_matrix = G_grid[np.newaxis, :] - mu_grid[:, np.newaxis] * x_grid[np.newaxis, :, 1]

    # Find minimum for each mu
    min_idx = np.argmin(omega_matrix, axis=1)
    omega_min = omega_matrix[np.arange(num_mu), min_idx]
    x_opt = x_grid[min_idx]

    return omega_min, x_opt


def solve_binary(optimizers, phase_records, phases, nonvacant,
                 T, P, mu_grid, verbose=False):
    """
    Solve grand potential for a binary system at a single temperature.

    Uses vectorized computation: precomputes G on a fine composition grid,
    then finds optimal Omega for all mu values at once via numpy broadcasting.

    Parameters
    ----------
    optimizers : dict of {phase_name: SublatticeOptimizer}
    phase_records : dict of {phase_name: PhaseRecord}
    phases : list of str
    nonvacant : list of str, sorted non-vacancy components
    T, P : float
    mu_grid : array of delta_mu values (mu_1 - mu_0)
    verbose : bool

    Returns
    -------
    result : dict with omega, compositions, stable phases, boundaries
    """
    num_mu = len(mu_grid)
    num_phases = len(phases)

    omega_all = np.full((num_phases, num_mu), np.inf)
    x_at_mu = np.zeros((num_phases, num_mu, 2))

    for phase_idx, phase_name in enumerate(phases):
        optimizer = optimizers[phase_name]
        pr = phase_records[phase_name]

        if verbose:
            print(f"  Phase: {phase_name} (free DOFs: {optimizer.free_dof})")

        if optimizer.free_dof <= 1:
            # Fast vectorized path for simple substitutional phases
            omega_min, x_opt = _vectorized_binary_omega(
                optimizer, pr, T, P, mu_grid, x_resolution=2000
            )
            omega_all[phase_idx] = omega_min
            x_at_mu[phase_idx] = x_opt
        else:
            # Fallback to per-point optimization for complex sublattice models
            prev_y = None
            for mu_idx, delta_mu in enumerate(mu_grid):
                mu_vector = np.array([0.0, delta_mu])
                omega, y_opt, x_opt_pt, converged = optimizer.minimize_grand_potential(
                    T, P, mu_vector, y_init=prev_y
                )
                omega_all[phase_idx, mu_idx] = omega
                x_at_mu[phase_idx, mu_idx, :] = x_opt_pt
                if converged:
                    prev_y = y_opt

        if verbose:
            valid = omega_all[phase_idx] < np.inf
            if np.any(valid):
                print(f"    Omega range: [{np.min(omega_all[phase_idx, valid]):.1f}, "
                      f"{np.max(omega_all[phase_idx, valid]):.1f}]")
                print(f"    x_{nonvacant[1]} range: [{np.min(x_at_mu[phase_idx, valid, 1]):.6f}, "
                      f"{np.max(x_at_mu[phase_idx, valid, 1]):.6f}]")

    # Stable phase at each mu point
    stable_phase_idx = np.argmin(omega_all, axis=0)

    # Find phase boundaries with refinement
    mu_vector_func = lambda dm: np.array([0.0, dm])
    boundaries = []

    for i in range(1, num_mu):
        if stable_phase_idx[i] != stable_phase_idx[i - 1]:
            idx_a = stable_phase_idx[i - 1]
            idx_b = stable_phase_idx[i]
            phase_a = phases[idx_a]
            phase_b = phases[idx_b]

            # Refine the crossing point
            mu_star, x_a, x_b = _refine_boundary(
                optimizers[phase_a], optimizers[phase_b],
                T, P, mu_grid[i - 1], mu_grid[i],
                mu_vector_func
            )

            boundaries.append({
                'mu': mu_star,
                'phase_low_mu': phase_a,
                'phase_high_mu': phase_b,
                'x_phase_a': x_a.copy(),
                'x_phase_b': x_b.copy(),
            })

    # Build stable composition mapping
    stable_x = np.zeros((num_mu, 2))
    stable_phase_names = []
    for i in range(num_mu):
        stable_x[i] = x_at_mu[stable_phase_idx[i], i]
        stable_phase_names.append(phases[stable_phase_idx[i]])

    return {
        'mu_grid': mu_grid,
        'omega': omega_all,
        'x_at_mu': x_at_mu,
        'stable_phase_idx': stable_phase_idx,
        'stable_phase_names': stable_phase_names,
        'stable_x': stable_x,
        'phase_boundaries': boundaries,
    }


def compute_phase_diagram(dbf, comps, phases, conditions,
                          mu_resolution=300, verbose=False):
    """
    Compute phase diagram using the grand potential approach.

    Parameters
    ----------
    dbf : Database
    comps : list of str
        Component names (e.g., ['NB', 'TI'])
    phases : list of str
        Phase names (e.g., ['LIQUID', 'BCC_A2'])
    conditions : dict
        Must contain v.T (scalar or array), v.P.
        Composition conditions (v.X) are not needed -- the approach
        works in chemical potential space and covers all compositions.
    mu_resolution : int
        Number of grid points per chemical potential dimension
    verbose : bool

    Returns
    -------
    result : dict
        'components': list of non-vacancy components
        'phases': list of phase names
        'temperatures': array of T values
        'results_per_T': list of dicts, one per temperature, each containing:
            'T': temperature
            'mu_grid': chemical potential grid
            'omega': grand potential for each phase
            'stable_phase_idx': index of stable phase at each mu
            'phase_boundaries': list of boundary dicts with compositions
    """
    t_start = time.time()

    comps = sorted(set(comps) | {'VA'})
    nonvacant = sorted([c for c in comps if c != 'VA'])
    num_comps = len(nonvacant)

    # Build models and phase records
    models = {}
    active_phases = []
    for phase in phases:
        try:
            models[phase] = Model(dbf, comps, phase)
            active_phases.append(phase)
        except Exception as e:
            if verbose:
                print(f"Skipping phase {phase}: {e}")

    phases = active_phases
    state_variables = sorted(
        [sv for sv in models[phases[0]].state_variables], key=str
    )

    prf = PhaseRecordFactory(dbf, comps, state_variables, models)

    # Build optimizers
    optimizers = {}
    phase_records = {}
    for phase in phases:
        pr = prf[phase]
        phase_records[phase] = pr
        optimizers[phase] = SublatticeOptimizer(pr, models[phase])

    if verbose:
        print(f"Phases: {phases}")
        print(f"Components: {nonvacant}")
        for phase in phases:
            opt = optimizers[phase]
            print(f"  {phase}: {opt.phase_dof} site fracs, "
                  f"{opt.free_dof} free DOFs, "
                  f"sublattices: {opt.sublattice_sizes}")

    # Extract temperature values
    T_cond = conditions.get(v.T, 300)
    if isinstance(T_cond, (int, float)):
        T_values = np.array([T_cond])
    elif hasattr(T_cond, '__iter__'):
        T_values = np.array(list(T_cond))
    else:
        T_values = np.atleast_1d(np.array(T_cond))

    P = float(conditions.get(v.P, 101325))

    results_per_T = []

    for T_idx, T in enumerate(T_values):
        if verbose:
            print(f"\n--- T = {T:.1f} K ({T_idx+1}/{len(T_values)}) ---")

        # Estimate mu range
        mu_min, mu_max = _estimate_mu_range(optimizers, phase_records,
                                            T, P, nonvacant)
        if verbose:
            print(f"  mu range: [{mu_min:.1f}, {mu_max:.1f}] J/mol")

        mu_grid = np.linspace(mu_min, mu_max, mu_resolution)

        if num_comps == 2:
            result = solve_binary(
                optimizers, phase_records, phases, nonvacant,
                T, P, mu_grid, verbose
            )
        else:
            raise NotImplementedError(
                f"Multi-component ({num_comps}) grand potential not yet implemented. "
                "Binary systems only for now."
            )

        result['T'] = T
        result['P'] = P
        results_per_T.append(result)

    t_elapsed = time.time() - t_start

    return {
        'components': nonvacant,
        'phases': phases,
        'temperatures': T_values,
        'results_per_T': results_per_T,
        'elapsed_time': t_elapsed,
    }


def extract_phase_boundaries(diagram_result):
    """
    Extract phase boundary data suitable for plotting a T-x phase diagram.

    Parameters
    ----------
    diagram_result : dict
        Output from compute_phase_diagram

    Returns
    -------
    boundaries : dict
        'T': array of temperatures
        'x_boundaries': list of (x_a, x_b, phase_a, phase_b) tuples per T
        'single_phase_regions': list of dicts per T
    """
    T_values = diagram_result['temperatures']
    all_boundaries = []

    for res in diagram_result['results_per_T']:
        T = res['T']
        for bnd in res['phase_boundaries']:
            all_boundaries.append({
                'T': T,
                'mu': bnd['mu'],
                'phase_a': bnd['phase_low_mu'],
                'phase_b': bnd['phase_high_mu'],
                # x of second component (the one being varied)
                'x_a': bnd['x_phase_a'][1],
                'x_b': bnd['x_phase_b'][1],
            })

    return all_boundaries
