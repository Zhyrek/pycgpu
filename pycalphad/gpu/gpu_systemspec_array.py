"""Create an array of SystemSpecification structs for multi-condition GPU calculations."""

import numpy as np
from .gpu_equilibrium import (_get_c_define, _populate_system_specification, 
                             _create_system_specification_struct)
from .gpu_systemspec_flat import create_flat_system_specification, apply_safe_padding
from .gpu_properties_subset import PropertiesSubset


def create_system_specifications_array(wks_obj, num_conditions, dynamic_sizes, properties, verbose=False, device_xp=None):
    """
    Create an array of SystemSpecification structs, one per condition.
    
    This fixes the multi-condition GPU issue where all threads were sharing the same
    SystemSpecification, causing all conditions to use the same prescribed_mole_fraction_rhs.
    
    Parameters:
    -----------
    wks_obj : WorkspaceState
        The workspace object containing condition data
    num_conditions : int
        Number of conditions to process
    dynamic_sizes : dict
        Dynamic sizes for arrays
    properties : object
        Properties object with initial values
    verbose : bool
        Enable verbose output
        
    Returns:
    --------
    np.ndarray
        Flattened array of doubles containing all SystemSpecification structs
    """
    if verbose:
        print(f"[GPU] Creating SystemSpecification array for {num_conditions} conditions")
    
    # Get sizes
    max_components = dynamic_sizes["MAX_COMPONENTS"]
    max_statevars = dynamic_sizes["MAX_STATEVARS"]
    max_phases = dynamic_sizes["MAX_PHASES"]
    max_fixed_mole = dynamic_sizes["MAX_FIXED_MOLE_FRACTION_CONDITIONS"]
    
    
    # Extract condition arrays
    import pycalphad.variables as v
    
    # Collect ALL X() conditions for multi-component systems
    x_conditions = {}  # Will map component name to array of values
    x_components = []  # Ordered list of components with X() conditions
    
    for comp in wks_obj.components:
        if comp == 'VA':
            continue  # Skip vacancy
        x_var = v.X(comp)
        if x_var in wks_obj.conditions:
            condition_value = wks_obj.conditions[x_var]
            comp_str = str(comp) if not isinstance(comp, str) else comp
            x_conditions[comp_str] = np.asarray(condition_value).flatten()
            x_components.append(comp_str)
            if verbose:
                print(f"[GPU] Found X({comp}) condition with {len(x_conditions[comp_str])} values: {x_conditions[comp_str]}")
    
    # Phase-local conditions (scalar values; the dispatch gate rejects
    # array-valued ones): one system-wide entry list shared by every
    # per-condition spec. Entries are (model_idx, type, target, value):
    #   type 0 = X(phase, el), target = nonvacant component index;
    #   type 1 = Y(phase, subl, sp), target = site-fraction dof index in the
    #            generated-code ordering (get_ordered_symbols_for_diff).
    plc_entries = []
    _plc_conds = [(key, value) for key, value in wks_obj.conditions.items()
                  if getattr(key, 'phase_name', None) is not None]
    if _plc_conds:
        from .gpu_codegen import _unique_models_for_gpu, get_ordered_symbols_for_diff
        _u_models, _name_to_idx = _unique_models_for_gpu(wks_obj, validate=False)
        _nonvacant = [str(c) for c in sorted(wks_obj.components, key=str)
                      if str(c) != 'VA']
        for key, value in _plc_conds:
            _midx = _name_to_idx[key.phase_name]
            _val = float(np.asarray(value).reshape(-1)[0])
            if isinstance(key, v.MoleFraction):
                _target = _nonvacant.index(str(key.species.name))
                plc_entries.append((_midx, 0, _target, _val))
            elif isinstance(key, v.SiteFraction):
                _ysyms = get_ordered_symbols_for_diff(_u_models[_midx], wks_obj)[1:]
                _target = None
                for _yi, _sf in enumerate(_ysyms):
                    if (getattr(_sf, 'sublattice_index', None) == key.sublattice_index
                            and str(getattr(_sf, 'species', '')) == str(key.species)):
                        _target = _yi
                        break
                if _target is None:
                    raise ValueError(f'site fraction {key} not found in model dof')
                plc_entries.append((_midx, 1, _target, _val))
            else:
                raise ValueError(f'unsupported phase-local condition {key}')

    # For backward compatibility, keep x_ti_values as the first composition condition found
    x_ti_values = x_conditions[x_components[0]] if x_components else None
    
    if x_ti_values is None:
        # No mole fraction conditions, create single spec for all conditions
        x_ti_values = np.zeros(num_conditions)
        if verbose:
            print(f"[GPU] No mole fraction conditions found, creating {num_conditions} identical specs")
    
    # Get temperature and composition arrays
    temp_values = wks_obj.conditions[v.T]
    if not hasattr(temp_values, '__len__'):
        temp_values = [temp_values]
    temp_values = np.asarray(temp_values).flatten()
    
    # For multi-dimensional grids (e.g., T × X_CU × X_FE for ternary)
    # We need to properly calculate indices for each dimension
    grid_shape = [len(temp_values)]
    for comp in x_components:
        grid_shape.append(len(x_conditions[comp]))
    
    if verbose:
        print(f"[GPU] Grid shape: {grid_shape} (T × {' × '.join(['X('+c+')' for c in x_components])})")
    
    # ALL array-valued conditions in CPU result-dimension order (sorted by
    # str: MU_* < N < P < T < W_* < X_*). The C-order flat index over these
    # dims equals the flat condition index of the starting-point arrays.
    cond_dims = [(key, np.asarray(value).reshape(-1))
                 for key, value in sorted(wks_obj.conditions.items(), key=lambda kv: str(kv[0]))
                 if np.asarray(value).size > 1]
    cond_dim_shape = [vals.size for _, vals in cond_dims]

    def _build_one(condition_idx):
        """Build the padded flat spec for one condition (original per-condition path)."""
        # per-dim indices from the flat condition index (C order over cond_dims)
        dim_idx = {}
        remaining = condition_idx
        for (key, vals), dim_size in zip(reversed(cond_dims), reversed(cond_dim_shape)):
            dim_idx[key] = remaining % dim_size
            remaining //= dim_size

        temp_idx = int(dim_idx.get(v.T, 0))
        x_indices = {comp: int(dim_idx.get(v.X(comp), 0)) for comp in x_components}
        comp_idx = x_indices[x_components[0]] if x_components else 0

        global_spec_np = np.zeros(50, dtype=np.float64)  # Scalar fields
        global_spec_arrays = {
            'initial_chemical_potentials': np.zeros(max_components, dtype=np.float64),
            'prescribed_mole_fraction_coefficients': np.zeros((max_fixed_mole, max_components), dtype=np.float64),
            'prescribed_mole_fraction_rhs': np.zeros(max_fixed_mole, dtype=np.float64),
            'free_chemical_potential_indices': np.full(max_components, -1, dtype=np.int32),
            'free_statevar_indices': np.full(max_statevars, -1, dtype=np.int32),
            'fixed_chemical_potential_indices': np.full(max_components, -1, dtype=np.int32),
            'fixed_statevar_indices': np.full(max_statevars, -1, dtype=np.int32),
            'fixed_stable_compset_indices': np.full(max_phases, -1, dtype=np.int32)
        }

        class TempWorkspace:
            def __init__(self, original_wks, dim_idx):
                self.components = original_wks.components
                self.phase_record_factory = original_wks.phase_record_factory
                self.verbose = original_wks.verbose
                # every array condition (T, X, MU, W, ...) indexed by its own
                # dimension; scalars pass through
                self.conditions = {}
                for key, value in original_wks.conditions.items():
                    value_array = np.asarray(value).reshape(-1)
                    if value_array.size > 1:
                        self.conditions[key] = float(value_array[int(dim_idx.get(key, 0))])
                    else:
                        self.conditions[key] = float(value_array[0])

        temp_wks = TempWorkspace(wks_obj, dim_idx)
        if len(x_components) > 1:
            properties_subset = PropertiesSubset(properties, condition_idx, temp_idx, x_indices, verbose=verbose)
        else:
            properties_subset = PropertiesSubset(properties, condition_idx, temp_idx, comp_idx, verbose=verbose)

        global_spec_arrays['phase_local_conditions'] = plc_entries
        _populate_system_specification(global_spec_np, global_spec_arrays, temp_wks,
                                       dynamic_sizes, properties_subset)
        spec_doubles = create_flat_system_specification(global_spec_np, global_spec_arrays,
                                                        dynamic_sizes)
        return apply_safe_padding(spec_doubles, verbose=verbose)

    # FAST PATH: the flat spec is condition-invariant except for
    #   initial_chemical_potentials (per-condition starting MU from the hull) and
    #   prescribed_mole_fraction_rhs (per-condition X values).
    # Build condition 0 through the original machinery, tile it, and overwrite
    # those two field groups vectorized. The per-condition Python loop cost ~5s
    # at 10k conditions (pint conversions + object churn per condition).
    # PYCGPU_SPEC_SLOW=1 forces the original loop (verification tooling).
    import os as _os
    mu_full = np.asarray(properties.MU) if hasattr(properties, 'MU') else None
    # the vectorized rewrites below only cover T/X condition dims; MU/W/other
    # array conditions use the general per-condition path
    _plain_dims = all(key == v.T or (isinstance(key, v.MoleFraction)
                                     and getattr(key, 'phase_name', None) is None)
                      for key, _ in cond_dims)
    fast_ok = (
        not _os.environ.get('PYCGPU_SPEC_SLOW')
        and _plain_dims
        and mu_full is not None
        and mu_full.ndim >= 3
        and mu_full.shape[:2] == (1, 1)
        and mu_full.size == num_conditions * mu_full.shape[-1]
    )

    if fast_ok:
        spec0 = _build_one(0)
        # device_xp (cupy): tile + per-condition column writes happen ON
        # DEVICE — only the template row and the n-length columns cross the
        # bus instead of the full (n, stride) array (~1 GB at 1M conditions).
        _xp = device_xp if device_xp is not None else np
        specs_array = _xp.tile(_xp.asarray(spec0), (num_conditions, 1))

        # Multi-dim index arrays for every condition (same little-endian
        # decomposition as _build_one, which matches C-order flattening of
        # the [T, X1, X2, ...] grid).
        rem = np.arange(num_conditions)
        dim_indices = []
        for dim_size in reversed(grid_shape[1:]):
            dim_indices.append(rem % dim_size)
            rem = rem // dim_size
        dim_indices.append(rem)
        dim_indices.reverse()
        x_idx_arrs = {comp: dim_indices[i + 1] for i, comp in enumerate(x_components)}

        # Flat-layout offsets (create_flat_system_specification order; padding
        # appends at the end so offsets are stable). NOTE: the flat packer pins
        # MAX_FIXED_MOLE_FRACTION_CONDITIONS = MAX_COMPONENTS.
        MC = int(dynamic_sizes["MAX_COMPONENTS"])
        off_mu = 3
        off_rhs = 3 + MC + MC * MC

        # prescribed_mole_fraction_rhs: same constraint enumeration order as
        # _populate_system_specification (conditions dict order, nonvacant only).
        nonvacant = [c for c in wks_obj.components[:MC] if 'VA' not in str(c).upper()]
        nonvacant_names = [str(c).upper() for c in nonvacant]
        constraint_count = 0
        for cond, value in wks_obj.conditions.items():
            if isinstance(cond, v.MoleFraction) and cond.phase_name is None and constraint_count < MC:
                el = str(cond)[2:]
                if el not in nonvacant_names:
                    continue
                varr = np.asarray(value).flatten()
                if el in x_idx_arrs and varr.size > 1:
                    vals = varr[x_idx_arrs[el]]
                else:
                    vals = np.full(num_conditions, float(varr.flat[0]))
                specs_array[:, off_rhs + constraint_count] = _xp.asarray(vals)
                constraint_count += 1

        # initial_chemical_potentials: FREE chempots take per-condition starting
        # values from properties.MU (C-order flatten matches the condition index);
        # FIXED chempots (a MU condition) are condition-invariant, already in spec0.
        n_mu_comp = mu_full.shape[-1]
        mu_flat = np.ascontiguousarray(mu_full).reshape(num_conditions, n_mu_comp)
        for comp_idx, comp in enumerate(nonvacant):
            if v.ChemicalPotential(comp) in wks_obj.conditions:
                continue
            if comp_idx < n_mu_comp:
                specs_array[:, off_mu + comp_idx] = _xp.asarray(
                    np.ascontiguousarray(mu_flat[:, comp_idx]))

        if verbose:
            print(f"[GPU] SystemSpecification array built via fast path "
                  f"({num_conditions} conditions x {specs_array.shape[1]} doubles)")
        return specs_array.reshape(-1)

    # SLOW PATH (fallback / PYCGPU_SPEC_SLOW=1): original per-condition loop.
    all_specs = [_build_one(condition_idx) for condition_idx in range(num_conditions)]
    specs_array = np.vstack(all_specs)

    if verbose:
        print(f"\n[GPU] Created SystemSpecification array:")
        print(f"  Shape: {specs_array.shape}")
        print(f"  Total size: {specs_array.nbytes} bytes")
        print(f"  Specs per condition: {specs_array.shape[1]} doubles")

    return specs_array.flatten()