"""
GPU-accelerated Grand Potential Phase Diagram Solver

Uses CuPy to run the entire computation on GPU:
1. Gibbs energy evaluation for all (T, x) pairs in parallel
2. Grand potential Omega = G - mu*x computed via broadcasting
3. Stable phase identification via argmin

The CUDA kernel is generated dynamically from pycalphad's symbolic
Model expressions using symengine CSE + ccode.
"""

import numpy as np
import time
import re
import hashlib
import os

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory

MIN_SITE_FRACTION = 1e-12

# Import the CPU solver for mu range estimation and boundary refinement
from advanced_strategy.grand_potential import (
    SublatticeOptimizer, _estimate_mu_range, _build_site_fracs_for_composition,
    _refine_boundary, extract_phase_boundaries
)


def _generate_G_device_function(model, func_idx):
    """
    Generate a CUDA __device__ function for Gibbs energy evaluation.

    Converts the symbolic GM expression to C code using symengine's
    CSE (Common Subexpression Elimination) and ccode.

    Parameters
    ----------
    model : pycalphad.Model
    func_idx : int
        Index for unique function naming

    Returns
    -------
    func_code : str
        CUDA __device__ function source code
    func_name : str
        Name of the generated function
    site_frac_names : list of str
        Names of site fraction symbols in order
    """
    from symengine import cse
    from symengine.lib.symengine_wrapper import ccode

    gm_expr = model.GM

    # Get all free symbols and categorize them
    all_syms = sorted(gm_expr.free_symbols, key=str)

    # Use ccode() to get C-compatible names for site fractions
    # str(sym) gives display names like "Y(LIQUID,0,NB)"
    # ccode(sym) gives internal names like "LIQUID0NB" which are valid C identifiers
    site_frac_cnames = []  # C-compatible names (e.g., LIQUID0NB)
    site_frac_syms = []    # The actual symengine symbols
    for sym in all_syms:
        display_name = str(sym)
        if display_name not in ('T', 'P', 'N'):
            c_name = ccode(sym)
            site_frac_cnames.append(c_name)
            site_frac_syms.append(sym)

    # Apply CSE for optimized code generation
    replacements, reduced = cse([gm_expr])

    # Generate C code body
    lines = []
    for sym, subexpr in replacements:
        c_sym = ccode(sym)
        c_expr = ccode(subexpr)
        lines.append(f"    double {c_sym} = {c_expr};")
    lines.append(f"    return {ccode(reduced[0])};")

    body = "\n".join(lines)
    body = _fix_ccode(body)

    # Build __device__ function using C-compatible parameter names
    func_name = f"eval_G_{func_idx}"
    params = ["double T"]
    for cname in site_frac_cnames:
        params.append(f"double {cname}")

    func_code = f"__device__ double {func_name}({', '.join(params)}) {{\n"
    func_code += body
    func_code += "\n}\n"

    return func_code, func_name, site_frac_cnames


def _find_matching_paren(code, start):
    """Find the matching closing parenthesis for an opening paren at 'start'."""
    depth = 0
    for i in range(start, len(code)):
        if code[i] == '(':
            depth += 1
        elif code[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_func_args(code, start, end):
    """Split function arguments at top-level commas between start and end."""
    depth = 0
    args = []
    current_start = start
    for i in range(start, end):
        if code[i] == '(':
            depth += 1
        elif code[i] == ')':
            depth -= 1
        elif code[i] == ',' and depth == 0:
            args.append(code[current_start:i].strip())
            current_start = i + 1
    args.append(code[current_start:end].strip())
    return args


def _replace_function_call(code, func_name, replacement_op):
    """Replace func_name(arg1, arg2) with ((arg1) op (arg2)), handling nested parens."""
    while True:
        idx = code.find(func_name + '(')
        if idx == -1:
            break
        # Make sure it's not part of a longer identifier
        if idx > 0 and (code[idx - 1].isalnum() or code[idx - 1] == '_'):
            # Part of a longer name, skip (shouldn't happen with And/Or)
            break
        paren_start = idx + len(func_name)
        paren_end = _find_matching_paren(code, paren_start)
        if paren_end == -1:
            break
        # Extract arguments
        args = _split_func_args(code, paren_start + 1, paren_end)
        if len(args) == 2:
            replacement = f'(({args[0]}) {replacement_op} ({args[1]}))'
        elif len(args) > 2:
            # Chain: And(a, b, c) -> ((a) && (b) && (c))
            parts = ' '.join(f'{replacement_op} ({a})' for a in args[1:])
            replacement = f'(({args[0]}) {parts})'
        else:
            replacement = args[0] if args else '0'
        code = code[:idx] + replacement + code[paren_end + 1:]
    return code


def _fix_ccode(code):
    """Fix symengine ccode output for CUDA compatibility."""
    # Handle And() -> && and Or() -> || using proper parenthesis matching
    code = _replace_function_call(code, 'And', '&&')
    code = _replace_function_call(code, 'Or', '||')

    # True -> 1, False -> 0 (as standalone tokens)
    code = re.sub(r'\bTrue\b', '1', code)
    code = re.sub(r'\bFalse\b', '0', code)

    # Fix potential issues with Piecewise not being converted
    code = _replace_function_call(code, 'Not', '!')

    # Handle remaining Piecewise if any
    if 'Piecewise(' in code:
        code = _convert_remaining_piecewise(code)

    return code


def _convert_remaining_piecewise(code):
    """Convert any remaining Piecewise expressions to C ternary operators."""
    while 'Piecewise(' in code:
        idx = code.find('Piecewise(')
        paren_start = idx + len('Piecewise')
        paren_end = _find_matching_paren(code, paren_start)
        if paren_end == -1:
            break
        inner = code[paren_start + 1:paren_end]
        # Parse Piecewise pairs: (expr, cond), (expr2, cond2), ...
        # Find top-level pairs by matching parentheses
        pairs = []
        i = 0
        while i < len(inner):
            if inner[i] == '(':
                pair_end = _find_matching_paren(inner, i)
                if pair_end == -1:
                    break
                pair_content = inner[i + 1:pair_end]
                # Split at the last top-level comma
                args = _split_func_args(pair_content, 0, len(pair_content))
                if len(args) >= 2:
                    expr = ', '.join(args[:-1])  # everything except last is the expression
                    cond = args[-1]
                    pairs.append((expr.strip(), cond.strip()))
                i = pair_end + 1
            else:
                i += 1

        # Convert to nested ternary: (cond1 ? expr1 : (cond2 ? expr2 : default))
        if len(pairs) >= 2:
            result = pairs[-1][0]  # last pair's expression (default, typically cond=True/1)
            for expr, cond in reversed(pairs[:-1]):
                result = f'(({cond}) ? ({expr}) : ({result}))'
            code = code[:idx] + result + code[paren_end + 1:]
        elif len(pairs) == 1:
            code = code[:idx] + pairs[0][0] + code[paren_end + 1:]
        else:
            break
    return code


def _classify_phase(dbf, phase_name, nonvacant):
    """
    Classify a phase for GPU kernel generation.

    Returns
    -------
    phase_type : str
        'simple' - single mixing sublattice with both nonvacant species
        'stoichiometric' - all site fractions fixed (no composition DOF)
        'va_solution' - composition varies via VA mixing, but site fracs
                        uniquely determined by composition
        'multi_sublattice' - multiple sublattices with non-VA mixing (needs inner optimization)
    fixed_x : float or None
        Mole fraction of nonvacant[1] if stoichiometric, else None
    """
    phase_obj = dbf.phases[phase_name]
    constituents = phase_obj.constituents
    sublattices = phase_obj.sublattices
    nonvacant_set = set(nonvacant)

    # Analyze each sublattice
    mixing_sublattices = 0  # sublattices with >1 nonvacant species
    va_mixing_sublattices = 0  # sublattices with 1 nonvacant species + VA
    for subl_constituents in constituents:
        species_names = {s.name for s in subl_constituents}
        active = species_names & nonvacant_set
        has_va = 'VA' in species_names
        if len(active) > 1:
            mixing_sublattices += 1
        elif len(active) == 1 and has_va:
            va_mixing_sublattices += 1

    if mixing_sublattices > 1:
        return 'multi_sublattice', None

    if mixing_sublattices == 1:
        return 'simple', None

    # No nonvacant-species mixing sublattices
    if va_mixing_sublattices > 0:
        # Has VA mixing → composition varies, but no ordering DOF
        return 'va_solution', None

    # Truly stoichiometric: every sublattice has exactly one non-VA species
    numerator = 0.0
    denominator = 0.0
    for subl_idx, subl_constituents in enumerate(constituents):
        species_names = {s.name for s in subl_constituents}
        n_sites = float(sublattices[subl_idx])
        active = species_names & nonvacant_set
        if active:
            denominator += n_sites
            if nonvacant[1] in active:
                numerator += n_sites

    if denominator == 0:
        return 'stoichiometric', 0.5

    return 'stoichiometric', numerator / denominator


def _get_va_solution_info(dbf, phase_name, nonvacant):
    """
    For a VA-solution phase, compute the composition formula.

    For a phase like (AL)_a : (FE)_b : (AL,VA)_c,
    composition x_{nonvacant[1]} = fixed_num / (fixed_denom + c * y_species)
    where y_species is the non-VA fraction in the VA-mixing sublattice.

    Returns dict with keys:
        fixed_num: float
        fixed_denom: float
        va_subl_idx: int - sublattice index of the VA mixing sublattice
        va_subl_sites: float - site count of the VA sublattice
        va_species: str - the non-VA species in the VA sublattice
        va_species_is_comp1: bool - True if the VA species is nonvacant[1]
    """
    phase_obj = dbf.phases[phase_name]
    constituents = phase_obj.constituents
    sublattices = phase_obj.sublattices
    nonvacant_set = set(nonvacant)

    fixed_num = 0.0  # sites contributing to nonvacant[1]
    fixed_denom = 0.0  # sites from fixed sublattices
    va_subl_idx = None
    va_subl_sites = None
    va_species = None

    for subl_idx, subl_constituents in enumerate(constituents):
        species_names = {s.name for s in subl_constituents}
        n_sites = float(sublattices[subl_idx])
        active = species_names & nonvacant_set
        has_va = 'VA' in species_names

        if len(active) == 1 and has_va:
            # VA-mixing sublattice
            va_subl_idx = subl_idx
            va_subl_sites = n_sites
            va_species = list(active)[0]
        elif active:
            # Fixed sublattice (single non-VA species)
            fixed_denom += n_sites
            if nonvacant[1] in active:
                fixed_num += n_sites

    va_species_is_comp1 = (va_species == nonvacant[1])

    return {
        'fixed_num': fixed_num,
        'fixed_denom': fixed_denom,
        'va_subl_idx': va_subl_idx,
        'va_subl_sites': va_subl_sites,
        'va_species': va_species,
        'va_species_is_comp1': va_species_is_comp1,
    }


def _get_multi_sublattice_info(dbf, phase_name, nonvacant):
    """
    For a multi-sublattice ordering phase, compute composition formula
    and identify the mixing sublattices.

    Returns dict with:
        mixing_subls: list of (subl_idx, n_sites) for sublattices with
                      multiple nonvacant species
        fixed_comp1_atoms: float - total contribution of fixed sublattices
                           to nonvacant[1] atom count
        total_non_va_sites: float - total non-VA site count
                            (sum of sites for sublattices with nonvacant species)
    """
    phase_obj = dbf.phases[phase_name]
    constituents = phase_obj.constituents
    sublattices = phase_obj.sublattices
    nonvacant_set = set(nonvacant)

    mixing_subls = []
    fixed_comp1_atoms = 0.0
    total_non_va_sites = 0.0

    for subl_idx, subl_constituents in enumerate(constituents):
        species_names = {s.name for s in subl_constituents}
        active = species_names & nonvacant_set
        n_sites = float(sublattices[subl_idx])

        if len(active) > 1:
            # Mixing sublattice
            mixing_subls.append((subl_idx, n_sites))
            total_non_va_sites += n_sites
        elif active:
            # Fixed sublattice (single nonvacant species)
            total_non_va_sites += n_sites
            if nonvacant[1] in active:
                fixed_comp1_atoms += n_sites
        # VA-only sublattices don't contribute to composition

    return {
        'mixing_subls': mixing_subls,
        'fixed_comp1_atoms': fixed_comp1_atoms,
        'total_non_va_sites': total_non_va_sites,
    }


def _parse_sublattice_index(sf_cname, phase_name):
    """
    Parse the sublattice index from a ccode site fraction name.

    E.g., 'B2_BCC0FE' for phase 'B2_BCC' -> sublattice 0, species 'FE'
          'ALMG_GAMMA1AL' for phase 'ALMG_GAMMA' -> sublattice 1, species 'AL'
    """
    # The ccode name is the phase name (with underscores removed in some cases)
    # followed by the sublattice index digit(s) then species name.
    # Try to match by stripping the phase prefix.
    # ccode converts Y(PHASE,idx,SPECIES) to PHASE + idx + SPECIES (no separators)
    # but underscores in phase names are preserved.
    prefix = phase_name.replace('_', '')
    name_upper = sf_cname.upper()
    prefix_upper = prefix.upper()

    if name_upper.startswith(prefix_upper):
        rest = sf_cname[len(prefix):]
    else:
        # Try with underscores intact
        if name_upper.startswith(phase_name.upper()):
            rest = sf_cname[len(phase_name):]
        else:
            return -1, sf_cname

    # rest should be like "0FE", "1AL", "2VA"
    digits = ''
    for c in rest:
        if c.isdigit():
            digits += c
        else:
            break

    if digits:
        subl_idx = int(digits)
        species = rest[len(digits):]
        return subl_idx, species
    return -1, rest


def _build_kernel_source(models, phases, nonvacant, dbf=None):
    """
    Build the complete CUDA kernel source for all phases.

    The kernel evaluates G for all (T, x) pairs and finds the stable
    phase at each (T, mu) point. Stoichiometric phases (line compounds)
    are handled as fixed-composition phases. Multi-sublattice phases
    with 2 mixing sublattices use a 2D inner optimization loop.

    Returns
    -------
    kernel_source : str
    skipped_phases : list of str
        Phases that were skipped due to multi-sublattice mixing
    """
    # Generate G functions for each phase, classifying each
    g_functions = []
    g_func_names = []
    all_sf_names = []
    phase_info = []  # (phase_type, fixed_x)
    skipped_phases = []

    for idx, phase_name in enumerate(phases):
        if dbf is not None:
            phase_type, fixed_x = _classify_phase(dbf, phase_name, nonvacant)
        else:
            phase_type, fixed_x = 'simple', None

        if phase_type == 'multi_sublattice':
            ms_info = _get_multi_sublattice_info(dbf, phase_name, nonvacant)
            if len(ms_info['mixing_subls']) != 2:
                skipped_phases.append(phase_name)
                continue

        model = models[phase_name]
        func_code, func_name, sf_names = _generate_G_device_function(
            model, len(g_functions))
        g_functions.append(func_code)
        g_func_names.append(func_name)
        all_sf_names.append(sf_names)
        phase_info.append((phase_type, fixed_x, phase_name))

    # Build the main kernel — only include non-skipped phases
    n_kernel_phases = len(phase_info)
    # Map from kernel phase index to original phase name
    kernel_phase_names = [pi[2] for pi in phase_info]

    # Generate per-phase G evaluation + inner loop code
    phase_blocks = []
    for idx, (func_name, sf_names) in enumerate(zip(g_func_names, all_sf_names)):
        phase_type, fixed_x, pname = phase_info[idx]

        if phase_type == 'stoichiometric':
            # All site fracs = 1.0
            sf_args = ["1.0"] * len(sf_names)
            args_str = ", ".join(sf_args)
            block = f"""
        // Phase {idx}: {pname} (stoichiometric, x_fixed={fixed_x:.10f})
        {{
            double G = {func_name}(T, {args_str});
            double fixed_x = {fixed_x:.10f};
            double omega = G - delta_mu * fixed_x;
            int out_idx = {idx} * n_T * n_mu + idx_flat;
            omega_min_out[out_idx] = omega;
            x_opt_out[out_idx] = fixed_x;
            if (omega < best_omega_global) {{
                best_omega_global = omega;
                best_phase_global = {idx};
            }}
        }}"""

        elif phase_type == 'va_solution':
            # Sweep over the free VA-mixing site fraction
            va_info = _get_va_solution_info(dbf, pname, nonvacant)
            va_si = va_info['va_subl_idx']
            va_sites = va_info['va_subl_sites']
            fixed_num = va_info['fixed_num']
            fixed_denom = va_info['fixed_denom']
            va_sp = va_info['va_species']
            va_is_comp1 = va_info['va_species_is_comp1']

            # Build site fraction args with the sweep variable
            sf_args = []
            for sf_name in sf_names:
                # Parse sublattice index from ccode name: PHASE{SUBL_IDX}{SPECIES}
                # e.g., AL13FE42AL -> sublattice 2, species AL
                sf_upper = sf_name.upper()
                if 'VA' in sf_upper:
                    sf_args.append("(1.0 - y_va_free)")
                elif sf_upper.endswith(va_sp.upper()):
                    # Could be the VA sublattice species or a fixed sublattice
                    # Check if this is the VA-mixing sublattice by looking for the
                    # sublattice index in the name
                    # The ccode name format is PHASENAME + SUBL_IDX + SPECIES
                    # We need to check if SUBL_IDX matches va_si
                    phase_prefix = pname.replace('_', '').upper()
                    # Strip the phase prefix and species suffix to get subl idx
                    stripped = sf_upper
                    if stripped.startswith(phase_prefix):
                        stripped = stripped[len(phase_prefix):]
                    # Now stripped should be like "0AL", "1FE", "2AL"
                    subl_char = ''
                    for c in stripped:
                        if c.isdigit():
                            subl_char += c
                        else:
                            break
                    try:
                        subl_idx = int(subl_char) if subl_char else -1
                    except ValueError:
                        subl_idx = -1

                    if subl_idx == va_si:
                        sf_args.append("y_va_free")
                    else:
                        sf_args.append("1.0")  # fixed sublattice
                else:
                    sf_args.append("1.0")  # fixed sublattice species

            args_str = ", ".join(sf_args)

            # Composition formula: x_comp1 = (fixed_num + va_contrib) / (fixed_denom + va_sites * y_va_free)
            if va_is_comp1:
                x_formula = (f"({fixed_num} + {va_sites} * y_va_free) / "
                             f"({fixed_denom} + {va_sites} * y_va_free)")
            else:
                x_formula = (f"{fixed_num} / "
                             f"({fixed_denom} + {va_sites} * y_va_free)")

            block = f"""
        // Phase {idx}: {pname} (va_solution, VA sublattice {va_si})
        {{
            double best_omega = 1e30;
            double best_x = 0.5;
            double dy_va = (1.0 - 2.0 * MIN_SF) / (double)(n_x - 1);
            for (int yi = 0; yi < n_x; yi++) {{
                double y_va_free = MIN_SF + yi * dy_va;
                double x_comp = {x_formula};
                double G = {func_name}(T, {args_str});
                double omega = G - delta_mu * x_comp;
                if (omega < best_omega) {{
                    best_omega = omega;
                    best_x = x_comp;
                }}
            }}
            int out_idx = {idx} * n_T * n_mu + idx_flat;
            omega_min_out[out_idx] = best_omega;
            x_opt_out[out_idx] = best_x;
            if (best_omega < best_omega_global) {{
                best_omega_global = best_omega;
                best_phase_global = {idx};
            }}
        }}"""

        elif phase_type == 'multi_sublattice':
            # 2D inner loop over ordering DOFs in the two mixing sublattices
            ms_info = _get_multi_sublattice_info(dbf, pname, nonvacant)
            mixing_subls = ms_info['mixing_subls']
            subl_a_idx, subl_a_sites = mixing_subls[0]
            subl_b_idx, subl_b_sites = mixing_subls[1]
            fixed_comp1 = ms_info['fixed_comp1_atoms']
            total_sites = ms_info['total_non_va_sites']

            # Build site fraction args mapped to y_a, y_b
            sf_args = []
            for sf_name in sf_names:
                subl_idx, species = _parse_sublattice_index(sf_name, pname)
                if species.upper() == 'VA':
                    sf_args.append("1.0")
                elif subl_idx == subl_a_idx:
                    if species.upper() == nonvacant[1].upper():
                        sf_args.append("y_a")
                    else:
                        sf_args.append("(1.0 - y_a)")
                elif subl_idx == subl_b_idx:
                    if species.upper() == nonvacant[1].upper():
                        sf_args.append("y_b")
                    else:
                        sf_args.append("(1.0 - y_b)")
                else:
                    sf_args.append("1.0")  # fixed sublattice

            args_str = ", ".join(sf_args)
            x_formula = (f"({fixed_comp1} + {subl_a_sites} * y_a + "
                         f"{subl_b_sites} * y_b) / {total_sites}")

            block = f"""
        // Phase {idx}: {pname} (multi_sublattice, sublattices {subl_a_idx} and {subl_b_idx})
        {{
            double best_omega = 1e30;
            double best_x = 0.5;
            double dy_ord = (1.0 - 2.0 * MIN_SF) / (double)(n_order - 1);
            for (int ya_i = 0; ya_i < n_order; ya_i++) {{
                double y_a = MIN_SF + ya_i * dy_ord;
                for (int yb_i = 0; yb_i < n_order; yb_i++) {{
                    double y_b = MIN_SF + yb_i * dy_ord;
                    double x_comp = {x_formula};
                    double G = {func_name}(T, {args_str});
                    double omega = G - delta_mu * x_comp;
                    if (omega < best_omega) {{
                        best_omega = omega;
                        best_x = x_comp;
                    }}
                }}
            }}
            int out_idx = {idx} * n_T * n_mu + idx_flat;
            omega_min_out[out_idx] = best_omega;
            x_opt_out[out_idx] = best_x;
            if (best_omega < best_omega_global) {{
                best_omega_global = best_omega;
                best_phase_global = {idx};
            }}
        }}"""

        else:
            # Simple solution phase: sweep over composition
            sf_args = []
            for sf_name in sf_names:
                if 'VA' in sf_name.upper():
                    sf_args.append("1.0")
                elif sf_name.endswith(nonvacant[1]) or sf_name.endswith(nonvacant[1].upper()):
                    sf_args.append("x_comp")
                elif sf_name.endswith(nonvacant[0]) or sf_name.endswith(nonvacant[0].upper()):
                    sf_args.append("(1.0 - x_comp)")
                else:
                    matched = False
                    for comp_idx, comp_name in enumerate(nonvacant):
                        if comp_name in sf_name:
                            sf_args.append("x_comp" if comp_idx == 1 else "(1.0 - x_comp)")
                            matched = True
                            break
                    if not matched:
                        sf_args.append("0.5")

            args_str = ", ".join(sf_args)
            block = f"""
        // Phase {idx}: {pname} (solution)
        {{
            double best_omega = 1e30;
            double best_x = 0.5;
            for (int xi = 0; xi < n_x; xi++) {{
                double x_comp = MIN_SF + xi * dx;
                double G = {func_name}(T, {args_str});
                double omega = G - delta_mu * x_comp;
                if (omega < best_omega) {{
                    best_omega = omega;
                    best_x = x_comp;
                }}
            }}
            int out_idx = {idx} * n_T * n_mu + idx_flat;
            omega_min_out[out_idx] = best_omega;
            x_opt_out[out_idx] = best_x;
            if (best_omega < best_omega_global) {{
                best_omega_global = best_omega;
                best_phase_global = {idx};
            }}
        }}"""

        phase_blocks.append(block)

    phase_code = "\n".join(phase_blocks)

    kernel_source = f"""
#define MIN_SF {MIN_SITE_FRACTION}
#define N_PHASES {n_kernel_phases}

// ============================================================
// Auto-generated Gibbs energy device functions
// ============================================================
{"".join(g_functions)}

// ============================================================
// Main kernel: for each (T, mu) pair, find the stable phase
// by minimizing Omega = G(x) - mu * x over all compositions
// ============================================================
extern "C" __global__ void grand_potential_kernel(
    int* stable_phase_out,     // shape: (n_T * n_mu)
    double* omega_min_out,     // shape: (n_phases * n_T * n_mu)
    double* x_opt_out,         // shape: (n_phases * n_T * n_mu)
    const double* T_arr,       // shape: (n_T)
    const double* mu_arr,      // shape: (n_mu)
    int n_T,
    int n_mu,
    int n_x,                   // composition grid resolution
    int n_order                // ordering grid resolution per dimension
) {{
    int idx_flat = threadIdx.x + blockIdx.x * blockDim.x;
    int total = n_T * n_mu;
    if (idx_flat >= total) return;

    int t_idx = idx_flat / n_mu;
    int mu_idx = idx_flat % n_mu;

    double T = T_arr[t_idx];
    double delta_mu = mu_arr[mu_idx];

    double dx = (1.0 - 2.0 * MIN_SF) / (double)(n_x - 1);

    double best_omega_global = 1e30;
    int best_phase_global = 0;
{phase_code}

    stable_phase_out[idx_flat] = best_phase_global;
}}
"""
    return kernel_source, kernel_phase_names, skipped_phases


def compute_phase_diagram_gpu(dbf, comps, phases, conditions,
                              mu_resolution=500, x_resolution=2000,
                              order_resolution=80, verbose=False):
    """
    Compute phase diagram using GPU-accelerated grand potential approach.

    Parameters
    ----------
    dbf : Database
    comps : list of str
    phases : list of str
    conditions : dict
        Must contain v.T (scalar or array), v.P
    mu_resolution : int
        Number of chemical potential grid points
    x_resolution : int
        Number of composition grid points for inner minimization
    verbose : bool

    Returns
    -------
    result : dict
    """
    if not HAS_GPU:
        raise RuntimeError("CuPy not available. Install CuPy for GPU support.")

    t_total_start = time.time()

    comps = sorted(set(comps) | {'VA'})
    nonvacant = sorted([c for c in comps if c != 'VA'])
    num_comps = len(nonvacant)

    if num_comps != 2:
        raise NotImplementedError("GPU grand potential only supports binary systems currently.")

    # Build models
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

    if verbose:
        print(f"Phases: {phases}")
        print(f"Components: {nonvacant}")

    # Build phase records for mu range estimation (CPU-side)
    state_variables = sorted(
        [sv for sv in models[phases[0]].state_variables], key=str
    )
    prf = PhaseRecordFactory(dbf, comps, state_variables, models)
    optimizers = {}
    phase_records = {}
    for phase in phases:
        pr = prf[phase]
        phase_records[phase] = pr
        optimizers[phase] = SublatticeOptimizer(pr, models[phase])

    # Generate and compile CUDA kernel
    t_compile_start = time.time()
    kernel_source, gpu_phases, skipped = _build_kernel_source(
        models, phases, nonvacant, dbf=dbf)

    if skipped and verbose:
        print(f"\nSkipped multi-sublattice phases: {skipped}")
    if verbose:
        print(f"GPU phases: {gpu_phases}")
        print(f"Generated kernel: {len(kernel_source)} chars")

    # Hash for caching
    kernel_hash = hashlib.md5(kernel_source.encode()).hexdigest()[:12]

    # Compile
    try:
        module = cp.RawModule(code=kernel_source, options=('--std=c++14',))
        kernel = module.get_function('grand_potential_kernel')
    except Exception as e:
        # Save kernel source for debugging
        debug_path = f'/tmp/gp_kernel_{kernel_hash}.cu'
        with open(debug_path, 'w') as f:
            f.write(kernel_source)
        print(f"Kernel compilation failed. Source saved to {debug_path}")
        raise

    t_compile = time.time() - t_compile_start
    if verbose:
        print(f"Kernel compilation: {t_compile:.3f}s")

    # Extract temperature values
    T_cond = conditions.get(v.T, 300)
    if isinstance(T_cond, (int, float)):
        T_values = np.array([T_cond], dtype=np.float64)
    elif hasattr(T_cond, '__iter__'):
        T_values = np.array(list(T_cond), dtype=np.float64)
    else:
        T_values = np.atleast_1d(np.array(T_cond, dtype=np.float64))

    P = float(conditions.get(v.P, 101325))
    n_T = len(T_values)

    # Estimate mu range (using CPU-side evaluation at a representative temperature)
    T_mid = T_values[len(T_values) // 2]
    mu_min, mu_max = _estimate_mu_range(optimizers, phase_records,
                                        T_mid, P, nonvacant)
    # Expand range for temperature variation
    if n_T > 1:
        mu_min_lo, mu_max_lo = _estimate_mu_range(
            optimizers, phase_records, T_values[0], P, nonvacant)
        mu_min_hi, mu_max_hi = _estimate_mu_range(
            optimizers, phase_records, T_values[-1], P, nonvacant)
        mu_min = min(mu_min, mu_min_lo, mu_min_hi)
        mu_max = max(mu_max, mu_max_lo, mu_max_hi)

    mu_grid = np.linspace(mu_min, mu_max, mu_resolution, dtype=np.float64)

    if verbose:
        print(f"\nmu range: [{mu_min:.1f}, {mu_max:.1f}] J/mol")
        print(f"Grid: {n_T} temps x {mu_resolution} mu x {x_resolution} x-points")
        print(f"Threads: {n_T * mu_resolution}")

    # Transfer data to GPU
    t_compute_start = time.time()
    d_T = cp.asarray(T_values)
    d_mu = cp.asarray(mu_grid)

    # Allocate output arrays (sized for GPU phases only)
    n_gpu_phases = len(gpu_phases)
    d_stable = cp.zeros(n_T * mu_resolution, dtype=cp.int32)
    d_omega = cp.zeros(n_gpu_phases * n_T * mu_resolution, dtype=cp.float64)
    d_x_opt = cp.zeros(n_gpu_phases * n_T * mu_resolution, dtype=cp.float64)

    # Launch kernel
    threads_per_block = 256
    total_threads = n_T * mu_resolution
    blocks = (total_threads + threads_per_block - 1) // threads_per_block

    kernel(
        (blocks,), (threads_per_block,),
        (d_stable, d_omega, d_x_opt, d_T, d_mu,
         np.int32(n_T), np.int32(mu_resolution), np.int32(x_resolution),
         np.int32(order_resolution))
    )
    cp.cuda.Device().synchronize()

    t_compute = time.time() - t_compute_start

    # Transfer results back to CPU
    stable_phase_idx = cp.asnumpy(d_stable).reshape(n_T, mu_resolution)
    omega_all = cp.asnumpy(d_omega).reshape(n_gpu_phases, n_T, mu_resolution)
    x_opt_all = cp.asnumpy(d_x_opt).reshape(n_gpu_phases, n_T, mu_resolution)

    if verbose:
        print(f"GPU compute: {t_compute*1000:.1f} ms")
        total_evals = n_T * mu_resolution * x_resolution * n_gpu_phases
        print(f"Total G evaluations: {total_evals:,}")
        if t_compute > 0:
            print(f"Throughput: {total_evals / t_compute / 1e9:.2f} billion G-evals/s")

    # Find phase boundaries for each temperature
    # Map kernel phase indices back to phase names
    results_per_T = []
    mu_vector_func = lambda dm: np.array([0.0, dm])

    for t_idx in range(n_T):
        T = T_values[t_idx]
        sp_idx = stable_phase_idx[t_idx]

        # Find boundary crossings
        boundaries = []
        for i in range(1, mu_resolution):
            if sp_idx[i] != sp_idx[i - 1]:
                idx_a = sp_idx[i - 1]
                idx_b = sp_idx[i]
                phase_a = gpu_phases[idx_a]
                phase_b = gpu_phases[idx_b]

                # Refine using CPU-side bisection
                mu_star, x_a, x_b = _refine_boundary(
                    optimizers[phase_a], optimizers[phase_b],
                    T, P, mu_grid[i - 1], mu_grid[i],
                    mu_vector_func
                )
                # Skip boundaries with negligible composition gap
                # (e.g., B2_BCC/BCC_A2 oscillations in disordered region)
                if abs(x_a[1] - x_b[1]) < 0.005:
                    continue
                boundaries.append({
                    'mu': mu_star,
                    'phase_low_mu': phase_a,
                    'phase_high_mu': phase_b,
                    'x_phase_a': x_a.copy(),
                    'x_phase_b': x_b.copy(),
                })

        # Build per-temperature result (compatible with CPU version)
        stable_names = [gpu_phases[sp_idx[i]] for i in range(mu_resolution)]
        stable_x = np.zeros((mu_resolution, 2))
        for i in range(mu_resolution):
            stable_x[i, 1] = x_opt_all[sp_idx[i], t_idx, i]
            stable_x[i, 0] = 1.0 - stable_x[i, 1]

        results_per_T.append({
            'T': T,
            'P': P,
            'mu_grid': mu_grid,
            'omega': omega_all[:, t_idx, :],
            'x_at_mu': x_opt_all[:, t_idx, :],
            'stable_phase_idx': sp_idx,
            'stable_phase_names': stable_names,
            'stable_x': stable_x,
            'phase_boundaries': boundaries,
        })

    t_total = time.time() - t_total_start

    return {
        'components': nonvacant,
        'phases': gpu_phases,
        'skipped_phases': skipped,
        'temperatures': T_values,
        'results_per_T': results_per_T,
        'elapsed_time': t_total,
        'compile_time': t_compile,
        'compute_time': t_compute,
    }
