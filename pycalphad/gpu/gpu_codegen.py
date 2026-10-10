# gpu_codegen.py
# 
# GPU C code generation functions for pycalphad GPU acceleration
# Contains all string building and code generation functionality

import os
import re
import math
import tempfile
import subprocess
import time
from typing import List, Set, Optional, Dict, Tuple
from pycalphad.core.workspace import Workspace
from pycalphad.model import Model

# Import SymEngine for improved code generation
import symengine as se
from symengine import cse
from symengine.lib.symengine_wrapper import ccode

# Global constants for C code generation
_C_PYCALPHAD_VARIABLE_PREFIX_NOTEBOOK = "x"  # Variable array name in C device functions

# Validation constants
MAX_EXPRESSION_LENGTH = 10000  # Maximum allowed C expression length
MAX_VARIABLE_INDEX = 1000     # Maximum allowed variable index
DANGEROUS_C_FUNCTIONS = {
    'system', 'exec', 'popen', 'fork', 'malloc', 'free', 'strcpy', 'strcat'
}
ALLOWED_MATH_FUNCTIONS = {
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'exp', 'exp2', 'exp10', 'expm1', 'log', 'log2', 'log10', 'log1p',
    'pow', 'sqrt', 'cbrt', 'hypot', 'fabs', 'abs',
    'floor', 'ceil', 'round', 'trunc', 'fmod', 'remainder',
    'copysign', 'nextafter', 'ldexp', 'frexp', 'modf',
    'isnan', 'isinf', 'isfinite', 'isnormal', 'signbit',
    'fmax', 'fmin', 'fdim', 'fma'
}


def _read_gpu_header(filename):
    """Helper function to read header files from the gpu directory and clean them for inlining."""
    current_dir = os.path.dirname(__file__)
    try:
        # Try to read from the same directory as this file (gpu folder)
        with open(os.path.join(current_dir, filename), "r") as f:
            content = f.read()
    except FileNotFoundError:
        # Fallback if running from a different context
        proj_base_dir = os.path.join(current_dir, "..")
        with open(os.path.join(proj_base_dir, "gpu", filename), "r") as f:
            content = f.read()
    
    # Clean the content for inlining:
    # 1. Remove include statements for other .h files (since we're inlining everything)
    # 2. Keep standard library includes but comment them out (since we include them at the top)
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Remove #pragma once directives to avoid warnings in main compilation unit
        if stripped.startswith('#pragma once'):
            cleaned_lines.append(f"// {line}  // Removed for inlining")
        # Skip include statements for local .h files, but keep standard library includes
        elif stripped.startswith('#include "') and '.h"' in stripped:
            # Skip local header includes like #include "phase_rec.h" (with or without comments)
            cleaned_lines.append(f"// {line}  // Removed for inlining")
        elif stripped.startswith('#include <') and stripped.endswith('.h>'):
            # Keep standard library includes but comment them out since we include them at the top
            cleaned_lines.append(f"// {line}  // Already included at top level")
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _get_c_define(define_name, default_val=64):
    """
    Helper to get MAX_... defines. These should match the C header definitions.
    """
    # These values should match the definitions in minimizer.h
    constants = {
        "MAX_COMPONENTS": 32,
        "MAX_PHASES": 64, 
        "MAX_STATEVARS": 8,
        "MAX_DOF_PER_PHASE": 64,
        "MAX_INTERNAL_CONSTRAINTS": 32,
        "MAX_FIXED_MOLE_FRACTION_CONDITIONS": 32,  # Same as MAX_COMPONENTS
        "MAX_GRID_POINTS": 10000,  # Reasonable default for grid
        "MIN_PHASE_FRACTION": 1e-6,  # From constants.py
    }
    return constants.get(define_name, default_val)


def compute_dynamic_kernel_sizes(wks_obj: Workspace) -> Dict[str, int]:
    """
    Compute kernel size parameters from the workspace instead of hard-coded
    MAX_* values: sizes derive from the phase records/models and are passed
    to the kernel as -D compiler defines.
    
    Args:
        wks_obj: Workspace containing phase records and models
        
    Returns:
        Dictionary mapping define names to computed values
    """
    # Compute actual requirements from the workspace
    actual_components = len(wks_obj.components)
    actual_phases = len(wks_obj.phases)
    
    # Find maximum DOF and internal-constraint count per phase across all models.
    # Internal constraints are PER-PHASE (one per sublattice, e.g. a 4-sublattice
    # ordering model with a VA interstitial sublattice has FIVE) — sizing them
    # from the number of phases under-allocated for few-phase/many-sublattice
    # systems and caused out-of-bounds writes (segfault on alnifcc4sl FCC_L12).
    max_dof_per_phase = 0
    max_internal_cons = 0
    for phase_name in wks_obj.phases:
        model = wks_obj.models[phase_name]
        phase_dof = len(model.site_fractions)
        max_dof_per_phase = max(max_dof_per_phase, phase_dof)
        max_internal_cons = max(max_internal_cons, len(model.get_internal_constraints()))
    
    # Get state variables count
    if hasattr(wks_obj, 'phase_record_factory') and hasattr(wks_obj.phase_record_factory, 'state_variables'):
        actual_statevars = len(wks_obj.phase_record_factory.state_variables)
    else:
        actual_statevars = 3  # Default: T, P, N
    
    # Add some padding for safety but avoid excessive over-allocation
    padding_factor = 1.2  # 20% padding
    safety_minimum = 4    # At least 4 for any dimension
    
    # MAX_PHASES sizes COMPOSITION SET slots, not phase names: the hull
    # starting point can seed up to one compset per nonvacant component
    # (all of the same phase for single-phase problems, e.g. a 5-component
    # FCC miscibility gap starts from 5 FCC vertices), so capacity must
    # scale with components as well as phases or the starting point is
    # silently truncated.
    nonvacant_comps = sum(1 for c in wks_obj.components if getattr(c, 'name', str(c)) != 'VA')
    compset_capacity = max(actual_phases, nonvacant_comps + 1)
    computed_sizes = {
        "MAX_COMPONENTS": max(safety_minimum, int(actual_components * padding_factor)),
        "MAX_PHASES": max(safety_minimum, int(compset_capacity * padding_factor)),
        "MAX_STATEVARS": max(safety_minimum, int(actual_statevars * padding_factor)),
        "MAX_DOF_PER_PHASE": max(safety_minimum, int(max_dof_per_phase * padding_factor)),
        "MAX_INTERNAL_CONSTRAINTS": max(safety_minimum, int(max_internal_cons * padding_factor) + 1),
        "MAX_FIXED_MOLE_FRACTION_CONDITIONS": max(safety_minimum, int(actual_components * padding_factor)),
        "MAX_GRID_POINTS": 10000,  # Keep reasonable default for grid
        "MIN_PHASE_FRACTION": 1e-6,  # Keep constant
        # Fit parameters occupy trailing dof slots (runtime inputs; see
        # _fit_parameter_symbols). Sized exactly: parameter count changes the
        # generated code, and the cache key includes dynamic sizes.
        "MAX_PARAMS": len(_fit_parameter_symbols(wks_obj)),
        # Phase-local conditions (conditions with a phase name): total count
        # system-wide. 0 when absent -> all dependent sizes unchanged.
        "MAX_PHASE_LOCAL_CONDITIONS": _max_phase_local_conditions(wks_obj),
    }

    # The equilibrium-matrix work-array strides MUST be passed as -D
    # defines. The kernel has #ifndef fallbacks (1000/50/50) for these; if they
    # differ from the Python-side allocation strides, threads index past the end
    # of the allocations and silently corrupt other buffers (this caused
    # nondeterministic results for batches beyond ~150 conditions).
    eq_rows = 2 * computed_sizes["MAX_PHASES"] + computed_sizes["MAX_COMPONENTS"] + 1
    eq_cols = computed_sizes["MAX_COMPONENTS"] + computed_sizes["MAX_PHASES"] + computed_sizes["MAX_STATEVARS"]
    computed_sizes["MAX_EQ_MATRIX_ROWS"] = eq_rows
    computed_sizes["MAX_EQ_SOLN_LEN"] = eq_cols
    computed_sizes["MAX_EQ_MATRIX_SIZE"] = eq_rows * eq_cols

    # Per-thread SystemState slot size in doubles. Mirrors the SystemState /
    # CompsetState / CompositionSet layouts in minimizer.h with a 1.3x margin;
    # a static_assert in eqsolver.h fails the build if the struct outgrows it.
    # (The old fixed 50000 was TOO SMALL for 21-phase systems — the in-kernel
    # memset spilled into the next thread's slot — and 10x too big for small
    # systems, wasting memory locality.)
    S = computed_sizes["MAX_STATEVARS"]
    D = computed_sizes["MAX_DOF_PER_PHASE"]
    C = computed_sizes["MAX_COMPONENTS"]
    I = computed_sizes["MAX_INTERNAL_CONSTRAINTS"]
    P = computed_sizes["MAX_PHASES"]
    csst_doubles = (2 * (S + D) + (S + D) ** 2 + C + C * (S + D) + 2 * (D + I) ** 2
                    + 3 * D + D * S + C * D + I + (S + D) + I * (S + D) + 32)
    compset_doubles = 4 + (S + D) + C + 12
    state_doubles = P * (csst_doubles + compset_doubles + 8) + 8 * C + 4 * S + 128
    computed_sizes["SYSTEM_STATE_SIZE"] = int(state_doubles * 1.3)

    # Phases whose moles are non-linear in the site fractions (two-sublattice
    # ionic liquids) add a moles-Hessian term to the phase matrix; the solver
    # compiles that term (and its buffer) in only when some phase needs it.
    if any(_formulamole_hessian_exprs(wks_obj.models[phase_name], wks_obj) is not None
           for phase_name in wks_obj.phases):
        computed_sizes["PYCGPU_FORMULAMOLE_HESS"] = 1

    return computed_sizes


# --- Validation Functions ---

class CodeValidationError(Exception):
    """Raised when generated C code fails validation."""
    pass


def validate_c_expression(expression: str, context: str = "") -> List[str]:
    """
    Validate a C expression for safety and correctness.
    
    Args:
        expression: The C expression to validate
        context: Additional context for error messages
        
    Returns:
        List of validation warnings (empty if no issues)
        
    Raises:
        CodeValidationError: If critical validation errors are found
    """
    warnings = []
    
    # Check expression length
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CodeValidationError(f"Expression too long ({len(expression)} > {MAX_EXPRESSION_LENGTH}): {context}")
    
    # Check for dangerous functions
    for dangerous_func in DANGEROUS_C_FUNCTIONS:
        if dangerous_func in expression:
            raise CodeValidationError(f"Dangerous function '{dangerous_func}' found in expression: {context}")
    
    # Check for buffer overflow patterns
    if re.search(r'\[\s*\d{4,}\s*\]', expression):  # Array access with large index
        warnings.append(f"Large array index detected, check bounds: {context}")
    
    # Check for division by potentially zero values
    div_patterns = [r'/\s*0\s*[;\)]', r'/\s*0\.0+\s*[;\)]']
    for pattern in div_patterns:
        if re.search(pattern, expression):
            warnings.append(f"Division by zero detected: {context}")
    
    # Check for unbalanced parentheses
    paren_count = expression.count('(') - expression.count(')')
    if paren_count != 0:
        raise CodeValidationError(f"Unbalanced parentheses (diff: {paren_count}): {context}")
    
    # Check for unbalanced brackets
    bracket_count = expression.count('[') - expression.count(']')
    if bracket_count != 0:
        raise CodeValidationError(f"Unbalanced brackets (diff: {bracket_count}): {context}")
    
    # Check variable indices
    var_indices = re.findall(r'x\[(\d+)\]', expression)
    for idx_str in var_indices:
        idx = int(idx_str)
        if idx > MAX_VARIABLE_INDEX:
            raise CodeValidationError(f"Variable index too large: x[{idx}] > {MAX_VARIABLE_INDEX}: {context}")
    
    # Check for valid C identifiers in function calls
    func_calls = re.findall(r'(\w+)\s*\(', expression)
    for func_name in func_calls:
        if func_name not in ALLOWED_MATH_FUNCTIONS and not func_name.startswith('pycgpu_'):
            warnings.append(f"Unknown function '{func_name}' in expression: {context}")
    
    # Check for potential numerical overflow/underflow
    large_numbers = re.findall(r'\b\d*\.?\d+[eE][+-]?\d+\b', expression)
    for num_str in large_numbers:
        try:
            num = float(num_str)
            if abs(num) > 1e100:
                warnings.append(f"Very large number {num_str} may cause overflow: {context}")
            elif 0 < abs(num) < 1e-100:
                warnings.append(f"Very small number {num_str} may cause underflow: {context}")
        except ValueError:
            warnings.append(f"Invalid number format {num_str}: {context}")
    
    return warnings


def validate_variable_indices(model_obj: Model, wks_obj: Workspace) -> Tuple[int, List[str]]:
    """
    Validate that variable indices will be within bounds.
    
    Returns:
        Tuple of (max_index, warnings)
    """
    warnings = []
    all_syms = notebook_get_all_syms_for_model(model_obj, wks_obj)
    max_index = len(all_syms) - 1
    
    max_statevars = _get_c_define("MAX_STATEVARS")
    max_dof = _get_c_define("MAX_DOF_PER_PHASE")
    max_total_vars = max_statevars + max_dof + len(_fit_parameter_symbols(wks_obj))
    
    if max_index >= max_total_vars:
        raise CodeValidationError(
            f"Model requires {max_index + 1} variables but MAX limit is {max_total_vars} "
            f"(MAX_STATEVARS={max_statevars} + MAX_DOF_PER_PHASE={max_dof})"
        )
    
    if max_index > max_total_vars * 0.8:  # Warning if using >80% of available space
        warnings.append(f"Using {max_index + 1}/{max_total_vars} available variable slots")
    
    return max_index, warnings


def validate_model_properties(model_obj: Model) -> List[str]:
    """
    Validate that all required model properties exist and are accessible.
    
    Returns:
        List of validation warnings
    """
    warnings = []
    required_properties = ['GM', 'G']
    optional_properties = ['nonvacant_elements', 'site_fractions']
    
    # Check required properties
    for prop in required_properties:
        if not hasattr(model_obj, prop):
            raise CodeValidationError(f"Model missing required property: {prop}")
        
        prop_value = getattr(model_obj, prop)
        if prop_value is None:
            raise CodeValidationError(f"Model property '{prop}' is None")
    
    # Check optional properties
    for prop in optional_properties:
        if not hasattr(model_obj, prop):
            warnings.append(f"Model missing optional property: {prop}")
        elif getattr(model_obj, prop) is None:
            warnings.append(f"Model property '{prop}' is None")
    
    # Check that we can access site fractions
    try:
        site_fracs = model_obj.site_fractions
        if len(site_fracs) == 0:
            warnings.append("Model has no site fractions")
        elif len(site_fracs) > _get_c_define("MAX_DOF_PER_PHASE"):
            raise CodeValidationError(
                f"Model has {len(site_fracs)} site fractions > MAX_DOF_PER_PHASE "
                f"({_get_c_define('MAX_DOF_PER_PHASE')})"
            )
    except Exception as e:
        warnings.append(f"Error accessing site fractions: {e}")
    
    # Check nonvacant elements
    try:
        nonvacant = model_obj.nonvacant_elements
        if len(nonvacant) > _get_c_define("MAX_COMPONENTS"):
            raise CodeValidationError(
                f"Model has {len(nonvacant)} nonvacant elements > MAX_COMPONENTS "
                f"({_get_c_define('MAX_COMPONENTS')})"
            )
    except Exception as e:
        warnings.append(f"Error accessing nonvacant elements: {e}")
    
    return warnings


def validate_generated_c_code(c_code: str, function_name: str = "") -> List[str]:
    """
    Validate generated C code for basic syntax and safety.
    
    Args:
        c_code: The C code to validate
        function_name: Name of the function for context
        
    Returns:
        List of validation warnings
    """
    warnings = []
    context = f"function {function_name}" if function_name else "generated code"
    
    # Basic syntax checks
    if c_code.count('{') != c_code.count('}'):
        raise CodeValidationError(f"Unbalanced braces in {context}: {{{c_code.count('{')}, }}{c_code.count('}')}")
    
    # Check for proper function signature
    if '__device__' not in c_code:
        warnings.append(f"Missing __device__ qualifier in {context}")
    
    # Check for return statement in non-void functions
    if 'double ' in c_code and 'return ' not in c_code:
        warnings.append(f"Non-void function missing return statement in {context}")
    
    # Check each complete statement (handle multi-line statements)
    import re
    
    # First, reconstruct complete statements by joining lines that don't end with semicolon
    complete_statements = []
    current_statement = ""
    
    lines = c_code.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('//') and not line.startswith('/*'):
            current_statement += " " + line if current_statement else line
            
            # Check if this completes a statement
            if line.endswith(';') or line.endswith('{') or line.endswith('}'):
                complete_statements.append(current_statement.strip())
                current_statement = ""
    
    # Add any remaining partial statement
    if current_statement.strip():
        complete_statements.append(current_statement.strip())
    
    # Now validate each complete statement
    for i, statement in enumerate(complete_statements):
        if statement and not statement.startswith('{') and not statement.startswith('}'):
            # Extract expression to validate
            expression_to_check = None
            
            # Check for return statement
            if 'return ' in statement:
                # Extract expression after 'return '
                return_pos = statement.find('return ') + 7
                expression_to_check = statement[return_pos:].rstrip(';').strip()
            
            # Check for assignment (single = not preceded by !, <, >, or followed by =)
            elif '=' in statement:
                # Look for assignment operator (not >=, <=, ==, !=)
                assignment_match = re.search(r'(?<![!<>=])=(?!=)', statement)
                if assignment_match:
                    # This is a real assignment, extract RHS
                    assignment_pos = assignment_match.start()
                    expression_to_check = statement[assignment_pos + 1:].rstrip(';').strip()
            
            # Validate the extracted expression only if it's complete
            if expression_to_check and not expression_to_check.endswith(('+', '-', '*', '/', '&', '|', '^')):
                try:
                    expr_warnings = validate_c_expression(expression_to_check, f"{context} statement {i+1}")
                    # Add debug output for expressions that generate warnings
                    if expr_warnings:
                        warnings.append(f"DEBUG - Expression with warnings: '{expression_to_check}'")
                    warnings.extend(expr_warnings)
                except CodeValidationError as e:
                    # Provide more detailed error context with expression
                    raise CodeValidationError(f"{context} statement {i+1}: {e}\nStatement: '{statement}'\nExpression: '{expression_to_check}'")
                except Exception as e:
                    raise CodeValidationError(f"{context} statement {i+1}: Unexpected error during validation: {e}\nStatement: '{statement}'\nExpression: '{expression_to_check}'")
    
    return warnings




# --- Notebook-style string manipulation functions ---

def parse_piecewise_argument(arg: str) -> tuple:
    """Parse Piecewise argument by finding the last comma that's not inside parentheses.
    
    Args:
        arg: Piecewise argument string like "expr, And(cond1, cond2)"
        
    Returns:
        Tuple of (expression, condition)
    """
    paren_count = 0
    last_comma_pos = -1
    
    # Scan backwards to find the real expression/condition boundary
    for i in range(len(arg) - 1, -1, -1):
        char = arg[i]
        if char == ')':
            paren_count += 1
        elif char == '(':
            paren_count -= 1
        elif char == ',' and paren_count == 0:  # Top-level comma
            last_comma_pos = i
            break
    
    if last_comma_pos == -1:
        return arg.strip(), "True"
    
    expr = arg[:last_comma_pos].strip()
    cond = arg[last_comma_pos + 1:].strip()
    
    # Handle And() conditions properly
    if cond.startswith("And(") and cond.endswith(")"):
        inner_cond = cond[4:-1].strip()
        if "," in inner_cond:
            # Split And() arguments and join with &&
            cond_parts = []
            part_start = 0
            paren_depth = 0
            
            for i, char in enumerate(inner_cond):
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    cond_parts.append(inner_cond[part_start:i].strip())
                    part_start = i + 1
            
            # Add the last part
            if part_start < len(inner_cond):
                cond_parts.append(inner_cond[part_start:].strip())
            
            # Join with && and wrap each condition in parentheses, then wrap the whole thing
            cond = "(" + " && ".join(f"({part})" for part in cond_parts if part) + ")"
        else:
            cond = inner_cond
    
    return expr, cond


def preprocess_sympy_expression(source: str) -> str:
    """Clean up SymPy expressions before conversion to make them parseable."""
    # Fix common SymPy malformations
    # Remove trailing commas in function calls like pow(x, 2,)
    source = re.sub(r',\s*\)', ')', source)
    
    # Fix malformed scientific notation "(1e)-15" -> "1e-15"
    source = re.sub(r'\((\d+e)\)-(\d+)', r'\1-\2', source)
    
    # Fix extra parentheses before comparison operators
    # Pattern: "(number) <=" -> "number <="
    source = re.sub(r'\((\d+\.?\d*)\)\s*([<>=]+)', r'\1 \2', source)
    
    # Note: We don't do simple And() replacement here anymore
    # The And() functions are handled properly in parse_piecewise_argument
    
    return source


def fix_all_zero_piecewise_from_logs(source: str) -> str:
    """
    Fix all-zero Piecewise expressions that come from differentiating log terms.
    
    When pycalphad differentiates ideal mixing terms Y*log(Y) that are protected
    by Piecewise, it can produce expressions like:
    Piecewise((0, 1e-15 < LIQUID0TI), (0, True))
    
    This is incorrect - the second derivative of Y*log(Y) should be 1/Y, not 0.
    
    This function detects these patterns and replaces them with the correct derivative.
    """
    import re
    
    # Pattern: Piecewise((0, 1e-15 < VARIABLE), (0, True))
    # This comes from d²/dY²[Piecewise((Y*log(Y), Y > 1e-15), (0, True))]
    all_zero_pattern = r'Piecewise\(\(0,\s*1e-15\s*<\s*([A-Za-z0-9_]+)\),\s*\(0,\s*True\)\)'
    
    def fix_pattern(match):
        variable = match.group(1)
        # Replace with the correct second derivative of ideal mixing
        # d²/dY²[Y*log(Y)] = 1/Y when Y > 1e-15
        return f'Piecewise((pow({variable}, -1), 1e-15 < {variable}), (0, True))'
    
    # Apply the fix
    source = re.sub(all_zero_pattern, fix_pattern, source)
    
    # Also fix the reversed pattern: Piecewise((0, VARIABLE > 1e-15), (0, True))
    reversed_pattern = r'Piecewise\(\(0,\s*([A-Za-z0-9_]+)\s*>\s*1e-15\),\s*\(0,\s*True\)\)'
    
    def fix_reversed_pattern(match):
        variable = match.group(1)
        return f'Piecewise((pow({variable}, -1), {variable} > 1e-15), (0, True))'
    
    source = re.sub(reversed_pattern, fix_reversed_pattern, source)
    
    return source


def notebook_replace_piecewise(source: str) -> str:
    """Converts SymPy Piecewise expressions to C ternary operators with robust And() handling."""
    # Preprocess to clean up SymPy expressions
    source = preprocess_sympy_expression(source)
    
    def convert_piecewise_to_ternary(piecewise_str):
        """
        Convert a Piecewise expression to properly formed ternary operators.
        
        Piecewise((expr1, cond1), (expr2, cond2), (expr3, True))
        becomes: ((cond1) ? (expr1) : (cond2) ? (expr2) : (expr3))
        """
        import re
        
        # Extract the content inside Piecewise(...)
        match = re.match(r'Piecewise\((.*)\)$', piecewise_str.strip(), re.DOTALL)
        if not match:
            return piecewise_str
        
        content = match.group(1)
        
        # Parse the (expr, cond) pairs
        pairs = []
        current = ""
        paren_depth = 0
        in_tuple = False
        
        i = 0
        while i < len(content):
            char = content[i]
            
            if char == '(' and paren_depth == 0:
                in_tuple = True
                paren_depth = 1
                current = ""
            elif in_tuple:
                if char == '(':
                    paren_depth += 1
                    current += char
                elif char == ')':
                    paren_depth -= 1
                    if paren_depth == 0:
                        # End of tuple - use parse_piecewise_argument to properly split
                        expr, cond = parse_piecewise_argument(current)
                        # parse_piecewise_argument already handles And() conditions
                        pairs.append((expr, cond))
                        in_tuple = False
                        current = ""
                    else:
                        current += char
                else:
                    current += char
            
            i += 1
        
        # Build the ternary expression
        if not pairs:
            return "0.0"
        
        # Build from the inside out
        result = None
        for i in range(len(pairs) - 1, -1, -1):
            expr, cond = pairs[i]
            
            if cond == "True":
                # Default case
                if result is None:
                    result = f"({expr})"
                else:
                    # This shouldn't happen if True is last
                    result = f"({expr})"
            else:
                # Conditional case
                # Add parentheses to condition if needed
                if not (cond.startswith('(') and cond.endswith(')')):
                    cond = f"({cond})"
                
                if result is None:
                    # Last condition without a default - add 0.0 as default
                    result = f"({cond} ? ({expr}) : 0.0)"
                else:
                    # Wrap the condition and previous result
                    # Don't add extra parentheses if result already has them
                    result = f"({cond} ? ({expr}) : {result})"
        
        return result
    
    # Replace all Piecewise expressions
    while 'Piecewise(' in source:
        # Find the start of a Piecewise
        start_idx = source.find('Piecewise(')
        if start_idx == -1:
            break
        
        # Find the matching closing parenthesis
        paren_count = 0
        i = start_idx + 9  # Skip 'Piecewise'
        while i < len(source):
            if source[i] == '(':
                paren_count += 1
            elif source[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    # Found the end
                    piecewise_expr = source[start_idx:i+1]
                    ternary_expr = convert_piecewise_to_ternary(piecewise_expr)
                    source = source[:start_idx] + ternary_expr + source[i+1:]
                    break
            i += 1
    
    # Fix all-zero Piecewise expressions that come from differentiating log terms
    source = fix_all_zero_piecewise_from_logs(source)
    
    return source


def convert_piecewise_to_ternary(expr):
    """
    Convert SymPy/SymEngine Piecewise expressions to C ternary operators.
    This handles nested Piecewise expressions and preserves function calls properly.
    """
    try:
        import sympy as sp
    except ImportError:
        sp = None
    try:
        import symengine
    except ImportError:
        symengine = None
    
    # Handle both SymPy and SymEngine types
    piecewise_types = []
    if sp and hasattr(sp, 'Piecewise'):
        piecewise_types.append(sp.Piecewise)
    if symengine and hasattr(symengine, 'Piecewise'):
        piecewise_types.append(symengine.Piecewise)
    
    def is_piecewise(e):
        return any(isinstance(e, pw_type) for pw_type in piecewise_types)
    
    def convert_expr(e):
        """Recursively convert an expression, handling Piecewise specially."""
        if is_piecewise(e):
            # Build ternary from inside out
            result = None
            # Process pieces in reverse order to build nested ternaries correctly
            args = list(e.args)
            
            # Handle the last piece (default/else case)
            if len(args) > 0:
                last_piece = args[-1]
                # Check if it's a default case (condition is True)
                if hasattr(last_piece, '__len__') and len(last_piece) == 2:
                    expr, cond = last_piece
                    if str(cond) == 'True' or str(cond) == 'true':
                        result = convert_expr(expr)
                        args = args[:-1]  # Remove the default case
                elif not hasattr(last_piece, '__len__'):
                    # Sometimes the last arg is just the default expression
                    result = convert_expr(last_piece)
                    args = args[:-1]
            
            # Process remaining pieces in reverse order
            for i in range(len(args)-1, -1, -1):
                piece = args[i]
                if hasattr(piece, '__len__') and len(piece) >= 2:
                    expr = piece[0]
                    cond = piece[1]
                    cond_str = str(cond)
                    expr_str = convert_expr(expr)
                    
                    if result is None:
                        result = expr_str
                    else:
                        # Build ternary: condition ? true_expr : false_expr
                        result = f"({cond_str}) ? ({expr_str}) : ({result})"
            
            return result if result else "0"
        elif hasattr(e, 'args') and e.args:
            # This is a function or operation with arguments
            # Check if any arguments contain Piecewise
            has_piecewise_args = any(is_piecewise(arg) or (hasattr(arg, 'args') and contains_piecewise(arg)) for arg in e.args)
            
            if has_piecewise_args:
                # Convert arguments that contain Piecewise
                converted_args = [convert_expr(arg) for arg in e.args]
                
                # Special handling for specific functions
                if hasattr(e, 'func'):
                    func_name = str(e.func).split('.')[-1]  # Get just the function name
                    
                    if func_name in ['Pow', 'pow']:
                        # Ensure pow() is formatted correctly
                        if len(converted_args) == 2:
                            return f"pow({converted_args[0]}, {converted_args[1]})"
                    elif func_name in ['Add', 'add', '+']:
                        return ' + '.join(f"({arg})" for arg in converted_args)
                    elif func_name in ['Mul', 'mul', '*']:
                        return ' * '.join(f"({arg})" for arg in converted_args)
                    elif func_name in ['log', 'Log']:
                        return f"log({converted_args[0]})"
                    else:
                        # Don't output raw function types - reconstruct the expression
                        # Skip function names that are type annotations
                        if ">" in func_name or "<" in func_name:
                            # This is a type-annotated function, reconstruct based on operation
                            if 'Add' in func_name:
                                return ' + '.join(f"({arg})" for arg in converted_args)
                            elif 'Mul' in func_name:
                                return ' * '.join(f"({arg})" for arg in converted_args)
                            else:
                                # Fallback: just join args
                                return ' '.join(converted_args)
                        else:
                            # Generic function call
                            return f"{func_name}({', '.join(converted_args)})"
                        
                # If not a recognized function, try to reconstruct
                return str(e).replace(str(e.args[0]), converted_args[0])
            else:
                # No Piecewise in arguments, return as string
                return str(e)
        else:
            # Simple expression without args
            return str(e)
    
    def contains_piecewise(expr):
        """Check if an expression contains any Piecewise."""
        if is_piecewise(expr):
            return True
        if hasattr(expr, 'args'):
            return any(contains_piecewise(arg) for arg in expr.args)
        return False
    
    # Convert the expression
    result = convert_expr(expr)
    
    # If result is still an expression object, convert to string
    if not isinstance(result, str):
        result = str(result)
    
    return result


def fix_ternary_operator_precedence(source: str) -> str:
    """Fix operator precedence issues with ternary operators in generated C code.
    
    The issue: expressions like "cond1 ? val1 : 0 + cond2 ? val2 : 0" are parsed as
    "cond1 ? val1 : (0 + (cond2 ? val2 : 0))" due to C operator precedence.
    
    This causes the second ternary to only be evaluated when cond1 is false.
    
    Fix: Add parentheses to ensure proper evaluation of multiple ternary operators.
    """
    import re
    
    # Fix the specific pattern where ternary operators are chained with + 
    # Pattern: "(condition ? value : 0 + (condition ? value : 0"
    # Should become: "(condition ? value : 0) + ((condition ? value : 0"
    
    # This pattern captures two ternary expressions separated by + where the first ends with : 0
    # Updated to handle the actual pattern: "ternary : 0 + number*(...) ? ... : 0"
    inner_pattern = r'(\([^?]+\?\s*[^:]+\s*:\s*0)\s*\+\s*([\d.]+\*\([^?]+\?\s*[^:]+\s*:\s*0)'
    
    def fix_inner_replacement(match):
        first_ternary = match.group(1)   # First ternary ending with : 0
        second_ternary = match.group(2)  # Second ternary ending with : 0  
        return f"{first_ternary}) + ({second_ternary}"
    
    # Apply the fix for chained ternary operators
    fixed = re.sub(inner_pattern, fix_inner_replacement, source)
    
    # Separate chained ternary operators that should be additions
    # The problem: After Piecewise conversion, we might have:
    # x[3]*(cond1) ? val1 : (cond2) ? val2 : 0 + x[4]*(cond3) ? val3 : 0
    # This chains NB and TI terms with ternary operators instead of adding them
    # When cond1 is true, val1 is returned and the TI term is never evaluated!
    
    # Look for patterns where we have ": 0 + x[4]*" which indicates chained terms
    # that should be separate additions
    if ": 0 + x[4]*" in fixed or ": 0 + (x[4]*" in fixed:
        # This is a mechanical mixture with improperly chained terms
        # We need to close the first ternary and start a new one
        
        # Pattern: ": 0 + x[4]*" should become ": 0) + (x[4]*"
        fixed = fixed.replace(": 0 + x[4]*", ": 0) + (x[4]*")
        fixed = fixed.replace(": 0 + (x[4]*", ": 0) + ((x[4]*")
        
        # Also handle the case where we already have some parentheses
        fixed = fixed.replace(": 0) + x[4]*", ": 0) + (x[4]*")
        
        # Now ensure division applies to the entire sum if present
        # Look for pattern: ) + (...)/(x[3] + x[4])
        # Should be: (...))/(x[3] + x[4])
        div_pattern = r'\) \+ \(([^)]+)\)/(x\[3\] \+ x\[4\])'
        def fix_division_scope(match):
            ti_term = match.group(1)
            # Move the closing paren after the TI term before the division
            return f") + ({ti_term}))/(x[3] + x[4]"
        
        fixed = re.sub(div_pattern, fix_division_scope, fixed)
        
        # Ensure we close any opened parentheses
        # Count parentheses and add closing ones if needed
        open_count = fixed.count('(')
        close_count = fixed.count(')')
        if open_count > close_count:
            fixed = fixed + ')' * (open_count - close_count)
    
    # Mechanical mixture term with ternary operators
    # The problem: 1.0*(x[3]*(cond1) ? val1 : val1b + x[4]*(cond2) ? val2 : val2b)/(x[3] + x[4])
    # Due to operator precedence, the + binds tighter than ?:, causing:
    # x[3]*cond1 ? val1 : (val1b + x[4]*cond2 ? val2 : val2b)
    # This means if cond1 is true, val2 is never evaluated!
    # We need: ((x[3]*cond1 ? val1 : val1b) + (x[4]*cond2 ? val2 : val2b))/(x[3] + x[4])
    
    # Fix mechanical mixture multiplication structure
    # The problem: x[3]*(condition) ? value : 0 
    # Is interpreted as: (x[3]*condition) ? value : 0
    # Which returns full value when x[3]*condition is true (non-zero)
    # We need: (condition) ? x[3]*value : 0
    
    # First, let's fix the site fraction multiplication issue
    # Pattern: x[3]*(condition) ? value : ...
    # Should be: (condition) ? x[3]*value : ...
    
    # Fix for x[3] terms
    sf_mult_pattern1 = r'x\[3\]\*\((x\[2\][^)]+)\)\s*\?\s*\(([^:]+)\)\s*:'
    def fix_sf_mult1(match):
        condition = match.group(1)  # Temperature condition
        value = match.group(2)      # Energy expression
        # Move x[3]* inside the ternary
        return f"({condition}) ? x[3]*({value}) :"
    fixed = re.sub(sf_mult_pattern1, fix_sf_mult1, fixed)
    
    # Fix for x[4] terms  
    sf_mult_pattern2 = r'x\[4\]\*\((x\[2\][^)]+)\)\s*\?\s*\(([^:]+)\)\s*:'
    def fix_sf_mult2(match):
        condition = match.group(1)  # Temperature condition
        value = match.group(2)      # Energy expression
        # Move x[4]* inside the ternary
        return f"({condition}) ? x[4]*({value}) :"
    fixed = re.sub(sf_mult_pattern2, fix_sf_mult2, fixed)
    
    # Now apply the parentheses fix for proper grouping
    # This ensures the sum is calculated before division
    if "1.0*(" in fixed and " + " in fixed and ")/(x[3] + x[4])" in fixed:
        # Find mechanical mixture terms that need grouping
        # Pattern: 1.0*(term1 + term2)/(x[3] + x[4])
        mech_pattern = r'(1\.0\*\()([^/]+)(\)\/\(x\[3\] \+ x\[4\]\))'
        
        def ensure_grouping(match):
            prefix = match.group(1)   # "1.0*("
            content = match.group(2)  # The terms
            suffix = match.group(3)   # ")/(x[3] + x[4])"
            
            # Don't add extra parentheses - the structure is already correct
            return match.group(0)
            
        fixed = re.sub(mech_pattern, ensure_grouping, fixed)
    
    # Ideal mixing term with improper division
    # After the first fix, we have:
    # 8.3145*x[2]*(term1) + (term2)/(x[3] + x[4])
    # But this applies division only to term2!
    # We need: 8.3145*x[2]*((term1) + (term2))/(x[3] + x[4])
    
    # Use a more direct approach due to nested parentheses
    if "8.3145*x[2]*" in fixed and ") + (" in fixed and "/(x[3] + x[4])" in fixed:
        # Find the start of the ideal mixing term
        ideal_start = fixed.find("8.3145*x[2]*")
        if ideal_start >= 0:
            # Find the division that should apply to the whole ideal mixing
            div_pattern = "/(x[3] + x[4])"
            div_pos = fixed.find(div_pattern, ideal_start)
            
            if div_pos > ideal_start:
                # Extract everything between the factor and the division
                factor_end = ideal_start + len("8.3145*x[2]*")
                between = fixed[factor_end:div_pos]
                
                # Check if this looks like our problematic pattern
                # Should be something like: (term1) + (term2)
                # Count parentheses to ensure we have the right structure
                if between.count('(') == between.count(')') and ') + (' in between:
                    # Find the split point - where we have balanced parentheses
                    # and a + operator at the top level
                    paren_count = 0
                    split_pos = -1
                    
                    for i, char in enumerate(between):
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                        elif char == '+' and paren_count == 0:
                            # This is a top-level +
                            split_pos = i
                            break
                    
                    if split_pos > 0:
                        # We found the split point, now wrap the whole expression
                        before = fixed[:factor_end]
                        after = fixed[div_pos:]
                        # Add extra parentheses around the entire sum
                        fixed = f"{before}({between}){after}"
    
    # Balance parentheses if needed
    open_count = fixed.count('(')
    close_count = fixed.count(')')
    if open_count > close_count:
        fixed = fixed + ')' * (open_count - close_count)
    
    return fixed


def notebook_replace_exp(source: str) -> str:
    """Converts Python ** operator to C pow() function. Original from Phase_rec.ipynb.txt with fix."""
    source += " " #to avoid infinite loop if the statement ends with a space!
    index = source.find("**")
    while(index > 0):
        exp_start = index+2
        # Look for the end of the exponent - could be space, comma, or closing paren
        exp_end = exp_start
        p_count = 0
        break_chars = " ,)+-*/"  # Added comma and closing paren to handle Piecewise expressions
        
        # Find the end of the exponent
        for i in range(exp_start, len(source)):
            if source[i] == "(":
                p_count += 1
            elif source[i] == ")":
                p_count -= 1
                # If we hit a closing paren at depth 0, we're done
                # This handles cases like x**3.0)) where the exponent is just 3.0
                if p_count < 0:
                    exp_end = i
                    break
                # Check if we've closed all parentheses from within the exponent
                elif p_count == 0 and i > exp_start:
                    exp_end = i + 1  # Include the closing parenthesis
                    break
            elif p_count == 0 and source[i] in break_chars:
                exp_end = i
                break
        
        # If we didn't find a break character, use the end of string
        if exp_end == exp_start:
            exp_end = len(source) - 1  # Subtract 1 for the added space
        
        base_end=index
        base_start = 0  # Initialize base_start
        p_count = 0
        break_chars = "+-/* ,("  # Added opening paren to stop at function/Piecewise boundaries
        for i in range(index-1, 0, -1):
            if(source[i] == ")"):
                p_count -= 1
            elif(source[i] == "("):
                p_count += 1
                # If we hit an opening paren at depth 0, we've gone too far
                if p_count > 0:
                    base_start = i+1
                    break
            elif(p_count == 0 and source[i] in break_chars):
                base_start = i+1
                break
        new_pow = "pow("+source[base_start:base_end]+", "+source[exp_start:exp_end]+")"
        source = source[:base_start]+new_pow+source[exp_end:]
        index = source.find("**")
    return source[:-1] #remove final space now that it's no longer needed


def _max_phase_local_conditions(wks_obj):
    """Total number of phase-local conditions (conditions with a phase name,
    e.g. X(FCC_A1,ZN) or Y(LIQUID,0,ZN)). Sizes both the spec's system-wide
    list and the per-compset attachment arrays (a compset attaches at most
    all of them)."""
    return sum(1 for key in (getattr(wks_obj, 'conditions', {}) or {})
               if getattr(key, 'phase_name', None) is not None)


def _fit_parameter_symbols(wks_obj):
    """Fit-parameter symbols carried by the phase-record factory (str-sorted by
    pycalphad's extract_parameters). Mapped to trailing dof slots
    x[num_statevars + phase_dof + j] so parameters are RUNTIME inputs of the
    generated functions (no recompile per parameter set; per-thread parameter
    vectors enable batching MCMC walkers)."""
    prf = getattr(wks_obj, 'phase_record_factory', None)
    syms = list(getattr(prf, 'param_symbols', []) or [])
    return syms


def notebook_get_all_syms_for_model(model_obj: Model, wks_obj: Workspace) -> list:
    """Get all symbols for a model in the correct order for derivatives. Original from Phase_rec.ipynb.txt."""
    import symengine as se
    from pycalphad.variables import pressure
    
    site_variables = model_obj.site_fractions
    # Use phase_record_factory's state variables to match CPU behavior
    # The CPU builds functions with all state variables, not just the ones the model uses
    if hasattr(wks_obj, 'phase_record_factory') and wks_obj.phase_record_factory is not None:
        state_variables = wks_obj.phase_record_factory.state_variables
    else:
        # Fallback to model's state variables if phase_record_factory not available
        state_variables = model_obj.state_variables
    
    # This should match exactly what the CPU expects:
    # For BCC_A2: [N, P, T] + [Y(BCC_A2,0,NB), Y(BCC_A2,0,TI)]
    # Fit parameters (if any) ride in trailing dof slots after the site fractions.
    return state_variables + site_variables + _fit_parameter_symbols(wks_obj)


def notebook_get_all_sym_names_for_model(model_obj: Model, wks_obj: Workspace) -> list:
    """Get all symbol names for a model. Original from Phase_rec.ipynb.txt."""
    site_fracs = model_obj.site_fractions
    site_variables = []
    for sf in site_fracs:
        # MUST use the species' ESCAPED name: SiteFraction symbol names are
        # built from it (variables.py), and charged species differ from the
        # raw name (FE+3 -> FE_POS3). The raw name never appears in printed
        # expressions, so using it leaves those symbols unmapped -> undeclared
        # identifiers in the generated C.
        site_variables.append(sf.phase_name+str(sf.sublattice_index)+sf.species.escaped_name)
    
    # Use phase_record_factory's state variables to match CPU behavior
    # The CPU builds functions with all state variables, not just the ones the model uses
    if hasattr(wks_obj, 'phase_record_factory') and wks_obj.phase_record_factory is not None:
        state_variables = [str(var) for var in wks_obj.phase_record_factory.state_variables]
    else:
        # Fallback to model's state variables if phase_record_factory not available
        state_variables = [str(var) for var in model_obj.state_variables]
    
    # This gives CPU DOF ordering:
    # For BCC_A2: ['N', 'P', 'T'] + ['BCC_A20NB', 'BCC_A20TI'] = indices 0, 1, 2, 3, 4
    # Fit parameters (if any) ride in trailing dof slots after the site fractions.
    return state_variables + site_variables + [str(sym) for sym in _fit_parameter_symbols(wks_obj)]


def notebook_convert_var_names(source_str: str, model_obj: Model, wks_obj: Workspace, verbose: bool = False) -> str:
    """Convert symbolic variable names to array indices in C code. Fixed to avoid substring issues."""
    import re
    names = notebook_get_all_sym_names_for_model(model_obj, wks_obj)
    
    # Debug: Print the variable mapping (only once per phase)
    if not hasattr(notebook_convert_var_names, '_printed_mappings'):
        notebook_convert_var_names._printed_mappings = set()
    if model_obj.phase_name not in notebook_convert_var_names._printed_mappings:
        if verbose:
            print(f"[GPU CODEGEN] Variable mapping for {model_obj.phase_name}: {dict(zip(names, [f'x[{i}]' for i in range(len(names))]))}")
        notebook_convert_var_names._printed_mappings.add(model_obj.phase_name)
    
    # Traverse in reverse order to replace longer names first (e.g., BCC_A20NB before N)
    for i, name in reversed(list(enumerate(names))):
        # Use word boundaries to avoid partial matches within other variable names
        # This prevents 'N' from being replaced inside 'BCC_A20NB' and 'T' inside 'BCC_A20TI'
        pattern = r'\b' + re.escape(name) + r'\b'
        source_str = re.sub(pattern, f"x[{i}]", source_str)
    return source_str


def notebook_model_c_func_name_prefix(model_c_idx: int) -> str:
    """Generate C function name prefix for a model."""
    return f"pycgpu_model_{model_c_idx}_"




def fix_piecewise_zeros(expression: str) -> str:
    """
    Fix Piecewise expressions where all branches are zero.
    
    Pattern like:
    ((x[0] < 2750.0) ? (0) : ((2750.0 <= x[0]) ? (0) : (0)))
    
    Should just be:
    0
    
    EXCEPTION: Ideal mixing Hessian patterns like:
    ((1e-15 < x[1]) ? (0) : (0))
    
    Should become:
    ((1e-15 < x[1]) ? (pow(x[1], (-1))) : (0))
    """
    import re
    
    # Keep applying fixes until no more changes
    prev_expr = ""
    iteration = 0
    max_iterations = 10  # Prevent infinite loops
    
    while prev_expr != expression and iteration < max_iterations:
        prev_expr = expression
        iteration += 1
        
        # Fix ideal mixing Hessian patterns FIRST before any simplification
        # These come from d²/dY²[Y*log(Y)] and should be 1/Y not 0
        
        # Fix 1: Simple pattern ((1e-15 < x[i]) ? (0) : (0))
        for i in range(10):  # Check indices 0-9
            pattern = rf'\(\(1e-15\s*<\s*x\[{i}\]\)\s*\?\s*\(0\)\s*:\s*\(0\)\)'
            replacement = f'((1e-15 < x[{i}]) ? (pow(x[{i}], (-1))) : (0))'
            expression = re.sub(pattern, replacement, expression)
        
        # Fix 2: Pattern with multiplication coefficients
        # e.g., 1.0*((1e-15 < x[1]) ? (0) : (0))
        for i in range(10):
            pattern = rf'(\d+\.?\d*)\s*\*\s*\(\(1e-15\s*<\s*x\[{i}\]\)\s*\?\s*\(0\)\s*:\s*\(0\)\)'
            replacement = rf'\1*((1e-15 < x[{i}]) ? (pow(x[{i}], (-1))) : (0))'
            expression = re.sub(pattern, replacement, expression)
        
        # Now apply other simplifications for truly all-zero patterns
        
        # Pattern 3: Simple ternary with both branches zero (but NOT ideal mixing)
        # Match any condition that doesn't involve 1e-15
        expression = re.sub(r'\(\((?!.*1e-15)[^()]+(?:\([^()]*\)[^()]*)*\)\s*\?\s*\(0\)\s*:\s*\(0\)\)', '0', expression)
    
        # Pattern 4: Nested ternary with all zeros (but NOT ideal mixing)
        # ((x[0] < 2750.0) ? (0) : ((2750.0 <= x[0]) ? (0) : (0)))
        expression = re.sub(r'\(\((?!.*1e-15)[^()]+(?:\([^()]*\)[^()]*)*\)\s*\?\s*\(0\)\s*:\s*\(\([^()]+(?:\([^()]*\)[^()]*)*\)\s*\?\s*\(0\)\s*:\s*\(0\)\)\)', '0', expression)
    
        # Pattern 5: Just (0) -> 0
        expression = expression.replace('(0)', '0')
    
        # Pattern 6: Clean up 1.0*0 (not 1.0*0.something)
        expression = re.sub(r'1\.0\s*\*\s*0(?!\.|\d)', '0', expression)
    
        # Pattern 7: Clean up x[i]*0
        expression = re.sub(r'x\[\d+\]\s*\*\s*0(?!\.|\d)', '0', expression)
    
        # Pattern 8: Clean up (0 + 0) -> 0
        expression = expression.replace('(0 + 0)', '0')
    
        # Pattern 9: Clean up 0 + expr -> expr (but not in 10 + expr)
        # Don't apply this pattern - it breaks expressions like:
        # "((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1 + 11.0*" 
        # by removing "0) + " and leaving invalid syntax ") 1 + 11.0*"
        # expression = re.sub(r'(?<![\.\d])0\s*\+\s*', '', expression)
        # Skip this pattern entirely for now
    
        # Pattern 10: Clean up expr + 0 -> expr (but not expr + 0.1)
        expression = re.sub(r'\s*\+\s*0(?![\.\d])', '', expression)
    
        # Pattern 11: Clean up - 0 -> nothing
        expression = re.sub(r'\s*-\s*0(?![\.\d])', '', expression)
    
        # Pattern 12: Clean up factor*0/divisor -> 0
        expression = re.sub(r'[\d.]+\s*\*\s*0\s*/\s*\([^)]+\)', '0', expression)
    
        # Pattern 13: Clean up 0/(anything) -> 0
        expression = re.sub(r'0\s*/\s*\([^)]+\)', '0', expression)
    
        # Pattern 14: Clean up parentheses around single 0
        expression = re.sub(r'\(\s*0\s*\)', '0', expression)
        
        # Pattern 15: More aggressive ternary cleanup
        # ((condition) ? (0) : 0) -> 0
        expression = re.sub(r'\(\([^()]+\)\s*\?\s*\(0\)\s*:\s*0\)', '0', expression)
        expression = re.sub(r'\(\([^()]+\)\s*\?\s*0\s*:\s*\(0\)\)', '0', expression)
    
    return expression











def notebook_source_from_expr_cse(
    expr_or_list_in, 
    c_function_name_base_suffix: str,
    model_obj: Model,
    model_c_idx: int,
    wks_obj: Workspace,
    expr_type: str = "func", 
    c_output_type: str = "double",
    validate: bool = True,
    verbose: bool = False,
) -> str:
    """
    NEW CSE-BASED: Converts SymEngine expression(s) to C __device__ function using CSE + ccode().
    This replaces the regex-heavy approach with SymEngine's native capabilities.
    """
    try:
        num_exprs = len(expr_or_list_in)
    except:
        num_exprs = 1 if hasattr(expr_or_list_in, 'free_symbols') else len(list(expr_or_list_in))
    if verbose:
        print(f"[CSE CODEGEN] Processing {expr_type} with {num_exprs} expressions")
    
    start_time = time.time()
    
    # Build function name and signature
    full_c_func_name = notebook_model_c_func_name_prefix(model_c_idx) + c_function_name_base_suffix
    c_output_arg_name = "out"
    c_input_arg_name = "x"
    
    # Start building the C function
    if expr_type == "fused":
        # Fused energy+gradient+Hessian: two output arrays (energy+gradient
        # small, Hessian written into the caller's existing work buffer).
        c_code = (f"__device__ void {full_c_func_name}(double* out_eg, "
                  f"double* out_hess, const double* {c_input_arg_name}) {{\n")
    elif expr_type == "func" and c_output_type == "double" and (hasattr(expr_or_list_in, 'free_symbols') or len(expr_or_list_in) == 1):
        # Scalar function - no output parameter
        c_code = f"__device__ {c_output_type} {full_c_func_name}(const double* {c_input_arg_name}) {{\n"
    else:
        # Vector function or void - include output parameter
        c_code = f"__device__ {c_output_type} {full_c_func_name}(double* {c_output_arg_name}, const double* {c_input_arg_name}) {{\n"
    
    try:
        if expr_type == "func":
            # Process function expressions (free energy, etc.)
            # Handle both single expressions and lists
            if hasattr(expr_or_list_in, 'free_symbols'):
                # Single expression
                all_expressions = [expr_or_list_in]
            else:
                # List of expressions
                all_expressions = list(expr_or_list_in)
            
            # Apply CSE to all expressions together for maximum efficiency
            if verbose:
                print(f"[CSE CODEGEN] Applying CSE to {len(all_expressions)} function expressions...")
            cse_start = time.time()
            replacements, reduced_exprs = cse(all_expressions)
            cse_time = time.time() - cse_start
            
            if verbose:
                print(f"[CSE CODEGEN] CSE found {len(replacements)} common subexpressions in {cse_time:.3f}s")
            
            # Generate subexpression assignments
            for symbol, subexpr in replacements:
                c_subexpr = ccode(subexpr)
                c_subexpr = apply_cse_variable_mapping(c_subexpr, model_obj, wks_obj, verbose)
                c_code += f"    double {ccode(symbol)} = {c_subexpr};\n"
            
            # Generate main expressions
            if len(reduced_exprs) == 1 and c_output_type == "double":
                # Single scalar function - return directly
                c_expr = ccode(reduced_exprs[0])
                c_expr = apply_cse_variable_mapping(c_expr, model_obj, wks_obj, verbose)
                c_code += f"    return {c_expr};\n"
            else:
                # Multiple expressions or void function - write to output array
                for i, reduced_expr in enumerate(reduced_exprs):
                    c_expr = ccode(reduced_expr)
                    c_expr = apply_cse_variable_mapping(c_expr, model_obj, wks_obj, verbose)
                    c_code += f"    {c_output_arg_name}[{i}] = {c_expr};\n"
        
        elif expr_type == "grad":
            # Process gradient expressions
            if verbose:
                print(f"[CSE CODEGEN] Processing gradient expressions...")
            all_grad_expressions = []
            
            # Get ordered symbols for differentiation
            ordered_symbols_for_diff = get_ordered_symbols_for_diff(model_obj, wks_obj, verbose)
            
            # Handle both single expressions and lists
            if hasattr(expr_or_list_in, 'free_symbols'):
                expr_list = [expr_or_list_in]
            else:
                expr_list = list(expr_or_list_in)
            
            # Collect all gradient expressions
            for m_expr_idx, sub_expr in enumerate(expr_list):
                for sym_to_diff_against in ordered_symbols_for_diff:
                    deriv_expr = sub_expr.diff(sym_to_diff_against)
                    all_grad_expressions.append(deriv_expr)
            
            # Apply CSE to all gradients together
            cse_start = time.time()
            replacements, reduced_exprs = cse(all_grad_expressions)
            cse_time = time.time() - cse_start
            
            if verbose:
                print(f"[CSE CODEGEN] Gradient CSE found {len(replacements)} subexpressions in {cse_time:.3f}s")
            
            # Generate subexpression assignments
            for symbol, subexpr in replacements:
                c_subexpr = ccode(subexpr)
                c_subexpr = apply_cse_variable_mapping(c_subexpr, model_obj, wks_obj, verbose)
                c_code += f"    double {ccode(symbol)} = {c_subexpr};\n"
            
            # Generate gradient assignments in the EXACT order we want
            # The reduced_exprs from CSE might not be in the order we need
            # We need to map them back to our intended order
            
            # Build a mapping of which reduced_expr corresponds to which derivative
            expr_index = 0
            output_index = 0
            
            for m_expr_idx, sub_expr in enumerate(expr_list):
                for sym_idx, sym_to_diff_against in enumerate(ordered_symbols_for_diff):
                    # This is the derivative we want at output position 'output_index'
                    # It corresponds to reduced_exprs[expr_index]
                    c_expr = ccode(reduced_exprs[expr_index])
                    c_expr = apply_cse_variable_mapping(c_expr, model_obj, wks_obj, verbose)
                    c_code += f"    {c_output_arg_name}[{output_index}] = {c_expr};\n"
                    
                    expr_index += 1
                    output_index += 1
        
        elif expr_type == "fused":
            # Fused energy + gradient + Hessian with ONE shared CSE pool.
            # The expression trees are built EXACTLY as the func/grad/hess
            # branches build them (same diff calls, same ordering), so every
            # output evaluates the identical tree — cse() only names shared
            # subtrees, it never reassociates arithmetic, hence the values
            # are bit-identical to the separate functions while shared
            # subexpressions are computed once.
            ordered_symbols_for_diff = get_ordered_symbols_for_diff(model_obj, wks_obj, verbose)
            if hasattr(expr_or_list_in, 'free_symbols'):
                expr_list = [expr_or_list_in]
            else:
                expr_list = list(expr_or_list_in)
            all_fused = list(expr_list)                     # energy
            n_eg = len(all_fused)
            for sub_expr in expr_list:                      # gradient
                for sym in ordered_symbols_for_diff:
                    all_fused.append(sub_expr.diff(sym))
            n_grad = len(all_fused) - n_eg
            for sub_expr in expr_list:                      # Hessian (row-major)
                for sym_j in ordered_symbols_for_diff:
                    first_deriv = sub_expr.diff(sym_j)
                    for sym_k in ordered_symbols_for_diff:
                        all_fused.append(first_deriv.diff(sym_k))
            cse_start = time.time()
            replacements, reduced_exprs = cse(all_fused)
            if verbose:
                print(f"[CSE CODEGEN] Fused CSE: {len(replacements)} shared "
                      f"subexpressions over {len(all_fused)} outputs in "
                      f"{time.time() - cse_start:.3f}s")
            for symbol, subexpr in replacements:
                c_subexpr = ccode(subexpr)
                c_subexpr = apply_cse_variable_mapping(c_subexpr, model_obj, wks_obj, verbose)
                c_code += f"    double {ccode(symbol)} = {c_subexpr};\n"
            for i, reduced_expr in enumerate(reduced_exprs):
                c_expr = ccode(reduced_expr)
                c_expr = apply_cse_variable_mapping(c_expr, model_obj, wks_obj, verbose)
                if i < n_eg + n_grad:
                    c_code += f"    out_eg[{i}] = {c_expr};\n"
                else:
                    c_code += f"    out_hess[{i - n_eg - n_grad}] = {c_expr};\n"

        elif expr_type == "hess":
            # Process Hessian expressions
            if verbose:
                print(f"[CSE CODEGEN] Processing Hessian expressions...")
            all_hess_expressions = []
            
            # Get ordered symbols for differentiation
            ordered_symbols_for_diff = get_ordered_symbols_for_diff(model_obj, wks_obj, verbose)
            
            # Handle both single expressions and lists
            if hasattr(expr_or_list_in, 'free_symbols'):
                expr_list = [expr_or_list_in]
            else:
                expr_list = list(expr_or_list_in)
            
            # Collect all Hessian expressions
            for m_expr_idx, sub_expr in enumerate(expr_list):
                for i_sym_idx, sym_j in enumerate(ordered_symbols_for_diff):
                    first_deriv = sub_expr.diff(sym_j)
                    for j_sym_idx, sym_k in enumerate(ordered_symbols_for_diff):
                        second_deriv_expr = first_deriv.diff(sym_k)
                        all_hess_expressions.append(second_deriv_expr)
            
            # Apply CSE to all Hessian expressions together
            cse_start = time.time()
            replacements, reduced_exprs = cse(all_hess_expressions)
            cse_time = time.time() - cse_start
            
            if verbose:
                print(f"[CSE CODEGEN] Hessian CSE found {len(replacements)} subexpressions in {cse_time:.3f}s")
            
            # Generate subexpression assignments
            for symbol, subexpr in replacements:
                c_subexpr = ccode(subexpr)
                c_subexpr = apply_cse_variable_mapping(c_subexpr, model_obj, wks_obj, verbose)
                c_code += f"    double {ccode(symbol)} = {c_subexpr};\n"
            
            # Generate Hessian assignments
            for i, reduced_expr in enumerate(reduced_exprs):
                c_expr = ccode(reduced_expr)
                c_expr = apply_cse_variable_mapping(c_expr, model_obj, wks_obj, verbose)
                c_code += f"    {c_output_arg_name}[{i}] = {c_expr};\n"
        
        # Close the function
        c_code += "}\n\n"
        
        total_time = time.time() - start_time
        if verbose:
            print(f"[CSE CODEGEN] Generated {expr_type} function in {total_time:.3f}s")
        
        return c_code
        
    except Exception as e:
        # A model expression the generator cannot lower (e.g. unevaluated
        # Derivative nodes): a capability limit, so the dispatch uses the
        # reference implementation for this system.
        from pycalphad.backend import AcceleratedCapabilityError
        raise AcceleratedCapabilityError(
            f"cannot generate code for {model_obj.phase_name} {expr_type}: {e}") from e

def apply_cse_variable_mapping(c_expr: str, model_obj: Model, wks_obj: Workspace, verbose: bool = False) -> str:
    """Apply variable name to array index mapping using existing logic."""
    # Use existing function to get the mapping
    result = notebook_convert_var_names(c_expr, model_obj, wks_obj, verbose)
    
    # Fix SymEngine CSE symbols that aren't handled by the original replacements
    result = fix_cse_symbols(result)
    
    return result

def _convert_and_calls(source: str) -> str:
    """Replace And(a, b, ...) with (a && b && ...), paren-depth aware.

    A naive regex split at the first comma breaks when a condition argument
    itself contains commas — e.g. the Brosh-EOS magnetic conditions
    And(0 < c*pow(e, f(P))*y, c*pow(e, f(P))*y < T), where the pow(e, ...)
    comma was captured and the '&&' landed inside pow's argument list.
    """
    out = source
    pos = 0
    while True:
        i = out.find('And(', pos)
        if i < 0:
            break
        if i > 0 and (out[i-1].isalnum() or out[i-1] == '_'):
            pos = i + 4
            continue
        depth = 0
        j = i + 3  # at '('
        args, start = [], i + 4
        while j < len(out):
            ch = out[j]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    args.append(out[start:j].strip())
                    break
            elif ch == ',' and depth == 1:
                args.append(out[start:j].strip())
                start = j + 1
            j += 1
        else:
            break  # unbalanced; leave as-is
        repl = '(' + ' && '.join(f'({a})' for a in args if a) + ')'
        out = out[:i] + repl + out[j+1:]
        pos = i  # rescan (handles nested And)
    return out


def fix_cse_symbols(source: str) -> str:
    """Fix SymEngine CSE symbols that need C replacements."""
    import re
    
    # Replace And() with logical AND (&&), respecting nested parentheses
    source = _convert_and_calls(source)
    
    # Replace True with 1 and False with 0
    source = re.sub(r'\bTrue\b', '1', source)
    source = re.sub(r'\bFalse\b', '0', source)
    
    return source

def get_ordered_symbols_for_diff(model_obj: Model, wks_obj: Workspace, verbose: bool = False) -> list:
    """Get ordered symbols for differentiation in the correct order for GPU.
    
    Returns T followed by the site fractions in the order of the phase
    record's degrees of freedom (Model.site_fractions, sorted by their string
    form). Gradient and Hessian outputs are indexed in this order, so it must
    match the DOF vector exactly; sorting by raw species names breaks for
    charged species, whose escaped names (V_POS2) sort differently (V+2).
    """
    from pycalphad import variables as v
    import symengine as se

    ordered_symbols = []
    # IMPORTANT: Sort free_symbols to ensure deterministic iteration order
    temp_found = False
    for sym in sorted(model_obj.ast.free_symbols, key=lambda s: str(s)):
        if str(sym) == 'T':
            ordered_symbols.append(sym)
            temp_found = True
            break
    # If T wasn't found in free symbols but is in state variables, add it
    if not temp_found:
        for var in model_obj.state_variables:
            if isinstance(var, v.Temperature) or str(var) == 'T':
                ordered_symbols.append(se.Symbol('T'))
                break

    ordered_symbols.extend(model_obj.site_fractions)

    if verbose:
        print(f"[CSE GRADIENT] Ordered symbols for {model_obj.phase_name}: {[str(s) for s in ordered_symbols]}")
    
    return ordered_symbols


def _zero_undefined_symbols(expr_or_list, wks_obj):
    """Reference semantics for symbols the database leaves undefined.

    PhaseRecordFactory.get_phase_property / build_functions force undefined
    non-state-variable, non-fit-parameter symbols to zero before compiling
    (e.g. COST507's GAS phase references RTLNP, which the database never
    defines). The generated code must do the same or those names leak into
    the C source as undeclared identifiers.
    """
    import pycalphad.variables as _v
    params = set(_fit_parameter_symbols(wks_obj))
    def _clean(expr):
        try:
            undefs = {x for x in expr.free_symbols
                      if not isinstance(x, _v.StateVariable)} - params
        except AttributeError:
            return expr  # plain numbers
        if undefs:
            expr = expr.xreplace({x: 0. for x in undefs})
        return expr
    if isinstance(expr_or_list, (list, tuple)):
        return type(expr_or_list)(_clean(e) for e in expr_or_list)
    return _clean(expr_or_list)


def _realify_eps_complex(expr_or_list):
    """Drop machine-eps imaginary residue from numeric constants.

    symengine's fractional-power arithmetic can leave complex constants with
    |imag| ~ 1e-16 in model expressions (e.g. the Brosh EOS terms in
    cfe_broshe LIQUID: -10/9 - 1.4e-16*I). The reference LLVM path tolerates
    these; the C printer cannot. Take the real part when the imaginary part
    is negligible relative to the real part; raise (-> silent fallback) if a
    genuinely complex constant ever appears.
    """
    import symengine as se

    def _clean(expr):
        try:
            atoms = expr.atoms(se.Number)
        except AttributeError:
            return expr
        subs = {}
        for a in atoms:
            try:
                if getattr(a, 'is_real', True):
                    continue
                c = complex(a)
                re_p, im_p = c.real, c.imag
            except (TypeError, RuntimeError):
                continue
            scale = max(abs(re_p), 1.0)
            if abs(im_p) <= 1e-12 * scale:
                subs[a] = se.Float(re_p)
            else:
                raise ValueError(
                    f"model expression contains a genuinely complex constant: {a}")
        return expr.xreplace(subs) if subs else expr

    if isinstance(expr_or_list, (list, tuple)):
        return type(expr_or_list)(_clean(e) for e in expr_or_list)
    return _clean(expr_or_list)


def notebook_source_from_expr(
    expr_or_list_in, 
    c_function_name_base_suffix: str,
    model_obj: Model,
    model_c_idx: int,
    wks_obj: Workspace,
    expr_type: str = "func", 
    c_output_type: str = "double",
    validate: bool = True,
    verbose: bool = False,
) -> str:
    """
    C source for a model expression (CSE + SymEngine's C printer).
    """
    expr_or_list_in = _zero_undefined_symbols(expr_or_list_in, wks_obj)
    expr_or_list_in = _realify_eps_complex(expr_or_list_in)
    return notebook_source_from_expr_cse(
        expr_or_list_in, c_function_name_base_suffix, model_obj, model_c_idx,
        wks_obj, expr_type, c_output_type, validate, verbose)


# --- Property-specific C code generator functions ---

def _nb_obj_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    return notebook_source_from_expr(model_obj.GM, "obj", model_obj, model_c_idx, wks_obj, expr_type="func", c_output_type="double", validate=validate, verbose=verbose)

def _nb_formulaobj_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    return notebook_source_from_expr(model_obj.G, "formulaobj", model_obj, model_c_idx, wks_obj, expr_type="func", c_output_type="double", validate=validate, verbose=verbose)

def _nb_formulagrad_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    return notebook_source_from_expr(model_obj.G, "formulagrad", model_obj, model_c_idx, wks_obj, expr_type="grad", c_output_type="void", validate=validate, verbose=verbose)

def _nb_formulafused_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    """Fused energy+gradient+Hessian device function (one shared CSE pool).

    Values are bit-identical to the separate formulaobj/formulagrad/
    formulahess functions (CSE never reassociates arithmetic, and
    fix_piecewise_zeros only folds all-zero piecewise branches to 0).
    """
    result = notebook_source_from_expr_cse(
        _realify_eps_complex(_zero_undefined_symbols(model_obj.G, wks_obj)),
        "fused", model_obj, model_c_idx, wks_obj,
        expr_type="fused", c_output_type="void", validate=validate, verbose=verbose)
    return fix_piecewise_zeros(result)


def _nb_formulahess_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    result = notebook_source_from_expr(model_obj.G, "formulahess", model_obj, model_c_idx, wks_obj,
                                       expr_type="hess", c_output_type="void", validate=validate,
                                       verbose=verbose)
    return fix_piecewise_zeros(result)


def _nb_formulaparamgrad_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    """dG/dp for each fit parameter (formula-unit G, matching formulagrad)."""
    params = _fit_parameter_symbols(wks_obj)
    exprs = [model_obj.G.diff(p) for p in params]
    return notebook_source_from_expr(exprs, "formulaparamgrad", model_obj, model_c_idx, wks_obj, expr_type="func", c_output_type="void", validate=validate, verbose=verbose)

def _nb_formulaparammixed_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    """d2G/(dy_j dp): j-major flat layout out[j*n_params + p], site-fraction
    order matching formulagrad's (get_ordered_symbols_for_diff minus T)."""
    params = _fit_parameter_symbols(wks_obj)
    ysyms = get_ordered_symbols_for_diff(model_obj, wks_obj)[1:]
    exprs = [model_obj.G.diff(y).diff(p) for y in ysyms for p in params]
    return notebook_source_from_expr(exprs, "formulaparammixed", model_obj, model_c_idx, wks_obj, expr_type="func", c_output_type="void", validate=validate, verbose=verbose)

def _nb_internal_cons_func_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    # Generate internal constraints function
    constraints = model_obj.get_internal_constraints()
    if not constraints:
        fname = notebook_model_c_func_name_prefix(model_c_idx) + "internal_cons_func"
        return f"__device__ void {fname}(double* out, const double* x) {{ /* No internal constraints */ }}\n\n"
    
    # Apply scaling factor as done in CPU code
    from pycalphad.core.constants import INTERNAL_CONSTRAINT_SCALING
    constraints = [INTERNAL_CONSTRAINT_SCALING * c for c in constraints]
    
    return notebook_source_from_expr(constraints, "internal_cons_func", model_obj, model_c_idx, wks_obj, expr_type="func", c_output_type="void", validate=validate, verbose=verbose)

def _nb_internal_cons_jac_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    # Generate internal constraints Jacobian
    constraints = model_obj.get_internal_constraints()
    if not constraints:
        fname = notebook_model_c_func_name_prefix(model_c_idx) + "internal_cons_jac"
        return f"__device__ void {fname}(double* out, const double* x) {{ /* No internal constraints */ }}\n\n"
    
    # Apply scaling factor as done in CPU code
    from pycalphad.core.constants import INTERNAL_CONSTRAINT_SCALING
    constraints = [INTERNAL_CONSTRAINT_SCALING * c for c in constraints]
    
    return notebook_source_from_expr(constraints, "internal_cons_jac", model_obj, model_c_idx, wks_obj, expr_type="grad", c_output_type="void", validate=validate, verbose=verbose)

def _nb_mass_obj_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    # Output indexed by NONVACANT components only, matching CPU PhaseRecord._masses
    # (indexed by nonvacant_elements). VA must be SKIPPED, not emitted as a zero
    # slot: VA is not guaranteed to sort last (e.g. AL < VA < ZN), and a zero slot
    # in the middle shifts every later component's index off by one.
    mass_funcs = []
    for comp in wks_obj.components:
        if comp.name == 'VA':
            continue
        mass_funcs.append(model_obj.moles(comp))
    return notebook_source_from_expr(mass_funcs, "mass_obj", model_obj, model_c_idx, wks_obj, expr_type="func", c_output_type="void", validate=validate, verbose=verbose)

def _nb_formulamole_obj_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    # Output indexed by NONVACANT components only, matching CPU
    # PhaseRecord._formulamoles (indexed by nonvacant_elements) and the
    # formulamole_grad generator below. VA must be SKIPPED, not emitted as a
    # zero slot: VA is not guaranteed to sort last (e.g. AL < VA < ZN), and a
    # zero slot in the middle shifts every later component's index off by one.
    import symengine
    funcs = []
    for comp in wks_obj.components:
        comp_name = comp.name
        if comp_name == 'VA':
            continue
        if comp_name in model_obj.nonvacant_elements:
            funcs.append(model_obj.moles(comp_name, per_formula_unit=True))
        else:
            # Component not in this phase - zero moles
            funcs.append(symengine.Float(0.0))

    if not funcs:
        fname = notebook_model_c_func_name_prefix(model_c_idx) + "formulamole_obj"
        return f"__device__ void {fname}(double* out, const double* x) {{ /* No components */ }}\n\n"
    return notebook_source_from_expr(funcs, "formulamole_obj", model_obj, model_c_idx, wks_obj, expr_type="func", c_output_type="void", validate=validate, verbose=verbose)

def _nb_formulamole_grad_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    # Generate formulamole gradient only for nonvacant elements (matching CPU)
    # The CPU generates a separate gradient function for each element, but we generate
    # a combined function that outputs gradients for all nonvacant elements
    import symengine
    funcs = []
    
    # Handle dependent site fractions
    # For phases with site fractions that sum to 1, we need to express dependent
    # site fractions in terms of independent ones before taking gradients
    # Get phase from Database via workspace
    from pycalphad import Database
    if hasattr(wks_obj, 'phase_record_factory') and hasattr(wks_obj.phase_record_factory, 'dbf'):
        phase = wks_obj.phase_record_factory.dbf.phases[model_obj.phase_name]
    else:
        # Fallback - assume phase info is available from model
        phase = model_obj
    site_fractions = model_obj.site_fractions
    
    # Build substitution dict for dependent site fractions
    dependent_subs = {}
    constituents_list = getattr(phase, 'constituents', getattr(model_obj, 'constituents', []))
    
    for subl_idx, constituents in enumerate(constituents_list):
        active_in_subl = sorted([c for c in constituents if c in model_obj.components])
        if len(active_in_subl) > 1:
            # This sublattice has multiple components - last one is dependent
            # Express Y(last) = 1 - sum(Y(others))
            independent_sfs = []
            for comp in active_in_subl[:-1]:
                # Use actual site fraction symbol from model
                matching_sf = None
                for sf in model_obj.site_fractions:
                    if sf.sublattice_index == subl_idx and sf.species.name == comp.name:
                        matching_sf = sf
                        break
                if matching_sf:
                    independent_sfs.append(matching_sf)
            
            if independent_sfs:
                dependent_comp = active_in_subl[-1]
                # Find the dependent site fraction symbol
                dependent_sf = None
                for sf in model_obj.site_fractions:
                    if sf.sublattice_index == subl_idx and sf.species.name == dependent_comp.name:
                        dependent_sf = sf
                        break
                
                if dependent_sf:
                    # The dependent site fraction is 1 minus the sum of independent ones
                    dependent_expr = 1 - sum(independent_sfs)
                    dependent_subs[dependent_sf] = dependent_expr
                    
                    if verbose:
                        print(f"[GPU CODEGEN] Sublattice {subl_idx}: dependent {dependent_sf} = 1 - sum({independent_sfs})")
    
    # Rows indexed by SYSTEM nonvacant components (same order as mass_obj /
    # formulamole_obj above), zero row for components absent from this phase.
    nonvacant_comps = [c.name for c in wks_obj.components if c.name != 'VA']
    for el in nonvacant_comps:
        if el in model_obj.nonvacant_elements:
            moles_expr = model_obj.moles(el, per_formula_unit=True)
        else:
            moles_expr = symengine.Float(0.0)
        # Do NOT apply dependent substitutions to match CPU behavior
        # The CPU treats all site fractions as independent variables
        # if dependent_subs:
        #     # Convert to symengine expression and substitute
        #     moles_expr = moles_expr.xreplace(dependent_subs)
        funcs.append(moles_expr)

    # Always print debug info for moles expressions
    if verbose:
        print(f"[GPU CODEGEN] _nb_formulamole_grad_from_model for {model_obj.phase_name}:")
        print(f"  nonvacant components (system order): {nonvacant_comps}")
        print(f"  Number of functions: {len(funcs)}")
        print(f"  Dependent substitutions IDENTIFIED but NOT APPLIED to match CPU: {dependent_subs}")
        for i, el in enumerate(nonvacant_comps):
            print(f"  moles({el}) = {funcs[i]}")
    
    if not funcs:
        fname = notebook_model_c_func_name_prefix(model_c_idx) + "formulamole_grad"
        return f"__device__ void {fname}(double* out, const double* x) {{ /* No nonvacant elements */ }}\n\n"
    
    # The GPU minimizer passes workspace DOF (which includes N, P, T, site_fractions)
    # but the gradient is generated for phase DOF ordering (P, T, site_fractions).
    # This is handled correctly in the minimizer by proper indexing.
    
    return notebook_source_from_expr(funcs, "formulamole_grad", model_obj, model_c_idx, wks_obj, expr_type="grad", c_output_type="void", validate=validate, verbose=verbose)


def _formulamole_hessian_exprs(model_obj: Model, wks_obj: Workspace):
    """Second derivatives of the moles of each nonvacant component per formula
    unit with respect to the site fractions, flattened as
    out[(comp * phase_dof + i) * phase_dof + j], or None when all of them are
    zero. The moles are linear in the site fractions for every model except
    the two-sublattice ionic liquid, whose site counts P and Q depend on them."""
    import symengine
    site_fractions = [s for s in get_ordered_symbols_for_diff(model_obj, wks_obj) if str(s) != 'T']
    site_fraction_set = set(site_fractions)
    zero = symengine.Integer(0)
    exprs, nonzero = [], False
    for el in (c.name for c in wks_obj.components if c.name != 'VA'):
        if el not in model_obj.nonvacant_elements:
            exprs.extend([zero] * len(site_fractions) ** 2)
            continue
        moles = symengine.sympify(model_obj.moles(el, per_formula_unit=True))
        for yi in site_fractions:
            first = moles.diff(yi)
            if not (first.free_symbols & site_fraction_set):
                exprs.extend([zero] * len(site_fractions))
                continue
            for yj in site_fractions:
                second = first.diff(yj)
                nonzero = nonzero or second != zero
                exprs.append(second)
    return exprs if nonzero else None


def _nb_formulamole_hess_from_model(model_obj: Model, model_c_idx: int, wks_obj: Workspace, validate: bool = True, verbose: bool = False) -> str:
    """Site-fraction Hessians of the moles per formula unit (see
    _formulamole_hessian_exprs); empty when they are all zero."""
    exprs = _formulamole_hessian_exprs(model_obj, wks_obj)
    if exprs is None:
        return ""
    return notebook_source_from_expr(exprs, "formulamole_hess", model_obj, model_c_idx, wks_obj, expr_type="func", c_output_type="void", validate=validate, verbose=verbose)


def _unique_models_for_gpu(wks_obj: Workspace, validate: bool = True):
    """Deduplicated Model list + phase-name-to-index map for the GPU pipeline.

    Cheap (no C code generation) so it can run on every call; the expensive
    string generation in _generate_c_code_for_phase_models only needs to run
    on kernel-cache misses.
    """
    unique_py_models = []
    py_phase_name_to_unique_idx_map = {}

    # Build list of unique models (by phase name)
    # IMPORTANT: Sort phases to ensure deterministic ordering for kernel caching
    from pycalphad.model import Model as _PlainModel
    from pycalphad.backend import AcceleratedCapabilityError
    for ph_name in sorted(wks_obj.phases):
        if ph_name not in py_phase_name_to_unique_idx_map:
            _mod = wks_obj.models[ph_name]
            if type(_mod) is not _PlainModel:
                # Custom Model subclasses (ModelMQMQA, user models) are a
                # DECLARED capability limit: the codegen only supports the
                # plain Model's symbolic structure. Declaring it here (before
                # any code generation or compilation) keeps genuine compile
                # failures as hard errors.
                raise AcceleratedCapabilityError(
                    f'accelerated backends support plain Model instances '
                    f'only; phase {ph_name} uses {type(_mod).__name__}')
            py_phase_name_to_unique_idx_map[ph_name] = len(unique_py_models)
            unique_py_models.append(_mod)

    if wks_obj.verbose:
        print(f"[GPU] Phase name to unique index mapping: {py_phase_name_to_unique_idx_map}")

    # Validate workspace-level constraints if validation enabled
    if validate:
        if len(unique_py_models) > _get_c_define("MAX_PHASES"):
            raise CodeValidationError(
                f"Too many unique phases: {len(unique_py_models)} > MAX_PHASES ({_get_c_define('MAX_PHASES')})"
            )

        if len(wks_obj.components) > _get_c_define("MAX_COMPONENTS"):
            raise CodeValidationError(
                f"Too many components: {len(wks_obj.components)} > MAX_COMPONENTS ({_get_c_define('MAX_COMPONENTS')})"
            )

    return unique_py_models, py_phase_name_to_unique_idx_map


def _generate_c_code_for_phase_models(wks_obj: Workspace, include_hess: bool = False, validate: bool = True):
    """
    Generates C __device__ functions for phase properties and
    C code snippets for initializing PhaseRecord structs on the GPU.

    Args:
        validate: Whether to validate generated code (default True)
    """
    if wks_obj.verbose:
        print("[GPU] Generating C code for phase models...")

    validation_warnings = []
    unique_py_models, py_phase_name_to_unique_idx_map = _unique_models_for_gpu(wks_obj, validate)
    # Parameter-denominator Jansson builds (PYCGPU_JANSSON_KIND=2) need dG/dp
    # and d2G/dydp device functions; generated only for those builds (the
    # module source differs, so they land in distinct cache entries).
    _jansson_params = (os.environ.get('PYCGPU_JANSSON_KIND') == '2'
                       and len(_fit_parameter_symbols(wks_obj)) > 0)

    # Generate C functions for all unique models
    all_model_device_functions_c_code = ""
    g_phase_record_array_init_calls_c_code = []

    for model_c_idx, model_obj in enumerate(unique_py_models):
        if wks_obj.verbose:
            print(f"[GPU] Generating functions for model {model_c_idx}: {model_obj.phase_name}")

        try:
            # Generate all required device functions for this model
            all_model_device_functions_c_code += _nb_obj_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            all_model_device_functions_c_code += _nb_formulaobj_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            all_model_device_functions_c_code += _nb_formulagrad_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            
            if include_hess:
                if wks_obj.verbose:
                    print(f"[GPU DEBUG] Generating Hessian for model {model_c_idx}, verbose={wks_obj.verbose}")
                all_model_device_functions_c_code += _nb_formulahess_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
                # Fused energy+grad+hess (hot path of recompute); the solver
                # uses the separate functions when it is absent.
                _fused_ok = not os.environ.get('PYCGPU_NO_FUSED')
                if _fused_ok:
                    try:
                        all_model_device_functions_c_code += _nb_formulafused_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
                    except Exception as _fused_err:
                        _fused_ok = False
                        if wks_obj.verbose:
                            print(f"[GPU] fused generation failed for model {model_c_idx}: {_fused_err}")
                
            all_model_device_functions_c_code += _nb_internal_cons_func_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            all_model_device_functions_c_code += _nb_internal_cons_jac_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            all_model_device_functions_c_code += _nb_mass_obj_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            all_model_device_functions_c_code += _nb_formulamole_obj_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            all_model_device_functions_c_code += _nb_formulamole_grad_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            formulamole_hess_c = _nb_formulamole_hess_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            all_model_device_functions_c_code += formulamole_hess_c
            if _jansson_params:
                all_model_device_functions_c_code += _nb_formulaparamgrad_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
                all_model_device_functions_c_code += _nb_formulaparammixed_from_model(model_obj, model_c_idx, wks_obj, validate, wks_obj.verbose)
            
        except CodeValidationError as e:
            if validate:
                raise CodeValidationError(f"Validation failed for model {model_obj.phase_name}: {e}")
            else:
                validation_warnings.append(f"Model {model_obj.phase_name}: {e}")
                if wks_obj.verbose:
                    print(f"[GPU] Warning: {e}")
                continue

        # Generate PhaseRecord initialization call
        func_prefix = notebook_model_c_func_name_prefix(model_c_idx)
        fn_obj = func_prefix + "obj"
        fn_formulaobj = func_prefix + "formulaobj"
        fn_formulagrad = func_prefix + "formulagrad"
        fn_formulahess = func_prefix + "formulahess" if include_hess else "nullptr"
        fn_icf = func_prefix + "internal_cons_func"
        fn_icj = func_prefix + "internal_cons_jac"
        fn_mass = func_prefix + "mass_obj"
        fn_fmo = func_prefix + "formulamole_obj"
        fn_fmg = func_prefix + "formulamole_grad"

        # Get model metadata
        # Use phase_record_factory's state variables for consistency with CPU
        # The phase record needs to know about ALL state variables, not just ones the model uses
        # The generated functions expect workspace DOF format with ALL state variables
        if hasattr(wks_obj, 'phase_record_factory') and wks_obj.phase_record_factory is not None:
            num_statevars = len(wks_obj.phase_record_factory.state_variables)
        else:
            # Fallback to model's state variables if phase_record_factory not available
            num_statevars = len(model_obj.state_variables)
        phase_dof = len(model_obj.site_fractions)
        num_elements = len(wks_obj.components)
        num_internal_cons = len(model_obj.get_internal_constraints())

        # Count non-vacancy components
        nonvacant_count = sum(1 for comp in wks_obj.components if comp.name != 'VA')
        init_call = f"    g_phase_records_array[{model_c_idx}].init(&{fn_obj}, &{fn_formulaobj}, &{fn_formulagrad}, {fn_formulahess if fn_formulahess == 'nullptr' else '&' + fn_formulahess}, &{fn_icf}, &{fn_icj}, &{fn_mass}, &{fn_fmo}, &{fn_fmg}, {num_statevars}, {phase_dof}, {num_elements}, {num_internal_cons}, {nonvacant_count});\n"
        if _jansson_params:
            init_call += f"    g_phase_records_array[{model_c_idx}].formulaparamgrad = &{func_prefix}formulaparamgrad;\n"
            init_call += f"    g_phase_records_array[{model_c_idx}].formulaparammixed = &{func_prefix}formulaparammixed;\n"
        if include_hess and _fused_ok:
            init_call += f"    g_phase_records_array[{model_c_idx}].formulafused = &{func_prefix}fused;\n"
        if formulamole_hess_c:
            init_call += f"    g_phase_records_array[{model_c_idx}].formulamole_hess = &{func_prefix}formulamole_hess;\n"
        g_phase_record_array_init_calls_c_code.append(init_call)

    return (all_model_device_functions_c_code, g_phase_record_array_init_calls_c_code, 
            unique_py_models, py_phase_name_to_unique_idx_map)




# Kernel headers in dependency order. The LAPACK transliteration chain is
# bitwise-identical to reference LAPACK (leaf -> dbdsqr -> mid -> top).
_SOLVER_HEADERS = ("lapack_leaf.h", "lapack_dbdsqr.h", "lapack_mid.h", "lapack_top.h",
                   "phase_rec.h", "comp_set.h", "hyperplane.h", "minimizer.h",
                   "eqsolver.h", "solve_condition.h")


def solver_headers_source(include_semismooth=False):
    """The solver's kernel headers, inlined in dependency order."""
    names = _SOLVER_HEADERS + (("semismooth.h",) if include_semismooth else ())
    return "\n".join(_read_gpu_header(name) for name in names)


def solver_headers_fingerprint():
    """Hash of everything that shapes generated kernel source."""
    import hashlib
    h = hashlib.md5()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _SOLVER_HEADERS + ("semismooth.h", "gpu_codegen.py"):
        with open(os.path.join(here, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _generate_full_gpu_source(wks_obj: Workspace,
                               model_functions_c_code: str,
                               g_phase_record_array_init_calls_c_code: list,
                               num_unique_models: int):
    """Assemble the CUDA/C++ source for the equilibrium module.

    The per-condition logic lives in solve_condition.h; the kernels here only
    hand each thread its slices of the shared work arrays.
    """
    num_records = num_unique_models if num_unique_models > 0 else 1
    # Semismooth pass-2 solver (PYCGPU_SS=1): opt-in experimental kernel;
    # excluded from the module entirely when off (cache key carries the flag)
    headers = solver_headers_source(include_semismooth=bool(os.environ.get('PYCGPU_SS')))
    full_source = f"""
#if !defined(__CUDACC_RTC__) && !defined(__HIPCC_RTC__)
#include <float.h>
#include <math.h>
#include <stdio.h>
#endif

// Per-thread work-array table (one device pointer per array; each thread
// uses its own slice).
typedef struct WorkArrays {{
    double* arrays[23];
}} WorkArrays;

{headers}

// --- Generated model functions ---
// Strength-reduced pow for the generated model functions (CUDA/HIP only):
// every exponent the codegen emits is a small integer literal, and libdevice
// pow costs ~100+ cycles vs single digits for an inlined multiply chain.
// Accuracy differs from libdevice pow by <=1-2 ulp. The C++ backend is
// excluded so it stays bit-identical to CPU pycalphad. Defined AFTER all
// system headers so the macro only rewrites calls in the generated code.
#if defined(__CUDACC__) || defined(__HIP__)
__device__ inline double pycgpu_pow(double b, double e) {{
    int ei = (int)e;
    if ((double)ei == e && ei > -64 && ei < 64) {{
        unsigned int n = ei < 0 ? (unsigned int)(-ei) : (unsigned int)ei;
        double r = 1.0, p = b;
        while (n) {{ if (n & 1u) r *= p; p *= p; n >>= 1u; }}
        return ei < 0 ? 1.0 / r : r;
    }}
    return (pow)(b, e);
}}
#define pow(b, e) pycgpu_pow((b), (e))
#endif
{model_functions_c_code}

__device__ PhaseRecord g_phase_records_array[{num_records}];

extern "C" {{

__global__ void init_all_gpu_phase_records() {{
{"".join(g_phase_record_array_init_calls_c_code)}
}}

// Grid energy evaluation for calculate(): one thread per sample point, dof
// rows in workspace format [statevars..., Y...].
__global__ void grid_eval_kernel(int model_idx, const double* dof, double* out,
                                 long long n_points, int dof_stride) {{
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_points) return;
    out[i] = g_phase_records_array[model_idx].obj(&dof[i * (long long)dof_stride]);
}}

// Walker-grid energy evaluation: rows are one phase's cached sample dofs
// (T-combo major, column minor, WITHOUT trailing parameter slots); the
// kernel assembles [dof_row..., params...] locally and writes into the
// combo-major GM buffer at (row/ncols)*row_stride + col_start + row%ncols.
__global__ void grid_eval_params_kernel(
    int model_idx, const double* dof, const double* params, int n_params,
    long long n_rows, int dof_stride, double* out_base,
    int ncols, int col_start, long long row_stride)
{{
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_rows) return;
    double x[MAX_STATEVARS + MAX_DOF_PER_PHASE + MAX_PARAMS];
    for (int k = 0; k < dof_stride; ++k) x[k] = dof[i * dof_stride + k];
    for (int k = 0; k < n_params && k < MAX_PARAMS; ++k) x[dof_stride + k] = params[k];
    out_base[(i / ncols) * row_stride + col_start + (i % ncols)] =
        g_phase_records_array[model_idx].obj(x);
}}

// Per-condition lower convex hull (device twin of pycalphad's
// lower_convex_hull/hyperplane): one thread per condition. Each condition
// selects its grid sample via a flat base ROW into the (rows x N) X buffer
// and aligned GM buffer. Chemical potentials are in-out (fixed entries
// pre-set).
__global__ void point_hull_kernel(
    const double* grid_X, const double* grid_GM,
    const long long* x_base_row, const long long* gm_base_row,
    const int* m_points, int num_components,
    const int* fixed_chempot_indices, const int* num_fixed_chempots,
    const double* lincomb_coefs, const double* lincomb_rhs,
    const int* num_lincomb, int max_lincomb,
    double* chemical_potentials, double* out_energy,
    double* result_fractions, int* result_simplex, int n_conditions)
{{
    int t = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (t >= n_conditions) return;
    const double* X = &grid_X[x_base_row[t] * (long long)num_components];
    const double* GM = &grid_GM[gm_base_row[t]];
    out_energy[t] = pycgpu_hyperplane(
        X, GM, m_points[t], num_components,
        &chemical_potentials[(long long)t * num_components],
        &fixed_chempot_indices[(long long)t * num_components], num_fixed_chempots[t],
        &lincomb_coefs[(long long)t * max_lincomb * num_components],
        &lincomb_rhs[(long long)t * max_lincomb], num_lincomb[t],
        &result_fractions[(long long)t * (num_components + 1)],
        &result_simplex[(long long)t * (num_components + 1)]);
}}

// One condition per thread. The debug_* arguments are unused and kept only
// for launch-signature compatibility.
__global__ void top_level_equilibrium_kernel(
    const void* global_spec_ptr_raw,
    const void* condition_args_list_ptr_raw,
    void* results_list_ptr_raw,
    int num_conditions_total,
    int condition_stride,
    int python_max_statevars,
    const void* initial_phase_data_ptr,
    int initial_phase_data_stride,
    int system_spec_stride,
    const void* grid_data_ptr_raw,
    double* debug_gm_history,
    double* debug_mu_history,
    int* debug_convergence_history,
    int* debug_iteration_count,
    int debug_max_steps,
    const WorkArrays* work_arrays,
    const int* grid_block_indices,
    long long grid_block_stride_bytes,
    int max_solver_iterations)
{{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_conditions_total || work_arrays == nullptr || results_list_ptr_raw == nullptr) return;
#ifdef PYCGPU_GUARD_SLICES
    // Memory-safety validation mode (PYCGPU_GUARD=1): threads use even slices
    // of doubled allocations; odd slices hold magic guards scanned afterwards.
    const long long slot = (long long)tid * 2;
#else
    const long long slot = tid;
#endif
    const long long svd_mn = (long long)MAX_SVD_M * MAX_SVD_N;
    const long long svd_nn = (long long)MAX_SVD_N * MAX_SVD_N;
    const long long pm = (long long)MAX_PHASE_MATRIX_DIM * MAX_PHASE_MATRIX_DIM;
    const long long ndof = MAX_STATEVARS + MAX_DOF_PER_PHASE;
    const long long cons = (long long)(MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS)
                           * (MAX_DOF_PER_PHASE + MAX_INTERNAL_CONSTRAINTS);
    const long long pc = (long long)MAX_PHASES * MAX_COMPONENTS;
    double* const* w = work_arrays->arrays;
    PycgpuWork work;
    work.A_lstsq_copy = w[0] + slot * svd_mn;
    work.U_lstsq = w[1] + slot * svd_mn;
    work.V_lstsq = w[2] + slot * svd_nn;
    work.singular_values_lstsq = w[3] + slot * MAX_SVD_N;
    work.superdiag_lstsq = w[4] + slot * MAX_SVD_N;
    work.U_inv = w[5] + slot * pm;
    work.V_inv = w[6] + slot * pm;
    work.singular_values_inv = w[7] + slot * MAX_PHASE_MATRIX_DIM;
    work.superdiag_inv = w[8] + slot * MAX_PHASE_MATRIX_DIM;
    work.work_inv = w[9] + slot * pm;
    work.x_dof = w[10] + slot * ndof;
    work.grad = w[11] + slot * ndof;
    work.hess = w[12] + slot * ndof * ndof;
    work.masses = w[13] + slot * MAX_COMPONENTS;
    work.mass_jac = w[14] + slot * MAX_COMPONENTS * ndof;
    work.phase_matrix = w[15] + slot * cons;
    work.equilibrium_matrix = w[16] + slot * MAX_EQ_MATRIX_SIZE;
    work.equilibrium_rhs = w[17] + slot * MAX_EQ_MATRIX_ROWS;
    work.eq_soln = w[18] + slot * MAX_EQ_SOLN_LEN;
    work.system_state = w[19] + slot * SYSTEM_STATE_SIZE;
    work.delta_ms = w[20] + slot * pc;
    work.phase_compositions = w[21] + slot * pc;
    work.phase_amounts_per_mole_atoms = w[22] + slot * pc;
    double* results = (double*)results_list_ptr_raw;
    double* jansson_out = (double*)0;
#ifdef PYCGPU_JANSSON_TARGET
    // Jansson delta block: trailing region of the results buffer, laid out
    // after all per-condition result records.
    jansson_out = results + (long long)num_conditions_total * PYCGPU_RESULTS_PER_CONDITION
                  + (long long)tid * PYJAN_COND_STRIDE;
#endif
    pycgpu_solve_condition(
        tid,
        (const double*)global_spec_ptr_raw, system_spec_stride,
        (const double*)condition_args_list_ptr_raw, condition_stride, python_max_statevars,
        results,
        (const double*)initial_phase_data_ptr, initial_phase_data_stride,
        (const char*)grid_data_ptr_raw, grid_block_indices, grid_block_stride_bytes,
        g_phase_records_array, {num_records},
        work, max_solver_iterations, jansson_out);
}}

}} // extern "C"
"""
    return full_source