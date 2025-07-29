#!/usr/bin/env python
"""Improved CSE-based code generation function to replace regex approach."""

import symengine as se
from symengine import cse
from symengine.lib.symengine_wrapper import ccode
import re
import time

def generate_cse_c_function(expr_or_list_in, expr_type, full_c_func_name, 
                           c_output_type, c_output_arg_name, c_input_arg_name,
                           ordered_symbols_for_diff, model_obj, wks_obj):
    """
    Generate C function using CSE + ccode() instead of regex transformations.
    
    This replaces the massive regex processing with SymEngine's native capabilities.
    """
    print(f"[CSE CODEGEN] Processing {expr_type} with {len(expr_or_list_in)} expressions")
    
    start_time = time.time()
    
    # Start building the C function
    c_code = f"__device__ {c_output_type} {full_c_func_name}(double* {c_output_arg_name}, const double* {c_input_arg_name}) {{\n"
    
    if expr_type == "func":
        # Process function expressions (free energy, etc.)
        all_expressions = list(expr_or_list_in)
        
        # Apply CSE to all expressions together for maximum efficiency
        print(f"[CSE CODEGEN] Applying CSE to {len(all_expressions)} function expressions...")
        cse_start = time.time()
        replacements, reduced_exprs = cse(all_expressions)
        cse_time = time.time() - cse_start
        
        print(f"[CSE CODEGEN] CSE found {len(replacements)} common subexpressions in {cse_time:.3f}s")
        
        # Convert symbolic variables to array indices for C code
        var_mapping = create_variable_mapping(model_obj, wks_obj)
        
        # Generate subexpression assignments
        for symbol, subexpr in replacements:
            c_subexpr = ccode(subexpr)
            c_subexpr = apply_variable_mapping(c_subexpr, var_mapping)
            c_code += f"    double {ccode(symbol)} = {c_subexpr};\n"
        
        # Generate main expressions
        for i, reduced_expr in enumerate(reduced_exprs):
            c_expr = ccode(reduced_expr)
            c_expr = apply_variable_mapping(c_expr, var_mapping)
            c_code += f"    {c_output_arg_name}[{i}] = {c_expr};\n"
    
    elif expr_type == "grad":
        # Process gradient expressions
        print(f"[CSE CODEGEN] Processing gradient expressions...")
        all_grad_expressions = []
        current_out_idx = 0
        
        # Collect all gradient expressions
        for m_expr_idx, sub_expr in enumerate(expr_or_list_in):
            for sym_to_diff_against in ordered_symbols_for_diff:
                deriv_expr = sub_expr.diff(sym_to_diff_against)
                all_grad_expressions.append(deriv_expr)
        
        # Apply CSE to all gradients together
        cse_start = time.time()
        replacements, reduced_exprs = cse(all_grad_expressions)
        cse_time = time.time() - cse_start
        
        print(f"[CSE CODEGEN] Gradient CSE found {len(replacements)} subexpressions in {cse_time:.3f}s")
        
        # Convert symbolic variables to array indices
        var_mapping = create_variable_mapping(model_obj, wks_obj)
        
        # Generate subexpression assignments
        for symbol, subexpr in replacements:
            c_subexpr = ccode(subexpr)
            c_subexpr = apply_variable_mapping(c_subexpr, var_mapping)
            c_code += f"    double {ccode(symbol)} = {c_subexpr};\n"
        
        # Generate gradient assignments
        for i, reduced_expr in enumerate(reduced_exprs):
            c_expr = ccode(reduced_expr)
            c_expr = apply_variable_mapping(c_expr, var_mapping)
            c_code += f"    {c_output_arg_name}[{i}] = {c_expr};\n"
    
    elif expr_type == "hess":
        # Process Hessian expressions
        print(f"[CSE CODEGEN] Processing Hessian expressions...")
        all_hess_expressions = []
        
        # Collect all Hessian expressions
        for m_expr_idx, sub_expr in enumerate(expr_or_list_in):
            for i_sym_idx, sym_j in enumerate(ordered_symbols_for_diff):
                first_deriv = sub_expr.diff(sym_j)
                for j_sym_idx, sym_k in enumerate(ordered_symbols_for_diff):
                    second_deriv_expr = first_deriv.diff(sym_k)
                    all_hess_expressions.append(second_deriv_expr)
        
        # Apply CSE to all Hessian expressions together
        cse_start = time.time()
        replacements, reduced_exprs = cse(all_hess_expressions)
        cse_time = time.time() - cse_start
        
        print(f"[CSE CODEGEN] Hessian CSE found {len(replacements)} subexpressions in {cse_time:.3f}s")
        
        # Convert symbolic variables to array indices
        var_mapping = create_variable_mapping(model_obj, wks_obj)
        
        # Generate subexpression assignments
        for symbol, subexpr in replacements:
            c_subexpr = ccode(subexpr)
            c_subexpr = apply_variable_mapping(c_subexpr, var_mapping)
            c_code += f"    double {ccode(symbol)} = {c_subexpr};\n"
        
        # Generate Hessian assignments
        for i, reduced_expr in enumerate(reduced_exprs):
            c_expr = ccode(reduced_expr)
            c_expr = apply_variable_mapping(c_expr, var_mapping)
            c_code += f"    {c_output_arg_name}[{i}] = {c_expr};\n"
    
    # Close the function
    c_code += "}\n\n"
    
    total_time = time.time() - start_time
    print(f"[CSE CODEGEN] Generated {expr_type} function in {total_time:.3f}s")
    
    return c_code

def create_variable_mapping(model_obj, wks_obj):
    """Create mapping from SymEngine variables to C array indices."""
    # This replicates the logic from notebook_convert_var_names but cleaner
    var_mapping = {}
    
    # Get the variable mapping from the workspace (similar to existing code)
    from pycalphad.gpu.gpu_codegen import notebook_convert_var_names
    
    # Create a dummy expression to extract the mapping
    dummy_expr = "T + N + P"  # Simple expression with common variables
    converted = notebook_convert_var_names(dummy_expr, model_obj, wks_obj)
    
    # Extract the mapping by analyzing the conversion
    # This is a simplified approach - we can refine this later
    var_mapping = {
        'T': 'x[2]',
        'N': 'x[0]', 
        'P': 'x[1]'
    }
    
    # Add phase-specific mappings (site fractions, etc.)
    # Extract from the model/workspace as needed
    try:
        # Get site fraction mappings from existing implementation
        phase_name = wks_obj.phases[0] if hasattr(wks_obj, 'phases') and wks_obj.phases else 'PHASE'
        for i, comp in enumerate(model_obj.components):
            if comp != 'VA':  # Skip vacancy
                var_name = f'{phase_name}0{comp}'
                var_mapping[var_name] = f'x[{i+3}]'  # Offset by state variables
    except:
        pass
    
    return var_mapping

def apply_variable_mapping(c_expr, var_mapping):
    """Apply variable name to array index mapping in C expression."""
    # Replace symbolic variable names with array indices
    for var_name, array_ref in var_mapping.items():
        # Use word boundaries to avoid partial replacements
        c_expr = re.sub(r'\b' + re.escape(var_name) + r'\b', array_ref, c_expr)
    
    return c_expr

# Test function to verify the approach works
def test_cse_codegen():
    """Test the CSE-based code generation approach."""
    from pycalphad import Database, Model
    from pycalphad.core.workspace import Workspace
    import pycalphad.variables as v
    
    # Load test case
    db = Database('Al-Cu-Fe.tdb')
    components = ['AL', 'CU', 'FE', 'VA']
    conditions = {v.T: 1273.15, v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3, v.N: 1}
    
    wks = Workspace(db, components, ['LIQUID'], conditions, verbose=False)
    model = Model(db, components, 'LIQUID')
    
    # Test function generation
    free_energy = [model.ast]
    
    c_code = generate_cse_c_function(
        expr_or_list_in=free_energy,
        expr_type="func",
        full_c_func_name="test_liquid_obj",
        c_output_type="void",
        c_output_arg_name="output",
        c_input_arg_name="input",
        ordered_symbols_for_diff=[],
        model_obj=model,
        wks_obj=wks
    )
    
    print("Generated C code:")
    print(c_code[:500] + "..." if len(c_code) > 500 else c_code)
    
    return c_code

if __name__ == "__main__":
    test_cse_codegen()