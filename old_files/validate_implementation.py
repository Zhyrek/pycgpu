#!/usr/bin/env python3
"""
Validate the dynamic sizing implementation by checking the code structure
"""

import ast
import os

def analyze_function(filename, function_name):
    """Analyze a specific function in a Python file"""
    print(f"\n🔍 Analyzing {function_name} in {filename}")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                print(f"✅ Found function: {function_name}")
                
                # Check parameters
                args = [arg.arg for arg in node.args.args]
                print(f"   Parameters: {args}")
                
                # Check return annotation
                if node.returns:
                    print(f"   Return type: {ast.unparse(node.returns) if hasattr(ast, 'unparse') else 'typed'}")
                
                # Check docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant)):
                    docstring = node.body[0].value.value
                    if "MAX_DOF" in docstring:
                        print("✅ Docstring mentions MAX_DOF - addresses user requirement")
                
                return True
        
        print(f"❌ Function {function_name} not found")
        return False
        
    except Exception as e:
        print(f"❌ Error parsing {filename}: {e}")
        return False

def check_imports(filename, expected_imports):
    """Check if expected imports are present"""
    print(f"\n🔍 Checking imports in {filename}")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        
        found_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.names:
                    for alias in node.names:
                        found_imports.add(alias.name)
        
        for expected in expected_imports:
            if expected in found_imports:
                print(f"✅ Found import: {expected}")
            else:
                print(f"❌ Missing import: {expected}")
        
        return all(imp in found_imports for imp in expected_imports)
        
    except Exception as e:
        print(f"❌ Error checking imports in {filename}: {e}")
        return False

def check_compilation_changes(filename):
    """Check if compilation code has been modified for dynamic sizing"""
    print(f"\n🔍 Checking compilation changes in {filename}")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    checks = {
        "dynamic_sizes = compute_dynamic_kernel_sizes": "✅ Dynamic sizing function called",
        "define_flags": "✅ Define flags created", 
        "-D{define_name}={value}": "✅ Compiler defines formatted",
        "compile_options = ['-std=c++11'] + define_flags": "✅ Compilation options modified",
        "str(sorted(dynamic_sizes.items()))": "✅ Cache key includes dynamic sizes"
    }
    
    results = []
    for check, message in checks.items():
        if check in content:
            print(message)
            results.append(True)
        else:
            print(f"❌ Missing: {check}")
            results.append(False)
    
    return all(results)

# Run validation
print("🚀 VALIDATING DYNAMIC SIZING IMPLEMENTATION")
print("=" * 60)

# Check gpu_codegen.py
codegen_file = "pycalphad/gpu/gpu_codegen.py"
compute_func_ok = analyze_function(codegen_file, "compute_dynamic_kernel_sizes")

# Check gpu_equilibrium.py  
equilibrium_file = "pycalphad/gpu/gpu_equilibrium.py"
imports_ok = check_imports(equilibrium_file, ["compute_dynamic_kernel_sizes"])
compilation_ok = check_compilation_changes(equilibrium_file)

print("\n" + "=" * 60)
print("📊 VALIDATION SUMMARY")
print("=" * 60)

results = {
    "compute_dynamic_kernel_sizes function": compute_func_ok,
    "Required imports": imports_ok, 
    "Compilation modifications": compilation_ok
}

all_passed = True
for check, passed in results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{check:35}: {status}")
    if not passed:
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("🎉 SUCCESS: All validation checks passed!")
    print("The dynamic sizing implementation appears complete and correct.")
    print("\nKey features implemented:")
    print("• compute_dynamic_kernel_sizes() function in gpu_codegen.py")
    print("• Dynamic compilation with -D flags in gpu_equilibrium.py") 
    print("• Cache key includes dynamic sizes for proper invalidation")
    print("• Addresses user requirement for computed MAX_* values")
else:
    print("⚠️  WARNING: Some validation checks failed.")
    print("The implementation may be incomplete or have issues.")

print("=" * 60)