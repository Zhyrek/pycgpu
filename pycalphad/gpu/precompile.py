"""
Pre-compilation utilities for GPU kernels.

This module provides functions to pre-compile GPU kernels for specific
database and phase combinations, allowing for faster execution later.
"""

import hashlib
from typing import List, Optional, Dict, Any
import os

from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.property_framework.units import as_quantity
from collections import OrderedDict

from .kernel_manager import get_kernel_manager
from .gpu_codegen import (
    _generate_c_code_for_phase_models,
    _generate_full_gpu_source,
    compute_dynamic_kernel_sizes
)
from .nvcc_true_separate import compile_to_combined_source

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


def precompile_equilibrium_kernel(
    dbf: Database,
    comps: List[str],
    phases: List[str],
    cache_dir: Optional[str] = None,
    kernel_name: Optional[str] = None,
    tdb_path: Optional[str] = None,
    use_script_dir: bool = True,
    verbose: bool = True
) -> str:
    """
    Pre-compile an equilibrium kernel for a specific database and phase combination.
    
    This function compiles the GPU kernel and saves it to disk for later use.
    It returns a kernel identifier that can be used to load the kernel later.
    
    Args:
        dbf: Thermodynamic database
        comps: List of component names
        phases: List of phase names to include
        cache_dir: Directory to store the compiled kernel (if None, auto-determines)
        kernel_name: Custom name for the kernel (defaults to auto-generated)
        tdb_path: Path to TDB file for hash-based cache invalidation
        use_script_dir: If True (default), save in cwd/.pycgpu_kernels/
        verbose: Print compilation progress
        
    Returns:
        Kernel identifier string (name_version) that can be used to load the kernel
        
    Example:
        >>> # Pre-compile once
        >>> kernel_id = precompile_equilibrium_kernel(dbf, comps, phases)
        >>> 
        >>> # Later, in parallel workers
        >>> result = equilibrium(dbf, comps, phases, conditions,
        ...                     gpu=True, precompiled_kernel=kernel_id)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for pre-compilation")
    
    if verbose:
        print(f"[GPU Precompile] Starting kernel pre-compilation...")
        print(f"[GPU Precompile] Components: {comps}")
        print(f"[GPU Precompile] Phases ({len(phases)}): {phases}")
    
    # Create a minimal workspace just for code generation
    # We use dummy conditions since we only need the model code
    dummy_conditions = OrderedDict()
    dummy_conditions['T'] = 1000.0
    dummy_conditions['P'] = 101325.0
    
    # Create workspace
    wks = Workspace(
        database=dbf,
        components=comps,
        phases=phases,
        conditions=dummy_conditions,
        models=None,
        parameters=None,
        verbose=False
    )
    
    # Generate phase model code
    if verbose:
        print("[GPU Precompile] Generating phase model code...")
    
    result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    if len(result) == 5:
        model_funcs_c, pr_init_calls_c, unique_py_models, py_phase_name_to_unique_idx_map, individual_phase_codes = result
    else:
        model_funcs_c, pr_init_calls_c, unique_py_models, py_phase_name_to_unique_idx_map = result
        individual_phase_codes = None
    
    num_unique_models = len(unique_py_models)
    
    # Compute dynamic kernel sizes
    dynamic_sizes = compute_dynamic_kernel_sizes(wks)
    
    # Generate kernel name based on components and phases for better sharing
    if kernel_name is None:
        # Use a generic name that allows sharing across different scripts/TDBs
        # The uniqueness comes from the hash of phases+components+model code
        kernel_name = "equilibrium"
    
    # Version based on ALPHABETIZED components and phases for deterministic hashing
    # Sort everything to ensure identical hashes regardless of input order
    sorted_comps = sorted(comps)
    sorted_phases = sorted(phases)
    
    # Create a deterministic string representation
    version_input = (
        'COMPONENTS:' + ','.join(sorted_comps) + '|' +
        'PHASES:' + ','.join(sorted_phases) + '|' +
        'MODELS:' + model_funcs_c  # Full model code for accurate versioning
    )
    version_hash = hashlib.sha256(version_input.encode()).hexdigest()[:12]  # Use longer hash for uniqueness
    
    if verbose:
        print(f"[GPU Precompile] Kernel name: {kernel_name}")
        print(f"[GPU Precompile] Version hash: {version_hash}")
        print(f"[GPU Precompile] Components (sorted): {sorted_comps}")
        print(f"[GPU Precompile] Phases (sorted): {sorted_phases[:5]}..." if len(sorted_phases) > 5 else f"[GPU Precompile] Phases (sorted): {sorted_phases}")
    
    # Get kernel manager
    manager = get_kernel_manager(cache_dir, use_script_dir)
    
    # Compute TDB hash if path provided
    tdb_hash = None
    if tdb_path:
        tdb_hash = manager.compute_tdb_hash(tdb_path)
        if verbose:
            print(f"[GPU Precompile] TDB hash: {tdb_hash[:8]}")
    else:
        # If no TDB path provided, use a hash of the model code as a proxy
        # This ensures different TDBs still get different kernels
        tdb_hash = hashlib.sha256(model_funcs_c.encode()).hexdigest()
        if verbose:
            print(f"[GPU Precompile] Model-based hash: {tdb_hash[:8]}")
    
    # Check if already compiled
    if manager.exists(kernel_name, version_hash, tdb_hash):
        if verbose:
            print(f"[GPU Precompile] Kernel already exists, loading to verify...")
        
        # Try to load to verify it works
        module = manager.load(kernel_name, version_hash, tdb_hash, verbose=False)
        if module is not None:
            if verbose:
                print(f"[GPU Precompile] Existing kernel verified successfully!")
            if tdb_hash:
                return f"{kernel_name}_{version_hash}_{tdb_hash[:8]}"
            else:
                return f"{kernel_name}_{version_hash}"
    
    # Generate full kernel source
    if verbose:
        print(f"[GPU Precompile] Generating full kernel source...")
    
    # Decide compilation strategy based on phase count
    if num_unique_models >= 14 and individual_phase_codes is not None:
        # Use separate compilation for many phases
        if verbose:
            print(f"[GPU Precompile] Using separate compilation for {num_unique_models} phases...")
        
        pr_init_calls_str = ''.join(pr_init_calls_c)
        kernel_source = compile_to_combined_source(
            individual_phase_codes,
            pr_init_calls_str,
            dynamic_sizes,
            verbose=verbose,
            cache_key=None  # Don't use internal caching
        )
    else:
        # Standard compilation
        kernel_source = _generate_full_gpu_source(
            wks, model_funcs_c, pr_init_calls_c, num_unique_models
        )
    
    # Prepare compilation options based on phase count
    if num_unique_models <= 5:
        opt_level = '-O3'
        extra_opts = []  # AMD-compatible: removed NVIDIA-specific flags
    elif num_unique_models <= 10:
        opt_level = '-O2'
        extra_opts = []  # AMD-compatible: removed NVIDIA-specific flags
    else:
        opt_level = '-O1'
        extra_opts = []  # AMD-compatible: removed NVIDIA-specific flags
    
    # Add dynamic size definitions
    define_flags = []
    for define_name, value in dynamic_sizes.items():
        define_flags.append(f'-D{define_name}={value}')
    
    compile_options = tuple(['-std=c++11', opt_level] + extra_opts + define_flags)
    
    # Metadata to save
    metadata = {
        'components': comps,
        'phases': phases,
        'num_phases': len(phases),
        'num_unique_models': num_unique_models,
        'dynamic_sizes': dynamic_sizes,
        'optimization_level': opt_level,
        'tdb_path': tdb_path if tdb_path else 'unknown'
    }
    
    # Compile and save
    if verbose:
        print(f"[GPU Precompile] Compiling kernel ({len(kernel_source)} bytes)...")
    
    try:
        module = manager.compile_and_save(
            kernel_source,
            kernel_name,
            version_hash,
            compile_options,
            metadata,
            tdb_hash,
            verbose=verbose
        )
        
        # Verify the kernel has the expected functions
        try:
            init_func = module.get_function("init_all_gpu_phase_records")
            kernel_func = module.get_function("top_level_equilibrium_kernel")
            if verbose:
                print(f"[GPU Precompile] Kernel functions verified successfully!")
        except Exception as e:
            raise RuntimeError(f"Compiled kernel missing expected functions: {e}")
        
        if tdb_hash:
            kernel_id = f"{kernel_name}_{version_hash}_{tdb_hash[:8]}"
        else:
            kernel_id = f"{kernel_name}_{version_hash}"
        
        if verbose:
            print(f"[GPU Precompile] Success! Kernel ID: {kernel_id}")
            print(f"[GPU Precompile] Kernel saved to: {manager.kernel_cache_dir}")
        
        return kernel_id
        
    except Exception as e:
        if verbose:
            print(f"[GPU Precompile] Compilation failed: {e}")
        raise


def list_precompiled_kernels(cache_dir: Optional[str] = None, use_script_dir: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    List all pre-compiled kernels in the cache directory.
    
    Args:
        cache_dir: Directory containing cached kernels (if None, auto-determines)
        use_script_dir: If True (default), look in cwd/.pycgpu_kernels/
        
    Returns:
        Dictionary mapping kernel IDs to their metadata
    """
    manager = get_kernel_manager(cache_dir, use_script_dir)
    return manager.list_kernels()


def clear_precompiled_kernels(
    cache_dir: Optional[str] = None,
    kernel_name: Optional[str] = None,
    version: Optional[str] = None
):
    """
    Clear pre-compiled kernels from cache.
    
    Args:
        cache_dir: Directory containing cached kernels
        kernel_name: If provided, only clear kernels with this name
        version: If provided with name, only clear this specific version
    """
    manager = get_kernel_manager(cache_dir)
    manager.clear_cache(kernel_name, version)