"""
GPU Kernel Manager for reliable compilation and caching.

This module provides functions to:
1. Compile GPU kernels once and save them to disk
2. Load pre-compiled kernels from disk
3. Manage kernel versions and compatibility
"""

import os
import pickle
import hashlib
import json
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


class GPUKernelManager:
    """Manages GPU kernel compilation and caching with explicit control."""
    
    def __init__(self, cache_dir: Optional[str] = None, use_script_dir: bool = True):
        """
        Initialize the kernel manager.
        
        Args:
            cache_dir: Directory to store compiled kernels. 
                      If not provided, auto-determines based on use_script_dir.
            use_script_dir: If True (default), save kernels in cwd/.pycgpu_kernels/
                           If False, use central cache (~/.cache/pycalphad/gpu_kernels)
        """
        import sys
        import platform
        
        if cache_dir is None:
            if use_script_dir:
                # Always save kernels in the current working directory under .pycgpu_kernels/
                # This ensures kernels are saved where the script is run from, not where it's located
                cache_dir = os.path.join(os.getcwd(), '.pycgpu_kernels')
            else:
                # Use a central cache location that can be shared across scripts
                if platform.system() == 'Windows':
                    # Windows: Use %LOCALAPPDATA%\pycalphad\gpu_kernels
                    appdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
                    cache_dir = os.path.join(appdata, 'pycalphad', 'gpu_kernels')
                else:
                    # Linux/Mac: Use ~/.cache/pycalphad/gpu_kernels
                    cache_home = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
                    cache_dir = os.path.join(cache_home, 'pycalphad', 'gpu_kernels')
        
        self.cache_dir = cache_dir
        self.kernel_cache_dir = cache_dir  # Use cache_dir directly as the kernel directory
        os.makedirs(self.kernel_cache_dir, exist_ok=True)
        
        # In-memory cache for this session
        self._loaded_modules = {}
        
    def get_kernel_path(self, name: str, version: str = "default", tdb_hash: Optional[str] = None) -> str:
        """Get the full path for a kernel cache file."""
        if tdb_hash:
            filename = f"kernel_{name}_{version}_{tdb_hash[:8]}.ptx"
        else:
            filename = f"kernel_{name}_{version}.ptx"
        return os.path.join(self.kernel_cache_dir, filename)
    
    def get_metadata_path(self, name: str, version: str = "default", tdb_hash: Optional[str] = None) -> str:
        """Get the full path for kernel metadata."""
        if tdb_hash:
            filename = f"kernel_{name}_{version}_{tdb_hash[:8]}.json"
        else:
            filename = f"kernel_{name}_{version}.json"
        return os.path.join(self.kernel_cache_dir, filename)
    
    @staticmethod
    def compute_tdb_hash(tdb_path: str) -> str:
        """Compute hash of a TDB file for cache invalidation."""
        if os.path.exists(tdb_path):
            with open(tdb_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        return "no_tdb"
    
    def compile_and_save(self, 
                        source_code: str,
                        name: str,
                        version: str = "default",
                        compile_options: Optional[Tuple[str, ...]] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        tdb_hash: Optional[str] = None,
                        verbose: bool = False) -> cp.RawModule:
        """
        Compile a GPU kernel and save it to disk for later use.
        
        Args:
            source_code: CUDA source code to compile
            name: Name identifier for the kernel
            version: Version identifier (useful for different phase combinations)
            compile_options: Tuple of compilation options
            metadata: Additional metadata to save with the kernel
            verbose: Print compilation information
            
        Returns:
            Compiled CuPy RawModule
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available")
        
        if verbose:
            print(f"[GPU KernelManager] Compiling kernel '{name}' version '{version}'...")
            
        # Default compile options if not provided
        if compile_options is None:
            compile_options = (
                '-std=c++11',
                '-O3',
                # AMD-compatible: removed NVIDIA-specific flags (--use_fast_math, -Xptxas)
            )
        
        # Compile the kernel
        try:
            module = cp.RawModule(
                code=source_code,
                options=compile_options,
                backend='nvcc'
            )
            
            if verbose:
                print(f"[GPU KernelManager] Compilation successful!")
            
            # Save the source code and metadata
            kernel_path = self.get_kernel_path(name, version, tdb_hash)
            metadata_path = self.get_metadata_path(name, version, tdb_hash)
            
            # Save source code (PTX is embedded in the module, we save source for recompilation)
            source_path = kernel_path.replace('.ptx', '.cu')
            with open(source_path, 'w') as f:
                f.write(source_code)
            
            # Save compilation options and metadata
            meta = {
                'name': name,
                'version': version,
                'compile_options': list(compile_options),
                'timestamp': datetime.now().isoformat(),
                'source_hash': hashlib.md5(source_code.encode()).hexdigest(),
                'source_path': source_path,
                'tdb_hash': tdb_hash,
            }
            if metadata:
                meta['user_metadata'] = metadata
                
            with open(metadata_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            if verbose:
                print(f"[GPU KernelManager] Kernel saved to: {source_path}")
                print(f"[GPU KernelManager] Metadata saved to: {metadata_path}")
            
            # Cache in memory
            if tdb_hash:
                cache_key = f"{name}_{version}_{tdb_hash[:8]}"
            else:
                cache_key = f"{name}_{version}"
            self._loaded_modules[cache_key] = module
            
            # Run cache cleanup if configured
            try:
                from .cache_config import get_cache_config
                config = get_cache_config(self.kernel_cache_dir)
                config.cleanup_old_kernels()
            except Exception as e:
                if verbose:
                    print(f"[GPU KernelManager] Warning: Could not run cache cleanup: {e}")
            
            return module
            
        except Exception as e:
            if verbose:
                print(f"[GPU KernelManager] Compilation failed: {e}")
            raise
    
    def load(self, 
             name: str,
             version: str = "default",
             tdb_hash: Optional[str] = None,
             verbose: bool = False) -> Optional[cp.RawModule]:
        """
        Load a pre-compiled kernel from disk.
        
        Args:
            name: Name identifier for the kernel
            version: Version identifier
            verbose: Print loading information
            
        Returns:
            Compiled CuPy RawModule if found, None otherwise
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available")
        
        if tdb_hash:
            cache_key = f"{name}_{version}_{tdb_hash[:8]}"
        else:
            cache_key = f"{name}_{version}"
        
        # Check in-memory cache first
        if cache_key in self._loaded_modules:
            if verbose:
                print(f"[GPU KernelManager] Using in-memory cached kernel '{name}' version '{version}'")
            return self._loaded_modules[cache_key]
        
        # Check if files exist
        metadata_path = self.get_metadata_path(name, version, tdb_hash)
        
        if not os.path.exists(metadata_path):
            if verbose:
                print(f"[GPU KernelManager] No cached kernel found for '{name}' version '{version}'")
            return None
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            
            source_path = meta.get('source_path')
            if not source_path or not os.path.exists(source_path):
                if verbose:
                    print(f"[GPU KernelManager] Source file not found: {source_path}")
                return None
            
            # Load and recompile the source
            # (CuPy doesn't support loading pre-compiled PTX directly in RawModule)
            with open(source_path, 'r') as f:
                source_code = f.read()
            
            # Verify source hasn't changed
            current_hash = hashlib.md5(source_code.encode()).hexdigest()
            if current_hash != meta.get('source_hash'):
                if verbose:
                    print(f"[GPU KernelManager] Source code has changed, recompilation needed")
                return None
            
            compile_options = tuple(meta.get('compile_options', []))
            
            if verbose:
                print(f"[GPU KernelManager] Recompiling cached kernel '{name}' version '{version}'...")
                print(f"[GPU KernelManager] Original compilation date: {meta.get('timestamp')}")
            
            # Recompile from cached source
            module = cp.RawModule(
                code=source_code,
                options=compile_options,
                backend='nvcc'
            )
            
            if verbose:
                print(f"[GPU KernelManager] Successfully loaded and recompiled kernel!")
            
            # Cache in memory
            self._loaded_modules[cache_key] = module
            
            return module
            
        except Exception as e:
            if verbose:
                print(f"[GPU KernelManager] Failed to load kernel: {e}")
            return None
    
    def exists(self, name: str, version: str = "default", tdb_hash: Optional[str] = None) -> bool:
        """Check if a kernel exists in the cache."""
        metadata_path = self.get_metadata_path(name, version, tdb_hash)
        return os.path.exists(metadata_path)
    
    def list_kernels(self) -> Dict[str, Dict[str, Any]]:
        """List all cached kernels with their metadata."""
        kernels = {}
        
        if not os.path.exists(self.kernel_cache_dir):
            return kernels
        
        for filename in os.listdir(self.kernel_cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.kernel_cache_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        meta = json.load(f)
                    kernel_id = f"{meta['name']}_{meta['version']}"
                    kernels[kernel_id] = meta
                except:
                    pass
        
        return kernels
    
    def clear_cache(self, name: Optional[str] = None, version: Optional[str] = None):
        """
        Clear cached kernels.
        
        Args:
            name: If provided, only clear kernels with this name
            version: If provided with name, only clear this specific version
        """
        import shutil
        
        if name is None:
            # Clear all
            if os.path.exists(self.kernel_cache_dir):
                shutil.rmtree(self.kernel_cache_dir)
                os.makedirs(self.kernel_cache_dir, exist_ok=True)
            self._loaded_modules.clear()
        elif version is None:
            # Clear all versions of a specific kernel
            pattern = f"kernel_{name}_"
            for filename in os.listdir(self.kernel_cache_dir):
                if filename.startswith(pattern):
                    os.remove(os.path.join(self.kernel_cache_dir, filename))
            # Clear from memory cache
            self._loaded_modules = {k: v for k, v in self._loaded_modules.items() 
                                   if not k.startswith(f"{name}_")}
        else:
            # Clear specific version
            for ext in ['.cu', '.json', '.ptx']:
                filepath = self.get_kernel_path(name, version).replace('.ptx', ext)
                if os.path.exists(filepath):
                    os.remove(filepath)
            # Clear from memory cache
            cache_key = f"{name}_{version}"
            self._loaded_modules.pop(cache_key, None)


# Global instance for convenience
_default_manager = None

def get_kernel_manager(cache_dir: Optional[str] = None, use_script_dir: bool = True) -> GPUKernelManager:
    """Get or create the default kernel manager.
    
    Args:
        cache_dir: Directory to store kernels. If None, auto-determines.
        use_script_dir: If True (default), save in script_dir/pycgpu_kernels/
                       If False, use central cache.
    """
    # Always create a new manager - let it handle the cache directory logic
    return GPUKernelManager(cache_dir, use_script_dir)