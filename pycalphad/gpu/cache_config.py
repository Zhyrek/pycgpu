"""
GPU Kernel Cache Configuration and Management

This module provides configuration management for GPU kernel caches,
including automatic cleanup of old kernels and cache size limits.
"""

import os
import json
import glob
from typing import Dict, Any, Optional, List
from datetime import datetime

DEFAULT_CONFIG = {
    "max_kernels": 20,
    "auto_cleanup": True,
    "cleanup_strategy": "oldest_first",  # or "least_recently_used"
    "verbose": False,
    "description": "Configuration for GPU kernel cache management"
}


class CacheConfig:
    """Manages configuration for a kernel cache directory."""
    
    def __init__(self, cache_dir: str):
        """
        Initialize cache configuration for a directory.
        
        Args:
            cache_dir: Directory containing cached kernels
        """
        self.cache_dir = cache_dir
        self.config_file = os.path.join(cache_dir, "config.json")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"[CacheConfig] Warning: Could not load config: {e}")
                return DEFAULT_CONFIG.copy()
        else:
            # Create default config
            self._save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[CacheConfig] Warning: Could not save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value and save."""
        self.config[key] = value
        self._save_config(self.config)
    
    def cleanup_old_kernels(self):
        """Remove old kernels if cache exceeds max_kernels limit."""
        if not self.get('auto_cleanup', True):
            return
        
        max_kernels = self.get('max_kernels', 20)
        verbose = self.get('verbose', False)
        
        # Find all kernel files
        kernel_files = glob.glob(os.path.join(self.cache_dir, "kernel_*.json"))
        
        if len(kernel_files) <= max_kernels:
            return  # No cleanup needed
        
        # Load metadata for all kernels
        kernel_info = []
        for json_file in kernel_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Get timestamp for sorting
                timestamp_str = data.get('timestamp') or data.get('user_metadata', {}).get('created_at')
                if timestamp_str:
                    try:
                        # Try ISO format first
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        # Fallback to current time if parsing fails
                        timestamp = datetime.now()
                else:
                    # Use file modification time as fallback
                    timestamp = datetime.fromtimestamp(os.path.getmtime(json_file))
                
                kernel_info.append({
                    'json_file': json_file,
                    'cu_file': json_file.replace('.json', '.cu'),
                    'timestamp': timestamp,
                    'name': data.get('name', 'unknown'),
                    'version': data.get('version', 'unknown')
                })
            except Exception as e:
                if verbose:
                    print(f"[CacheConfig] Warning: Could not read {json_file}: {e}")
        
        # Sort by timestamp (oldest first)
        kernel_info.sort(key=lambda x: x['timestamp'])
        
        # Remove oldest kernels
        num_to_remove = len(kernel_info) - max_kernels
        if num_to_remove > 0:
            if verbose:
                print(f"[CacheConfig] Removing {num_to_remove} old kernel(s) (keeping {max_kernels})")
            
            for kernel in kernel_info[:num_to_remove]:
                try:
                    # Remove both .json and .cu files
                    if os.path.exists(kernel['json_file']):
                        os.remove(kernel['json_file'])
                    if os.path.exists(kernel['cu_file']):
                        os.remove(kernel['cu_file'])
                    
                    if verbose:
                        print(f"[CacheConfig] Removed kernel {kernel['name']}_{kernel['version']} "
                              f"(created {kernel['timestamp'].isoformat()})")
                except Exception as e:
                    if verbose:
                        print(f"[CacheConfig] Warning: Could not remove kernel files: {e}")
    
    def list_kernels(self) -> List[Dict[str, Any]]:
        """List all kernels in cache with their metadata."""
        kernel_files = glob.glob(os.path.join(self.cache_dir, "kernel_*.json"))
        kernels = []
        
        for json_file in kernel_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Get file size
                cu_file = json_file.replace('.json', '.cu')
                if os.path.exists(cu_file):
                    size_kb = os.path.getsize(cu_file) / 1024
                else:
                    size_kb = 0
                
                kernels.append({
                    'file': os.path.basename(json_file),
                    'name': data.get('name', 'unknown'),
                    'version': data.get('version', 'unknown'),
                    'components': data.get('user_metadata', {}).get('components', []),
                    'phases': data.get('user_metadata', {}).get('phases', []),
                    'num_phases': data.get('user_metadata', {}).get('num_phases', 0),
                    'created_at': data.get('user_metadata', {}).get('created_at') or data.get('timestamp'),
                    'compilation_time': data.get('user_metadata', {}).get('compilation_time_seconds'),
                    'size_kb': size_kb
                })
            except Exception:
                pass
        
        # Sort by creation time (newest first)
        kernels.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return kernels


def get_cache_config(cache_dir: str) -> CacheConfig:
    """Get or create cache configuration for a directory."""
    return CacheConfig(cache_dir)