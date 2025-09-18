#!/usr/bin/env python
"""
GPU Kernel Cache Management Utility

Usage:
    python -m pycalphad.gpu.manage_cache [directory] [command] [options]
    
Commands:
    list      - List all cached kernels
    clean     - Remove old kernels (keeping max_kernels newest)
    config    - Show or modify configuration
    info      - Show detailed information about a kernel
    
Examples:
    python -m pycalphad.gpu.manage_cache . list
    python -m pycalphad.gpu.manage_cache . config --set max_kernels=10
    python -m pycalphad.gpu.manage_cache . clean

Note: Kernels are stored in .pycgpu_kernels/ in the specified directory
"""

import os
import sys
import json
import argparse
from typing import Optional
from .cache_config import get_cache_config


def list_kernels(cache_dir: str, verbose: bool = False):
    """List all kernels in the cache."""
    config = get_cache_config(cache_dir)
    kernels = config.list_kernels()
    
    if not kernels:
        print(f"No kernels found in {cache_dir}")
        return
    
    print(f"\nKernels in {cache_dir}:")
    print("-" * 80)
    
    total_size = 0
    for i, kernel in enumerate(kernels, 1):
        print(f"\n{i}. {kernel['file']}")
        print(f"   Version: {kernel['version']}")
        print(f"   Components: {', '.join(kernel['components']) if kernel['components'] else 'unknown'}")
        print(f"   Phases ({kernel['num_phases']}): {', '.join(kernel['phases'][:5]) if kernel['phases'] else 'unknown'}", end='')
        if kernel['phases'] and len(kernel['phases']) > 5:
            print(f"... and {len(kernel['phases']) - 5} more")
        else:
            print()
        print(f"   Created: {kernel['created_at']}")
        if kernel['compilation_time']:
            print(f"   Compilation time: {kernel['compilation_time']:.2f} seconds")
        print(f"   Size: {kernel['size_kb']:.1f} KB")
        total_size += kernel['size_kb']
    
    print("\n" + "-" * 80)
    print(f"Total: {len(kernels)} kernels, {total_size/1024:.1f} MB")
    
    # Show config
    print(f"\nConfiguration (from {os.path.join(cache_dir, 'config.json')}):")
    print(f"  Max kernels: {config.get('max_kernels')}")
    print(f"  Auto cleanup: {config.get('auto_cleanup')}")
    print(f"  Cleanup strategy: {config.get('cleanup_strategy')}")


def clean_cache(cache_dir: str, force: bool = False):
    """Clean old kernels from cache."""
    config = get_cache_config(cache_dir)
    
    if not force:
        kernels = config.list_kernels()
        max_kernels = config.get('max_kernels', 20)
        
        if len(kernels) <= max_kernels:
            print(f"Cache has {len(kernels)} kernels (max: {max_kernels}). No cleanup needed.")
            return
        
        print(f"This will remove {len(kernels) - max_kernels} oldest kernel(s).")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cleanup cancelled.")
            return
    
    # Force verbose during manual cleanup
    old_verbose = config.get('verbose')
    config.set('verbose', True)
    
    config.cleanup_old_kernels()
    
    # Restore original verbose setting
    config.set('verbose', old_verbose)
    
    print("Cleanup complete.")


def show_config(cache_dir: str, key: Optional[str] = None):
    """Show configuration."""
    config = get_cache_config(cache_dir)
    
    if key:
        value = config.get(key)
        print(f"{key}: {value}")
    else:
        print(f"\nConfiguration for {cache_dir}:")
        print(json.dumps(config.config, indent=2))


def set_config(cache_dir: str, key: str, value: str):
    """Set configuration value."""
    config = get_cache_config(cache_dir)
    
    # Convert value to appropriate type
    if value.lower() in ['true', 'false']:
        value = value.lower() == 'true'
    elif value.isdigit():
        value = int(value)
    
    old_value = config.get(key)
    config.set(key, value)
    
    print(f"Updated {key}: {old_value} -> {value}")
    print(f"Configuration saved to {config.config_file}")


def show_kernel_info(cache_dir: str, kernel_name: str):
    """Show detailed information about a specific kernel."""
    json_file = os.path.join(cache_dir, kernel_name)
    if not json_file.endswith('.json'):
        json_file += '.json'
    
    if not os.path.exists(json_file):
        print(f"Kernel not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nKernel Information: {os.path.basename(json_file)}")
    print("=" * 80)
    print(json.dumps(data, indent=2))
    
    # Check if source file exists
    cu_file = json_file.replace('.json', '.cu')
    if os.path.exists(cu_file):
        size_kb = os.path.getsize(cu_file) / 1024
        print(f"\nSource file: {cu_file} ({size_kb:.1f} KB)")


def main():
    """Main entry point for cache management utility."""
    parser = argparse.ArgumentParser(description='GPU Kernel Cache Management')
    parser.add_argument('directory', nargs='?', default='.',
                      help='Cache directory (default: current directory)')
    parser.add_argument('command', choices=['list', 'clean', 'config', 'info'],
                      help='Command to execute')
    parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'),
                      help='Set configuration value')
    parser.add_argument('--get', metavar='KEY',
                      help='Get configuration value')
    parser.add_argument('--kernel', metavar='NAME',
                      help='Kernel name for info command')
    parser.add_argument('--force', action='store_true',
                      help='Force cleanup without confirmation')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine cache directory
    if args.directory == '.':
        cache_dir = os.path.join(os.getcwd(), '.pycgpu_kernels')
    else:
        cache_dir = os.path.join(args.directory, '.pycgpu_kernels')
    
    if not os.path.exists(cache_dir) and args.command != 'config':
        print(f"Cache directory not found: {cache_dir}")
        print("Run GPU equilibrium calculations to create cache, or use 'config' command to initialize.")
        return 1
    
    # Execute command
    if args.command == 'list':
        list_kernels(cache_dir, args.verbose)
    elif args.command == 'clean':
        clean_cache(cache_dir, args.force)
    elif args.command == 'config':
        if args.set:
            set_config(cache_dir, args.set[0], args.set[1])
        elif args.get:
            show_config(cache_dir, args.get)
        else:
            show_config(cache_dir)
    elif args.command == 'info':
        if not args.kernel:
            print("Error: --kernel NAME required for info command")
            return 1
        show_kernel_info(cache_dir, args.kernel)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())