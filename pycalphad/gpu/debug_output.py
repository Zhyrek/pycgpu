"""
Debug output utilities for CPU-GPU comparison
"""
import os
import numpy as np
from datetime import datetime

# Global debug file handle
_debug_file = None
_debug_enabled = False
_file_initialized = False  # Track if file has been initialized for this session
_cpu_buffer = []  # Buffer for CPU debug messages
_gpu_buffer = []  # Buffer for GPU debug messages
_current_mode = "CPU"  # Track current mode

def init_debug_output(enabled=False, filename="CPU_VS_GPU_TRACE.txt", mode="CPU"):
    """Initialize debug output file
    
    Args:
        enabled: Whether to enable debug output
        filename: Output filename
        mode: Either "CPU" or "GPU" to indicate which solver is running
    """
    global _debug_file, _debug_enabled, _file_initialized, _current_mode, _cpu_buffer, _gpu_buffer
    _debug_enabled = enabled
    _current_mode = mode
    
    if enabled:
        if mode == "CPU":
            # Reset buffers and prepare for new comparison
            _cpu_buffer = []
            _gpu_buffer = []
            _file_initialized = True
        # Don't open file yet - we'll write everything at the end

def close_debug_output(mode="CPU", filename="CPU_VS_GPU_TRACE.txt"):
    """Close debug output and write interleaved results
    
    Args:
        mode: Either "CPU" or "GPU" to indicate which solver finished
        filename: Output filename
    """
    global _debug_file, _file_initialized, _cpu_buffer, _gpu_buffer
    
    if mode == "GPU" and _debug_enabled:
        # After GPU completes, write interleaved output
        _write_interleaved_output(filename)
        # Reset for next session
        _file_initialized = False
        _cpu_buffer = []
        _gpu_buffer = []

def reset_debug_session():
    """Reset debug session to ensure fresh start for next test"""
    global _file_initialized, _debug_file, _cpu_buffer, _gpu_buffer
    _file_initialized = False
    _cpu_buffer = []
    _gpu_buffer = []
    if _debug_file:
        _debug_file.close()
        _debug_file = None

def debug_log(segment_id, label=None, data=None, condition_idx=None):
    """Log debug information for a specific segment
    
    Can be called in two ways:
    1. debug_log(segment_number, label, ...)
    2. debug_log(message, verbose_flag) for simple messages
    """
    global _debug_enabled, _current_mode, _cpu_buffer, _gpu_buffer
    if not _debug_enabled:
        return
    
    
    # Handle the case where debug_log is called with just a message and verbose flag
    if isinstance(segment_id, str) and isinstance(label, bool):
        # This is a simple message call: debug_log("message", verbose)
        if label:  # verbose flag
            message = f"{segment_id}\n"
            if _current_mode == "CPU":
                _cpu_buffer.append(("message", message))
            else:
                _gpu_buffer.append(("message", message))
        return
    
    # Build the segment message
    message_parts = []
    
    # Format segment header
    if isinstance(segment_id, int):
        if condition_idx is not None:
            message_parts.append(f"\n[SEGMENT {segment_id:02d}] [CONDITION {condition_idx}] {label}\n")
        else:
            message_parts.append(f"\n[SEGMENT {segment_id:02d}] {label}\n")
    else:
        # Fallback for string segment IDs
        message_parts.append(f"\n[{segment_id}] {label if label else ''}\n")
    
    # Format data based on type
    if data is not None:
        if isinstance(data, dict):
            for key, value in data.items():
                message_parts.append(f"  {key}: {_format_value(value)}\n")
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                message_parts.append(f"  [{i}]: {_format_value(item)}\n")
        else:
            message_parts.append(f"  {_format_value(data)}\n")
    
    # Store in appropriate buffer
    message = "".join(message_parts)
    if _current_mode == "CPU":
        _cpu_buffer.append(("segment", segment_id, message))
    else:
        _gpu_buffer.append(("segment", segment_id, message))

def _format_value(value):
    """Format a value for debug output"""
    if isinstance(value, np.ndarray):
        if value.size <= 10:
            return f"{value.tolist()}"
        else:
            return f"ndarray(shape={value.shape}, dtype={value.dtype}, first_10={value.flat[:10].tolist()}...)"
    elif isinstance(value, (float, np.floating)):
        return f"{value:.15e}"
    elif isinstance(value, (int, np.integer)):
        return f"{value}"
    elif isinstance(value, str):
        return value
    elif hasattr(value, '__len__'):
        if len(value) <= 10:
            return str(value)
        else:
            return f"{type(value).__name__}(len={len(value)}, first_10={list(value)[:10]}...)"
    else:
        return str(value)

def _write_interleaved_output(filename="CPU_VS_GPU_TRACE.txt"):
    """Write interleaved CPU and GPU debug output"""
    global _cpu_buffer, _gpu_buffer
    
    with open(filename, 'w') as f:
        f.write(f"=== INTERLEAVED CPU-GPU EQUILIBRIUM DEBUG TRACE ===\n")
        f.write(f"Started: {datetime.now().isoformat()}\n\n")
        
        # Create segment dictionaries for easy lookup
        cpu_segments = {}
        gpu_segments = {}
        
        for entry in _cpu_buffer:
            if entry[0] == "segment":
                seg_id = entry[1]
                if seg_id not in cpu_segments:
                    cpu_segments[seg_id] = []
                cpu_segments[seg_id].append(entry[2])
            else:  # message
                # Store messages separately
                if "message" not in cpu_segments:
                    cpu_segments["message"] = []
                cpu_segments["message"].append(entry[1])
        
        for entry in _gpu_buffer:
            if entry[0] == "segment":
                seg_id = entry[1]
                if seg_id not in gpu_segments:
                    gpu_segments[seg_id] = []
                gpu_segments[seg_id].append(entry[2])
            else:  # message
                if "message" not in gpu_segments:
                    gpu_segments["message"] = []
                gpu_segments["message"].append(entry[1])
        
        # Find all segment IDs and sort them properly (integers first, then strings)
        all_segment_keys = set(list(cpu_segments.keys()) + list(gpu_segments.keys()))
        integer_segments = sorted([k for k in all_segment_keys if isinstance(k, int)])
        string_segments = sorted([k for k in all_segment_keys if isinstance(k, str)])
        all_segments = integer_segments + string_segments
        
        # Write interleaved output
        for seg_id in all_segments:
            if seg_id == "message":
                continue  # Skip messages for now
                
            f.write(f"\n{'='*80}\n")
            f.write(f"SEGMENT {seg_id:02d} COMPARISON\n")
            f.write(f"{'='*80}\n")
            
            # CPU section
            f.write(f"\n--- CPU ---\n")
            if seg_id in cpu_segments:
                for msg in cpu_segments[seg_id]:
                    f.write(msg)
            else:
                f.write("(No CPU data for this segment)\n")
            
            # GPU section
            f.write(f"\n--- GPU ---\n")
            if seg_id in gpu_segments:
                for msg in gpu_segments[seg_id]:
                    f.write(msg)
            else:
                f.write("(No GPU data for this segment)\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"TRACE COMPLETE - {datetime.now().isoformat()}\n")
        f.write(f"{'='*80}\n")

def debug_log_array_comparison(segment_id, label, cpu_array, gpu_array=None, condition_idx=None):
    """Special function for comparing CPU and GPU arrays"""
    global _debug_enabled, _current_mode, _cpu_buffer, _gpu_buffer
    if not _debug_enabled:
        return
    
    # Build comparison message
    message_parts = []
    
    if isinstance(cpu_array, np.ndarray):
        message_parts.append(f"  CPU array: shape={cpu_array.shape}, dtype={cpu_array.dtype}\n")
        if cpu_array.size <= 20:
            message_parts.append(f"  CPU values: {cpu_array.flatten().tolist()}\n")
        else:
            message_parts.append(f"  CPU first 10: {cpu_array.flat[:10].tolist()}\n")
            message_parts.append(f"  CPU last 10: {cpu_array.flat[-10:].tolist()}\n")
        message_parts.append(f"  CPU min: {np.nanmin(cpu_array):.15e}, max: {np.nanmax(cpu_array):.15e}, mean: {np.nanmean(cpu_array):.15e}\n")
    else:
        message_parts.append(f"  CPU value: {_format_value(cpu_array)}\n")
    
    if gpu_array is not None:
        if isinstance(gpu_array, np.ndarray):
            message_parts.append(f"  GPU array: shape={gpu_array.shape}, dtype={gpu_array.dtype}\n")
            if gpu_array.size <= 20:
                message_parts.append(f"  GPU values: {gpu_array.flatten().tolist()}\n")
            else:
                message_parts.append(f"  GPU first 10: {gpu_array.flat[:10].tolist()}\n")
                message_parts.append(f"  GPU last 10: {gpu_array.flat[-10:].tolist()}\n")
            message_parts.append(f"  GPU min: {np.nanmin(gpu_array):.15e}, max: {np.nanmax(gpu_array):.15e}, mean: {np.nanmean(gpu_array):.15e}\n")
            
            # Compare arrays
            if cpu_array.shape == gpu_array.shape:
                diff = np.abs(cpu_array - gpu_array)
                message_parts.append(f"  Max absolute difference: {np.nanmax(diff):.15e}\n")
                message_parts.append(f"  Mean absolute difference: {np.nanmean(diff):.15e}\n")
                if np.nanmax(diff) > 0.001:
                    message_parts.append(f"  WARNING: Difference exceeds tolerance!\n")
                    # Find indices of largest differences
                    flat_diff = diff.flatten()
                    largest_indices = np.argpartition(flat_diff, -min(5, flat_diff.size))[-min(5, flat_diff.size):]
                    for idx in largest_indices:
                        if flat_diff[idx] > 0.001:
                            message_parts.append(f"    Index {idx}: CPU={cpu_array.flat[idx]:.15e}, GPU={gpu_array.flat[idx]:.15e}, diff={flat_diff[idx]:.15e}\n")
        else:
            message_parts.append(f"  GPU value: {_format_value(gpu_array)}\n")
    
    # Call regular debug_log with the comparison data
    debug_log(segment_id, label, "".join(message_parts), condition_idx)