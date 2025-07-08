#!/usr/bin/env python
"""
Script to filter debug output by segment range.

Usage: python cull_debug_info.py <start_segment> <end_segment> [input_file] [output_file]

Arguments:
  start_segment: Starting segment number (inclusive)
  end_segment: Ending segment number (inclusive)
  input_file: Input debug file (default: CPU_VS_GPU_TRACE.txt)
  output_file: Output filtered file (default: CPU_VS_GPU_TRACE_filtered.txt)
"""

import sys
import re

def filter_debug_segments(input_file, output_file, start_segment, end_segment, mode=None, interleave=False):
    """
    Filter debug output to only include segments within the specified range.
    Only includes debug statements that have a [SEGMENT XX] label.
    
    Parameters
    ----------
    input_file : str
        Path to input debug file
    output_file : str
        Path to output filtered file
    start_segment : int
        Starting segment number (inclusive)
    end_segment : int
        Ending segment number (inclusive)
    mode : str, optional
        Filter by mode: 'CPU', 'GPU', or None for both
    interleave : bool, optional
        If True, interleave CPU and GPU segments for easy comparison
    """
    # Pattern to match segment headers
    segment_pattern = re.compile(r'^\[SEGMENT (\d+)\]')
    
    # If input and output are the same, read the entire file first
    if input_file == output_file:
        with open(input_file, 'r') as f:
            all_content = f.read()
        # Create a temporary file-like object
        import io
        input_handle = io.StringIO(all_content)
    else:
        input_handle = None
    
    if interleave and mode is None:
        # Collect segments separately for CPU and GPU, then interleave
        cpu_segments = _collect_segments(input_file, start_segment, end_segment, "CPU", segment_pattern)
        gpu_segments = _collect_segments(input_file, start_segment, end_segment, "GPU", segment_pattern)
        
        # Write interleaved output
        with open(output_file, 'w') as fout:
            fout.write(f"=== FILTERED DEBUG OUTPUT: SEGMENTS {start_segment}-{end_segment} (INTERLEAVED) ===\n\n")
            
            for seg_num in range(start_segment, end_segment + 1):
                if seg_num in cpu_segments or seg_num in gpu_segments:
                    fout.write(f"{'='*60}\n")
                    fout.write(f"SEGMENT {seg_num:02d} COMPARISON\n")
                    fout.write(f"{'='*60}\n\n")
                    
                    if seg_num in cpu_segments:
                        fout.write("--- CPU ---\n")
                        fout.writelines(cpu_segments[seg_num])
                        fout.write("\n")
                    else:
                        fout.write("--- CPU ---\n(No CPU segment found)\n\n")
                    
                    if seg_num in gpu_segments:
                        fout.write("--- GPU ---\n")
                        fout.writelines(gpu_segments[seg_num])
                        fout.write("\n")
                    else:
                        fout.write("--- GPU ---\n(No GPU segment found)\n\n")
            
            fout.write(f"=== END FILTERED OUTPUT ===\n")
        return
    
    # Original sequential logic for non-interleaved or mode-specific filtering
    current_segment = None
    capture_lines = False
    lines_to_write = []
    header_written = False
    in_segment = False
    current_mode = None  # Track whether we're in CPU or GPU section
    
    # Use input_handle if we're overwriting, otherwise open the file
    if input_handle:
        fin = input_handle
    else:
        fin = open(input_file, 'r')
    
    try:
        for line in fin:
            stripped_line = line.strip()
            
            # Check for mode markers
            if "=== CPU TRACE START ===" in stripped_line:
                current_mode = "CPU"
                if mode is None or mode == "CPU":
                    lines_to_write.append(line)
                continue
            elif "=== GPU TRACE START ===" in stripped_line:
                current_mode = "GPU"
                if mode is None or mode == "GPU":
                    lines_to_write.append(line)
                continue
            elif "=== CPU TRACE END ===" in stripped_line:
                if mode is None or mode == "CPU":
                    lines_to_write.append(line)
                current_mode = None
                continue
            elif "=== GPU TRACE END ===" in stripped_line:
                if mode is None or mode == "GPU":
                    lines_to_write.append(line)
                current_mode = None
                continue
            
            # Check if this is a segment header
            match = segment_pattern.match(line)
            if match:
                # If we were capturing lines from a previous segment, write them
                if capture_lines and lines_to_write:
                    if not header_written:
                        mode_str = f" ({mode} only)" if mode else ""
                        with open(output_file, 'w') as fout:
                            fout.write(f"=== FILTERED DEBUG OUTPUT: SEGMENTS {start_segment}-{end_segment}{mode_str} ===\n\n")
                        header_written = True
                    
                    with open(output_file, 'a') as fout:
                        fout.writelines(lines_to_write)
                    lines_to_write = []
                
                # Get the segment number
                current_segment = int(match.group(1))
                in_segment = True
                
                # Determine if we should capture this segment (both range and mode)
                in_range = start_segment <= current_segment <= end_segment
                mode_matches = mode is None or current_mode == mode
                capture_lines = in_range and mode_matches
                
                if capture_lines:
                    lines_to_write.append(line)
            
            elif in_segment and capture_lines:
                # Continue capturing lines for the current segment until we hit a new segment or mode marker
                if line.startswith('[SEGMENT') or "=== " in line:
                    # New segment or mode marker, stop capturing
                    in_segment = False
                    capture_lines = False
                else:
                    lines_to_write.append(line)
            
            # Skip all non-segment debug output unless we're in capture mode
    finally:
        if not input_handle and hasattr(fin, 'close'):
            fin.close()
    
    # Write any remaining lines
    if capture_lines and lines_to_write:
        if not header_written:
            mode_str = f" ({mode} only)" if mode else ""
            with open(output_file, 'w') as fout:
                fout.write(f"=== FILTERED DEBUG OUTPUT: SEGMENTS {start_segment}-{end_segment}{mode_str} ===\n\n")
            header_written = True
        
        with open(output_file, 'a') as fout:
            fout.writelines(lines_to_write)
    
    if header_written:
        with open(output_file, 'a') as fout:
            fout.write(f"\n=== END FILTERED OUTPUT ===\n")
    else:
        mode_str = f" (mode: {mode})" if mode else ""
        with open(output_file, 'w') as fout:
            fout.write(f"=== NO SEGMENTS FOUND IN RANGE {start_segment}-{end_segment}{mode_str} ===\n")


def _collect_segments(input_file, start_segment, end_segment, target_mode, segment_pattern):
    """Helper function to collect segments for a specific mode"""
    segments = {}
    current_segment = None
    current_mode = None
    in_segment = False
    segment_lines = []
    
    with open(input_file, 'r') as fin:
        for line in fin:
            stripped_line = line.strip()
            
            # Track mode
            if "=== CPU TRACE START ===" in stripped_line:
                current_mode = "CPU"
                continue
            elif "=== GPU TRACE START ===" in stripped_line:
                current_mode = "GPU"
                continue
            elif "=== CPU TRACE END ===" in stripped_line or "=== GPU TRACE END ===" in stripped_line:
                current_mode = None
                continue
            
            # Only process if we're in the target mode
            if current_mode != target_mode:
                continue
            
            # Check for segment headers
            match = segment_pattern.match(line)
            if match:
                # Save previous segment if we were collecting one
                if in_segment and current_segment is not None and segment_lines:
                    if start_segment <= current_segment <= end_segment:
                        segments[current_segment] = segment_lines[:]
                
                # Start new segment
                current_segment = int(match.group(1))
                in_segment = True
                segment_lines = [line]
            
            elif in_segment:
                # Continue collecting lines for current segment
                if line.startswith('[SEGMENT') or "=== " in line:
                    # New segment or mode marker, stop
                    in_segment = False
                else:
                    segment_lines.append(line)
    
    # Save final segment
    if in_segment and current_segment is not None and segment_lines:
        if start_segment <= current_segment <= end_segment:
            segments[current_segment] = segment_lines[:]
    
    return segments

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nAdditional usage:")
        print("  python cull_debug_info.py <start> <end> [input_file] [output_file] [mode] [--interleave]")
        print("  mode: 'CPU', 'GPU', or omit for both")
        print("  --interleave: Interleave CPU and GPU segments for side-by-side comparison")
        print("\nExamples:")
        print("  python cull_debug_info.py 10 15                          # Segments 10-15, both CPU and GPU")
        print("  python cull_debug_info.py 10 15 file.txt out.txt         # Custom files")
        print("  python cull_debug_info.py 10 15 file.txt out.txt CPU     # CPU only")
        print("  python cull_debug_info.py 10 15 file.txt out.txt GPU     # GPU only")
        print("  python cull_debug_info.py 10 15 '' '' '' --interleave    # Interleaved side-by-side")
        print("  python cull_debug_info.py 12 12 '' comparison.txt '' -i  # Single segment comparison")
        sys.exit(1)
    
    start_segment = int(sys.argv[1])
    end_segment = int(sys.argv[2])
    
    if start_segment > end_segment:
        print("Error: start_segment must be <= end_segment")
        sys.exit(1)
    
    if start_segment < 1 or end_segment > 40:
        print("Warning: Segment numbers should be between 1 and 40")
    
    # Parse arguments, handling empty strings as defaults
    input_file = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "CPU_VS_GPU_TRACE.txt"
    # Default output file is the same as input file (overwrite mode)
    output_file = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else input_file
    
    # Check for mode and interleave flag
    mode = None
    interleave = False
    
    for i in range(5, len(sys.argv)):
        arg = sys.argv[i].upper()
        if arg in ['CPU', 'GPU']:
            mode = arg
        elif arg in ['--INTERLEAVE', '-I']:
            interleave = True
        elif arg and arg not in ['', '--INTERLEAVE', '-I']:
            print(f"Warning: Unknown argument '{sys.argv[i]}'")
    
    if mode and mode not in ['CPU', 'GPU']:
        print(f"Error: mode must be 'CPU' or 'GPU', got '{mode}'")
        sys.exit(1)
    
    if interleave and mode:
        print("Warning: Interleave mode works best without mode filtering. Proceeding with mode filter.")
    
    try:
        filter_debug_segments(input_file, output_file, start_segment, end_segment, mode, interleave)
        
        mode_str = f" (mode: {mode})" if mode else ""
        interleave_str = " (interleaved)" if interleave else ""
        if input_file == output_file:
            print(f"Filtered segments {start_segment}-{end_segment}{mode_str}{interleave_str} - overwrote {input_file}")
        else:
            print(f"Filtered segments {start_segment}-{end_segment}{mode_str}{interleave_str} from {input_file} to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()