#!/bin/bash
# Quick debugging script for test.py

echo "Quick AMD GPU Debug for test.py"
echo "================================"

# Method 1: Simple crash location finder
echo -e "\nMethod 1: Finding crash location...\n"
rocgdb -batch \
  -ex "set environment HIP_LAUNCH_BLOCKING=1" \
  -ex "set environment AMD_SERIALIZE_KERNEL=3" \
  -ex "run" \
  -ex "bt" \
  -ex "info cuda kernels" \
  --args python test.py 2>&1 | tee crash_location.txt

echo -e "\n================================"
echo "Crash location saved to: crash_location.txt"
echo "Look for:"
echo "  - 'Program received signal' - shows the signal type"
echo "  - 'Thread X' - shows which thread crashed"
echo "  - '#0' in backtrace - shows exact function and line"
echo "  - 'kernel:' entries - shows which GPU kernel was running"