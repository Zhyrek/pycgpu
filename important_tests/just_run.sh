#!/bin/bash
# Just run and show the error

echo "Running test.py with basic error capture..."
echo "=========================================="

# Run with minimal debugging enabled
HIP_LAUNCH_BLOCKING=1 python test.py 2>&1 | tail -20

echo ""
echo "=========================================="
echo "Last 20 lines shown above."
echo "Look for:"
echo "  - 'Memory access fault' = array out of bounds or NULL pointer"
echo "  - 'illegal instruction' = corrupted kernel code"
echo "  - 'out of resources' = too much shared/local memory"