#!/bin/bash
# Advanced script to analyze GPU crash core files

if [ $# -ne 1 ]; then
    echo "Usage: $0 <core_file>"
    echo "Example: $0 python.core"
    exit 1
fi

CORE_FILE=$1

# Check if file exists
if [ ! -f "$CORE_FILE" ]; then
    echo "Error: Core file '$CORE_FILE' not found"
    exit 1
fi

echo "=== Core File Analysis ==="
echo "File: $CORE_FILE"
echo "Size: $(ls -lh "$CORE_FILE" | awk '{print $5}')"
echo "Type check:"
file "$CORE_FILE"

# Try to determine the executable that created the core
echo -e "\n=== Attempting to identify source executable ==="

# Method 1: Check for python in current environment
PYTHON_CANDIDATES=(
    "$(which python3 2>/dev/null)"
    "$(which python 2>/dev/null)"
    "/usr/bin/python3"
    "/usr/bin/python"
    "python3"
    "python"
)

EXEC_FOUND=""
for CANDIDATE in "${PYTHON_CANDIDATES[@]}"; do
    if [ -n "$CANDIDATE" ] && [ -f "$CANDIDATE" ]; then
        echo "Trying: $CANDIDATE"
        # Test if GDB can read the core with this executable
        if gdb -batch -ex "quit" "$CANDIDATE" "$CORE_FILE" 2>&1 | grep -q "Core was generated"; then
            EXEC_FOUND="$CANDIDATE"
            echo "✓ Successfully matched with: $EXEC_FOUND"
            break
        fi
    fi
done

if [ -z "$EXEC_FOUND" ]; then
    echo "Warning: Could not automatically determine executable"
    echo "Trying with generic ELF analysis..."
    EXEC_FOUND="python"
fi

echo -e "\n=== Core Dump Basic Info ==="
# Try to extract basic info even if executable doesn't match
strings "$CORE_FILE" | grep -E "^/.*python" | head -5

echo -e "\n=== Attempting GDB Analysis ==="
echo "Using executable: $EXEC_FOUND"

# More permissive GDB analysis
gdb "$EXEC_FOUND" "$CORE_FILE" -batch \
    -ex "set pagination off" \
    -ex "set print pretty on" \
    -ex "info inferiors" \
    -ex "info target" \
    -ex "maintenance info sections" 2>&1 | grep -E "(Core|core|signal|Signal)" | head -20

echo -e "\n=== Backtrace Attempt ==="
gdb "$EXEC_FOUND" "$CORE_FILE" -batch \
    -ex "set pagination off" \
    -ex "bt" \
    -ex "thread apply all bt 5" 2>&1 | head -100

echo -e "\n=== Register State (if available) ==="
gdb "$EXEC_FOUND" "$CORE_FILE" -batch \
    -ex "info registers" 2>&1 | head -20

echo -e "\n=== Searching for GPU/HIP signatures in core ==="
# Look for GPU-related strings in the core file
echo "HIP/ROCm references:"
strings "$CORE_FILE" | grep -iE "(hip|rocm|amd|gpu|kernel)" | sort -u | head -20

echo -e "\n=== Memory regions (from strings) ==="
strings "$CORE_FILE" | grep -E "(/dev/shm|/tmp/|\.so\.|hip|rocm)" | sort -u | head -20

echo -e "\n=== Python traceback (if present) ==="
strings "$CORE_FILE" | grep -A5 -B5 -E "(Traceback|File.*line|Error|equilibrium|pycalphad)" | head -50

echo -e "\n=== Alternative Analysis with eu-readelf (if available) ==="
if command -v eu-readelf &> /dev/null; then
    eu-readelf -n "$CORE_FILE" 2>/dev/null | head -50
else
    echo "eu-readelf not found (install elfutils for additional analysis)"
fi

echo -e "\n=== ROCgdb Analysis (if available) ==="
if command -v rocgdb &> /dev/null; then
    rocgdb "$EXEC_FOUND" "$CORE_FILE" -batch \
        -ex "set pagination off" \
        -ex "info rocm devices" \
        -ex "info rocm queues" \
        -ex "info rocm kernels" \
        -ex "maintenance info sections" 2>&1 | grep -vE "(warning|Warning)" | head -50
else
    echo "ROCgdb not found"
fi

echo -e "\n=== Summary ==="
echo "Core file size: $(stat -c%s "$CORE_FILE") bytes"
echo "Readable: $(if [ -r "$CORE_FILE" ]; then echo "Yes"; else echo "No"; fi)"
echo "Binary file: $(if file "$CORE_FILE" | grep -q "data\|ELF"; then echo "Yes"; else echo "No"; fi)"

# Check if it might be a different format
if file "$CORE_FILE" | grep -q "data"; then
    echo -e "\nNote: File appears to be raw binary data."
    echo "This could be:"
    echo "  - A core dump in non-standard format"
    echo "  - A HIP/ROCm specific dump format"
    echo "  - A partial or corrupted core dump"
    echo ""
    echo "Try these commands directly:"
    echo "  gdb python $CORE_FILE"
    echo "  rocgdb python $CORE_FILE"
    echo "  hexdump -C $CORE_FILE | head -100"
fi