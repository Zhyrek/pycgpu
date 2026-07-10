#!/bin/bash
# Script to analyze GPU crash core files

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

# Verify it's a core dump file (binary)
if ! file "$CORE_FILE" | grep -q "core file\|ELF.*core"; then
    echo "Warning: '$CORE_FILE' may not be a valid core dump file"
    echo "Attempting to analyze anyway..."
fi

echo "=== Basic Core File Info ==="
file "$CORE_FILE"

# Try to find the python executable
PYTHON_EXEC=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON_EXEC" ]; then
    echo "Warning: Python executable not found, using 'python'"
    PYTHON_EXEC="python"
fi

echo -e "\n=== Extracting Backtrace ==="
gdb -batch -ex "bt" -ex "quit" "$PYTHON_EXEC" "$CORE_FILE" 2>/dev/null | head -50

echo -e "\n=== Thread Information ==="
gdb -batch -ex "info threads" -ex "quit" "$PYTHON_EXEC" "$CORE_FILE" 2>/dev/null | head -20

echo -e "\n=== Crash Location ==="
gdb -batch -ex "frame 0" -ex "list" -ex "quit" "$PYTHON_EXEC" "$CORE_FILE" 2>/dev/null | head -20

echo -e "\n=== GPU-Specific Analysis (if ROCgdb available) ==="
if command -v rocgdb &> /dev/null; then
    rocgdb -batch \
        -ex "info rocm devices" \
        -ex "info rocm kernels" \
        -ex "bt" \
        -ex "quit" \
        "$PYTHON_EXEC" "$CORE_FILE" 2>/dev/null | head -50
else
    echo "ROCgdb not found - install ROCm debugger for GPU analysis"
fi

echo -e "\n=== Memory Map Around Crash ==="
gdb -batch -ex "info proc mappings" -ex "quit" "$PYTHON_EXEC" "$CORE_FILE" 2>/dev/null | grep -E "(hip|rocm|gpu)" | head -10

echo -e "\n=== Signals ==="
gdb -batch -ex "info signals" -ex "quit" "$PYTHON_EXEC" "$CORE_FILE" 2>/dev/null | grep -E "SEGV|BUS|ABRT|FPE"

echo -e "\n=== Quick Symbol Check for GPU Functions ==="
gdb -batch -ex "bt full" -ex "quit" "$PYTHON_EXEC" "$CORE_FILE" 2>/dev/null | grep -E "(equilibrium_kernel|solve_equilibrium|hipLaunchKernel|hip.*Sync)" | head -20

echo -e "\n=== Additional AMD GPU Error Info ==="
# Check for HIP/ROCm error messages in backtrace
gdb -batch -ex "thread apply all bt" -ex "quit" "$PYTHON_EXEC" "$CORE_FILE" 2>/dev/null | grep -E "(Memory access fault|hip|rocm|amd)" -A 2 -B 2 | head -30