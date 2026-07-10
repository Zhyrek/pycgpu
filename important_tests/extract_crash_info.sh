#!/bin/bash
# Extract crash information from core dump

if [ $# -ne 1 ]; then
    echo "Usage: $0 <core_file>"
    exit 1
fi

CORE_FILE=$1

# Since we know GDB can read it, let's extract the key info
echo "=== CRASH ANALYSIS ==="
echo "Core file: $CORE_FILE"
echo ""

# Find python executable
PYTHON_EXEC=$(which python3 2>/dev/null || which python 2>/dev/null || echo "python")

echo "=== CRASH BACKTRACE ==="
# Get the backtrace, filtering out the section info
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "set print frame-arguments all" \
    -ex "bt full" 2>/dev/null | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    grep -v "^$" | head -200

echo ""
echo "=== CRASH LOCATION ==="
# Get just the top of the stack
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "frame 0" \
    -ex "info frame" \
    -ex "info args" \
    -ex "info locals" 2>/dev/null | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    grep -v "^$" | head -50

echo ""
echo "=== SIGNAL INFORMATION ==="
# What signal caused the crash
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "print $_siginfo" \
    -ex "info signal $_siginfo.si_signo" 2>/dev/null | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    grep -v "^$"

echo ""
echo "=== THREAD INFORMATION ==="
# Show all threads
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "info threads" \
    -ex "thread apply all bt 3" 2>/dev/null | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    grep -v "^$" | head -100

echo ""
echo "=== GPU/HIP SPECIFIC ==="
# Look for GPU-specific functions in the backtrace
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "thread apply all bt" 2>/dev/null | \
    grep -E "(hip|Hip|HIP|gpu|GPU|kernel|rocm|amd)" | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    sort -u | head -50

echo ""
echo "=== MEMORY ACCESS INFORMATION ==="
# Try to get the faulting address
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "x/i \$pc" \
    -ex "info registers" 2>/dev/null | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    grep -E "(rip|rsp|rbp|rax|rsi|rdi|fault|0x)" | head -20

echo ""
echo "=== PYTHON CONTEXT ==="
# Try to get Python-specific information
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "py-bt" \
    -ex "py-list" 2>/dev/null | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    grep -v "^$" | head -50

# Alternative: look for Python info in the backtrace
if [ $? -ne 0 ]; then
    echo "Python GDB extensions not available, searching manually..."
    gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
        -ex "set pagination off" \
        -ex "bt" 2>/dev/null | \
        grep -E "(\.py|Python|pycalphad|equilibrium|cupy)" | \
        grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | head -30
fi

echo ""
echo "=== KEY SYMBOLS ==="
# Look for our specific functions
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "info functions equilibrium" \
    -ex "info functions kernel" 2>/dev/null | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    grep -v "^$" | head -30

echo ""
echo "=== QUICK DIAGNOSIS ==="
# Try to identify the type of crash
gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet \
    -ex "set pagination off" \
    -ex "bt" 2>/dev/null | \
    grep -v "READONLY\|ALLOC\|LOAD\|CONTENTS" | \
    head -5

SIGNAL=$(gdb "$PYTHON_EXEC" "$CORE_FILE" -batch -quiet -ex "print \$_siginfo.si_signo" 2>/dev/null | grep -v "READONLY" | grep "= ")

if echo "$SIGNAL" | grep -q "11"; then
    echo "→ Segmentation fault (SIGSEGV) - Memory access violation"
elif echo "$SIGNAL" | grep -q "6"; then
    echo "→ Abort signal (SIGABRT) - Program terminated itself"
elif echo "$SIGNAL" | grep -q "7"; then
    echo "→ Bus error (SIGBUS) - Invalid memory access"
else
    echo "→ Signal: $SIGNAL"
fi