#!/bin/bash
# ROCgdb debugging script for AMD GPU crash analysis

echo "AMD GPU Debugging with ROCgdb"
echo "=============================="

# Option 1: Run with ROCgdb directly
echo "Option 1: Interactive ROCgdb session"
cat << 'EOF'
# Run this command:
rocgdb python

# Then in ROCgdb:
(rocgdb) set environment HIP_LAUNCH_BLOCKING 1
(rocgdb) set environment AMD_LOG_LEVEL 4
(rocgdb) set environment HSA_ENABLE_DEBUG 1
(rocgdb) set cuda break_on_launch
(rocgdb) run important_tests/test_all_phases.py --alcufe-only
(rocgdb) bt        # When it crashes, get backtrace
(rocgdb) info registers
(rocgdb) info cuda kernels
(rocgdb) info cuda threads
EOF

echo -e "\n======================================\n"

# Option 2: Automated crash detection
echo "Option 2: Automated crash detection with core dump"
cat << 'EOF'
# Enable core dumps
ulimit -c unlimited
export HIP_ENABLE_COREDUMP=1
export ROCM_DEBUG_ENABLE=1

# Run with automatic backtrace on crash
rocgdb -batch \
  -ex "set environment HIP_LAUNCH_BLOCKING=1" \
  -ex "set environment AMD_LOG_LEVEL=4" \
  -ex "run" \
  -ex "bt" \
  -ex "info registers" \
  -ex "info cuda kernels" \
  -ex "info cuda threads" \
  -ex "thread apply all bt" \
  --args python important_tests/test_all_phases.py --alcufe-only
EOF

echo -e "\n======================================\n"

# Option 3: ROCprof for profiling
echo "Option 3: ROCprof to find which kernel crashes"
cat << 'EOF'
# Create ROCprof config
echo "pmc: SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_VMEM_WR,SQ_INSTS_VMEM_RD,SQ_INSTS_SALU,SQ_INSTS_SMEM,SQ_INSTS_FLAT,SQ_INSTS_FLAT_LDS_ONLY,SQ_INSTS_LDS,SQ_INSTS_GDS" > rocprof.txt

# Run with profiling
rocprof -i rocprof.txt -o results.csv python important_tests/test_all_phases.py --alcufe-only

# This will show which kernel was executing when crash occurred
EOF

echo -e "\n======================================\n"

# Option 4: HIP built-in debugging
echo "Option 4: HIP built-in error checking"
cat << 'EOF'
# Enable HIP error checking
export HIP_LAUNCH_BLOCKING=1
export HIP_CHECK_ERRORS=1
export HIP_ABORT_ON_ERROR=1
export HIP_DB=1
export AMD_SERIALIZE_KERNEL=3  # Wait for completion after each kernel
export AMD_SERIALIZE_COPY=3    # Wait for completion after each copy

python important_tests/test_all_phases.py --alcufe-only 2>&1 | tee debug_output.log
EOF