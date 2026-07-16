# Backend diagnostics

Small, self-contained scripts for validating the accelerated backends on a
new machine (workstation, cluster node, new GPU / ROCm stack). Run them from
this folder — database paths are resolved relative to the scripts.

Recommended order on a fresh machine:

| Script | What it answers | Typical runtime |
|---|---|---|
| `check_device_stack.py` | Does this GPU accept the per-thread stack raise the kernels want? (ROCm troubleshooting) | seconds |
| `check_backend_correctness.py [c++\|gpu]` | Do the accelerated answers match the reference — and did the run actually use the backend (fallback guard)? | ~1 min + one-time compile |
| `benchmark_speed.py [backend] [nx nt]` | What speedup does this machine get, with compile time excluded? | ~1-2 min |

Things these scripts teach you to watch for anywhere else:

- **Exact-zero dGM is a red flag, not a success.** The dispatch falls back
  to the reference silently by design; comparing the reference with itself
  gives perfect agreement. `check_backend_correctness.py` sets
  `PYCGPU_COUNT_DISPATCH` and reports fallbacks explicitly.
- **Exclude the first call from timings.** Each new system compiles its
  kernel once (15-35 s, then disk-cached). `benchmark_speed.py` warms the
  cache with a 1-point solve before timing.
- **`PYCGPU_TIME=1`** prints the kernel wall time separately from the
  Python overhead (grid sampling, starting point, result processing) —
  use it when a speedup looks low to see which side the time went to.
- **ROCm:** if the stack raise is rejected (`hipErrorUnknown`), gpu-backend
  correctness depends on the platform's static scratch sizing — the
  correctness check is the arbiter. `PYCGPU_DEVICE_STACK_BYTES=<n>` requests
  a different size (`0` skips the call); the c++ backend is numerically
  identical and immune.
