# Backend diagnostics

Small, self-contained scripts for validating the accelerated backends on a
new machine (workstation, cluster node, new GPU / ROCm stack). Run them from
this folder — database paths are resolved relative to the scripts.

Recommended order on a fresh machine:

| Script | What it answers | Typical runtime |
|---|---|---|
| `check_device_stack.py` | Does this GPU accept the per-thread stack raise the kernels want? (ROCm troubleshooting) | seconds |
| `check_backend_correctness.py [c++\|gpu]` | Do the accelerated answers match the reference — and did the run actually use the backend (fallback guard)? | ~1 min + one-time compile |
| `profile_pipeline.py <binary\|ternary\|quaternary> [--backend c++\|gpu] [--budget 60] [--threads N] [--solver-internals]` | Where does the time go on THIS machine — grid, hull, solver/kernel wall, packing, results, glue? Sizes the run to the budget; `--solver-internals` adds in-kernel segment shares. On gpu, the kernel-wall vs host split tells you whether the card (e.g. consumer FP64 rates) or the host pipeline bounds throughput. | the budget you pass (+ compile on first use) |
| `benchmark_speed.py [backend] [nx nt]` | What speedup does this machine get, with compile time excluded? | ~1-2 min |
| `check_reference_threading.py [nx nt]` | Is the reference solver helped, hurt, or untouched by BLAS thread pools on this machine? | ~1-2 min |
| `gpufast_pass_split.py [--systems ...] [--budget 60]` | On THIS card, is the gpu-fast solve bound by the lockstep pass-1 kernel or by the pass-2 straggler re-solve? Decides the next gpu-fast lever: straggler solver (semismooth) vs evaluation rate (tensorized GEMM). | ~2x budget per system |
| `warmstart_calibration.py <binary\|ternary\|quaternary> [--backend c++\|gpu] [--nx N]` | For dense-grid sweeps (billions of equilibria): how much does seeding each point from a solved neighbor save (iteration CDF vs cold and vs the convergence-gate floor), how often does it land in the wrong basin, and does the driving-force acceptance check catch those? Feeds the marching-solver design. | ~5-20 min |

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
