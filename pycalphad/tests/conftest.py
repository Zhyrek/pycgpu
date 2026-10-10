"""Test-suite configuration for backend selection.

The suite is meant to run once per backend:

    pytest pycalphad/tests                     # reference solver
    pytest pycalphad/tests --backend=cpp       # compiled C++ backend
    pytest pycalphad/tests --backend=metal     # Apple-silicon GPU backend
    pytest pycalphad/tests --backend=cuda      # CUDA GPU backend

The flag sets the global pycalphad backend for the whole session, so every
test exercises the selected dispatch path. Tests whose assertions are
bitwise against the reference report xfail (not skip) when the documented
eps-class engine difference trips them — divergences stay visible and
trackable, and turn into ordinary passes if the backend linear algebra
reaches bit-parity.

The ``PYCALPHAD_BACKEND`` environment variable is honored when the flag is
not given (the flag wins when both are set).
"""


def pytest_addoption(parser):
    parser.addoption(
        '--backend', action='store', default=None,
        choices=['default', 'cpu', 'reference', 'auto', 'cpp', 'c++', 'openmp', 'gpu', 'cuda',
                 'cuda-fast', 'metal'],
        help='pycalphad compute backend to run the suite under',
    )
    parser.addoption(
        '--metal-as-shipped', action='store_true', default=False,
        help="with --backend=metal, keep the backend's normal scheduling (small batches on "
             "the CPU, kernels compiled in the background) instead of forcing every "
             "condition through the GPU kernels",
    )


def pytest_configure(config):
    backend = config.getoption('--backend')
    if backend is not None:
        import os
        import pycalphad
        name = pycalphad.set_backend(backend)
        if name == 'metal' and not config.getoption('--metal-as-shipped'):
            # In normal use the Metal backend keeps small batches on the CPU
            # and compiles GPU kernels in the background. Under test, send
            # every condition through the GPU kernels (unconverged ones still
            # finish on the CPU) and compile before solving.
            os.environ.setdefault('PYCGPU_METAL_SYNC_COMPILE', '1')
            os.environ.setdefault('PYCGPU_METAL_MIN_BATCH', '0')
            os.environ.setdefault('PYCGPU_METAL_GPU_SHARE', '1')


def pytest_report_header(config):
    import pycalphad
    name, _ = pycalphad.get_backend()
    return f'pycalphad backend: {name}'
