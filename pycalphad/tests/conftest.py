"""Test-suite configuration for backend selection.

The suite is meant to run once per backend:

    pytest pycalphad/tests                     # reference solver
    pytest pycalphad/tests --backend=c++       # accelerated c++ backend
    pytest pycalphad/tests --backend=gpu       # accelerated gpu backend

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
        choices=['default', 'c++', 'cpp', 'gpu'],
        help='pycalphad compute backend to run the suite under',
    )


def pytest_configure(config):
    backend = config.getoption('--backend')
    if backend is not None:
        import pycalphad
        pycalphad.set_backend(backend)


def pytest_report_header(config):
    import pycalphad
    name, _ = pycalphad.get_backend()
    return f'pycalphad backend: {name}'
