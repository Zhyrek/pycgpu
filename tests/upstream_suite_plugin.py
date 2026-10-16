"""
pytest plugin: run pycalphad's own test suite with CGWorkspace in place of Workspace.

    python -m pytest --pyargs pycalphad.tests -p tests.upstream_suite_plugin

Every equilibrium the suite computes (pycalphad.equilibrium, Workspace(...) in tests, the mapping strategies'
starting points) then comes from column generation; CGWorkspace falls back to pycalphad's solver, with a warning,
for problems it does not handle (custom solver objects, calc_opts other than pdens), and the count is reported.
"""
import warnings

import pycalphad
import pycalphad.core.equilibrium
import pycalphad.core.workspace
from pycalphad_cg import CGWorkspace

_fallbacks = []


def pytest_configure(config):
    pycalphad.Workspace = CGWorkspace
    pycalphad.core.equilibrium.Workspace = CGWorkspace
    original = CGWorkspace.recompute

    def recompute(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = original(self)
        _fallbacks.extend(str(w.message) for w in caught if 'column generation does not handle' in str(w.message))
        return result
    CGWorkspace.recompute = recompute


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f'pycalphad_cg: CGWorkspace in place of Workspace; '
                                f'{len(_fallbacks)} equilibrium calls fell back to pycalphad\'s solver')
    for reason in sorted(set(_fallbacks)):
        terminalreporter.write_line(f'    {reason}')
