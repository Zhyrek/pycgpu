"""
Debugging output and overridden solver hooks, to compare between checkouts (diff the output).

    PYTHONHASHSEED=0 PYTHONPATH=<checkout> python hooks_check.py > out.txt

Prints the Solver(debugging_output=True) text of an Al-Ni solve, and the state seen by a SystemSpecification
subclass overriding post_solve_hook after every iteration of an Al-Cr-Ni solve.
"""
import contextlib
import io
import os

import numpy as np


def main():
    from pycalphad import Database, equilibrium, variables as v
    from pycalphad.core.solver import Solver
    from pycalphad.core.minimizer import SystemSpecification
    import pycalphad.tests.databases
    db_dir = os.path.dirname(pycalphad.tests.databases.__file__)

    dbf = Database(os.path.join(db_dir, 'alni_dupin_2001.tdb'))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        equilibrium(dbf, ['AL', 'NI', 'VA'], ['LIQUID', 'FCC_L12', 'BCC_B2'],
                    {v.T: 1500, v.P: 101325, v.N: 1, v.X('NI'): 0.4}, solver=Solver(debugging_output=True))
    print(buf.getvalue())

    seen = []

    class HookedSpecification(SystemSpecification):
        def post_solve_hook(self, state):
            seen.append((state.iteration, np.asarray(state.chemical_potentials).tolist()))
            return super().post_solve_hook(state)

    # Solver.get_system_spec builds the spec from this module-level name
    import pycalphad.core.solver as solver_module
    solver_module.SystemSpecification = HookedSpecification

    dbf = Database(os.path.join(db_dir, 'alcocrni.tdb'))
    try:
        eq = equilibrium(dbf, ['AL', 'CR', 'NI', 'VA'], ['LIQUID', 'FCC_L12', 'BCC_B2'],
                         {v.T: 1300, v.P: 101325, v.N: 1, v.X('AL'): 0.2, v.X('CR'): 0.2}, solver=Solver())
        print('GM', repr(float(eq.GM.values.squeeze())))
    except Exception as e:
        print('hooked solver failed:', type(e).__name__, e)
    print('hook calls', len(seen))
    for item in seen:
        print(item)


if __name__ == '__main__':
    main()
