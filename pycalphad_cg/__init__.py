"""
Column-generation equilibrium solver for pycalphad.

    from pycalphad_cg import CGWorkspace, equilibrium

``CGWorkspace`` is a pycalphad ``Workspace`` whose equilibria come from column generation instead of pycalphad's
grid, convex hull and refinement; ``equilibrium`` is pycalphad's ``equilibrium`` on it.
"""
from pycalphad_cg.workspace import CGWorkspace, equilibrium
from pycalphad_cg.solve import CGCapabilityError

__all__ = ['CGWorkspace', 'equilibrium', 'CGCapabilityError']
