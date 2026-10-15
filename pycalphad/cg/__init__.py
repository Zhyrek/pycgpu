"""
Column-generation equilibrium solver (the ``'cg'`` backend).

A linear program over phase points (a phase and its site fractions), with the equilibrium conditions as rows, finds
the stable phases and their amounts; its duals are the chemical potentials. New points come from minimizing each
phase's tangent-plane distance (driving force) at those potentials. Once the phases are identified, a Newton
iteration on the chemical potentials and phase amounts finishes the equilibrium, and a last pass over all phases
certifies that nothing lies below the tangent plane. There is no energy grid.

Select it with ``pycalphad.set_backend('cg')``; :func:`solve_workspace` is called from the backend dispatch.
"""
from pycalphad.cg.solve import solve_workspace, CGCapabilityError

__all__ = ['solve_workspace', 'CGCapabilityError']
