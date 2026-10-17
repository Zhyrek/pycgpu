.. title:: Properties for Workspace.get

Properties for ``Workspace.get``
================================

``Workspace.get`` and ``Workspace.get_dict`` take any number of properties and
return their values at every point of the calculation.
This page lists what can be passed to them.

All examples below use the Al-Ni database of `Dupin, Ansara and Sundman (2001)
<https://doi.org/10.1016/S0364-5916(01)00049-9>`_ that ships with the pycalphad
tests. At 1200 K and ``X(AL) = 0.2`` this alloy is two-phase (γ + γ′), so
``FCC_L12`` is stable twice:

.. code-block:: python

   from importlib.resources import files
   import pycalphad.tests.databases
   from pycalphad import Database, Workspace, as_property, variables as v

   dbf = Database(files(pycalphad.tests.databases) / "alni_dupin_2001.tdb")
   wks = Workspace(dbf, ["AL", "NI", "VA"], ["LIQUID", "FCC_L12", "BCC_B2", "AL3NI2"],
                   {v.N: 1, v.P: 101325, v.T: 1200, v.X("AL"): 0.2})

   wks.get("GM")                    # array(-87501.68...), in J/mol
   T, gm = wks.get("T", "GM")       # several properties give a list of arrays
   wks.get_dict("NP(*)")            # {NP_AL3NI2: nan, NP_BCC_B2: nan, NP_LIQUID: nan,
                                    #  NP_FCC_L12#1: 0.637, NP_FCC_L12#2: 0.363}


How properties are written
--------------------------

A property can be given as a string, as an object from ``pycalphad.variables``
(for example ``v.T`` or ``v.X("AL")``), or as a property object (for example
``as_property("HM.T")`` or the metaproperties below). Strings are converted with
``pycalphad.as_property``, which reads them as:

* ``NAME``, for example ``"T"`` or ``"GM"``;
* ``NAME(arg, ...)``, for example ``"X(AL)"``, ``"X(FCC_L12,AL)"`` or ``"GM(LIQUID)"``;
* ``NUMERATOR.DENOMINATOR``, a derivative such as ``"HM.T"`` (see `Derivatives`_).

If ``NAME`` is one of ``T``, ``P``, ``N``, ``X``, ``W``, ``MU``, ``NP`` or ``Y``,
it is a state variable from ``pycalphad.variables``. Any other name is looked up
as an attribute of each phase's ``Model`` (see `Model properties`_).
Names are case-sensitive, and component names must be upper case
(``"X(AL)"``, not ``"X(al)"``). Write phase names in upper case too: ``"X(fcc_l12,AL)"``
works, but ``"GM(fcc_l12)"`` raises a ``KeyError``.

Values are returned as NumPy arrays in the property's display units (see `Units`_).
The array has one axis for each condition that has more than one value, in the
order of ``wks.condition_axis_order``; a single-point calculation gives 0-d arrays.
``get`` returns one array per property, or a list when there is more than one.
``get_dict`` returns a dictionary keyed by the property objects, which shows how
wildcards and composition sets were expanded.


State variables and compositions
--------------------------------

Each entry can be written as a string or as the object below it.

.. list-table::
   :header-rows: 1
   :widths: 30 34 14 40

   * - Syntax
     - Meaning
     - Default units
     - Notes
   * - ``T``, ``P``, ``N``

       ``v.T``, ``v.P``, ``v.N``
     - Temperature, pressure, total moles of atoms
     - ``kelvin``, ``pascal``, ``mol``
     -
   * - ``X(AL)``

       ``v.X("AL")``
     - Overall mole fraction of a component
     - ``fraction``
     - ``X(*)`` gives every component except ``VA``.
   * - ``X(FCC_L12,AL)``

       ``v.X("FCC_L12", "AL")``
     - Mole fraction of a component in one phase
     - ``fraction``
     - ``NaN`` where the phase is not stable. ``X(*,AL)`` and ``X(*,*)`` are allowed.
   * - ``W(AL)``, ``W(FCC_L12,AL)``

       ``v.W("AL")``, ``v.W("FCC_L12", "AL")``
     - Overall or per-phase mass fraction
     - ``fraction``
     - Same wildcards as ``X``. Cannot be used in derivatives.
   * - ``MU(AL)``

       ``v.MU("AL")``
     - Chemical potential of a component
     - ``J / mol``
     - ``MU(*)`` gives every component except ``VA``.
   * - ``NP(FCC_L12)``

       ``v.NP("FCC_L12")``
     - Phase fraction (moles of atoms in the phase per mole of atoms)
     - ``fraction``
     - ``NaN``, not 0, where the phase is not stable. ``NP(*)`` gives every
       phase in the Workspace.
   * - ``Y(FCC_L12,0,AL)``

       ``v.Y("FCC_L12", 0, "AL")``
     - Site fraction of a species on a sublattice (counted from 0)
     - ``fraction``
     - ``Y(FCC_L12,0,*)``, ``Y(FCC_L12,*,*)`` and ``Y(*,0,AL)`` are allowed;
       ``Y(FCC_L12,*,AL)`` fails unless the species is on every sublattice.
       Cannot be used in derivatives.

``X``, ``W``, ``NP`` and ``Y`` count only stable composition sets.


Model properties
----------------

Any other name is computed from the ``Model`` of each phase. Without a phase
name it is the sum over the stable phases weighted by their phase fractions,
for example ``GM`` = Σ ``NP(phase)`` × ``GM(phase)``. With a phase name, such as
``GM(FCC_L12)``, it is the value for that phase, or ``NaN`` where the phase is not
stable. ``GM(*)`` gives one entry per phase in the Workspace.
The long names in parentheses are the same ``Model`` attributes.

.. list-table::
   :header-rows: 1
   :widths: 30 34 14 40

   * - Syntax
     - Meaning
     - Default units
     - Notes
   * - ``GM`` (``energy``)
     - Molar Gibbs energy
     - ``J / mol``
     -
   * - ``HM`` (``enthalpy``)
     - Molar enthalpy
     - ``J / mol``
     -
   * - ``SM`` (``entropy``)
     - Molar entropy
     - ``J / mol / K``
     -
   * - ``CPM`` (``heat_capacity``)
     - Molar heat capacity of each phase at fixed internal degrees of freedom,
       :math:`-T\,\partial^2 G_M / \partial T^2`
     - ``J / mol / K``
     - Leaves out changes of phase amounts and compositions with temperature;
       the equilibrium heat capacity is ``HM.T``. ``heat_capacity`` has no units attached.
   * - ``G``, ``H``

       (``formulaenergy``, ``formulaenthalpy``)
     - Gibbs energy and enthalpy per mole of formula units
     - ``J``
     - ``GM`` and ``HM`` times the number of atoms in the phase's formula unit.
       The long names have no units attached.
   * - ``GM_MIX``, ``HM_MIX``, ``SM_MIX``, ``CPM_MIX``

       (``mixing_energy``, ``mixing_enthalpy``, ``mixing_entropy``,
       ``mixing_heat_capacity``)
     - Mixing properties relative to the phase's own endmembers
     - none
     - Not meaningful for order-disorder (partitioned) phases such as ``FCC_L12``:
       a warning is emitted, and ``GM_MIX`` and ``HM_MIX`` are ``NaN``.
       For other reference states, see ``ReferenceState`` below.
   * - ``DOO`` (``degree_of_ordering``)
     - Degree of ordering, from 0 (disordered) to 1
     - none
     - Usually requested per phase, e.g. ``DOO(FCC_L12)``. Cannot be used in
       derivatives.
   * - ``TC``, ``NT``, ``BMAG``

       (``curie_temperature``, ``neel_temperature``, ``beta``)
     - Curie temperature, Néel temperature and magnetic moment from the magnetic model
     - none
     - Zero for phases without magnetic parameters. ``TC`` and ``NT`` are
       temperatures, but no units are attached.

Properties defined by a custom ``Model`` subclass are requested the same way, by
attribute name (see the viscosity example in :doc:`../examples/2_Computing_Properties/index`).
A name that is not a ``Model`` attribute raises an ``AttributeError``.


Phase names, composition sets and wildcards
-------------------------------------------

* A phase that is stable more than once at the same conditions (a miscibility
  gap or an ordered/disordered pair) is split into composition sets ``#1``, ``#2``, ...
  If this happens anywhere in the Workspace, a request such as ``X(FCC_L12,AL)``
  is expanded to ``X(FCC_L12#1,AL)`` and ``X(FCC_L12#2,AL)``, so ``get`` returns
  more arrays than were requested. Ask for ``X(FCC_L12#2,AL)`` to get one of them,
  or use ``get_dict`` to see which is which.
* ``*`` in place of a phase name expands to every phase in the Workspace; in place
  of a component, to every component except ``VA``; in place of a sublattice index
  or species of ``Y``, to those of the phase.
* Wildcards are not expanded inside the metaproperties below (:issue:`604`).


Derivatives
-----------

``"A.B"`` is the derivative of ``A`` with respect to ``B`` at equilibrium, with the
other conditions held fixed (Jansson derivatives; Sundman, Lu and Ohtani,
`Computational Materials Science 101 (2015) 127-137
<https://doi.org/10.1016/j.commatsci.2015.01.029>`_).
The best-known example is the equilibrium heat capacity ``"HM.T"``.

* The numerator can be ``T``, ``P``, ``N``, ``X(...)``, ``MU(...)``, ``NP(...)``,
  or a Model property such as ``GM``, ``HM`` or ``GM(FCC_L12#1)``.
  ``W(...)``, ``Y(...)`` and ``DOO`` cannot be differentiated.
* The denominator can be ``T``, ``P`` or an overall mole fraction ``X(AL)``, and
  must be one of the Workspace's conditions. Only one ``.`` is allowed, so second
  derivatives such as ``"HM.T.T"`` are not available.
* Wildcards are expanded, e.g. ``"NP(*).T"`` and ``"MU(*).X(AL)"``.
* The units are the numerator's units divided by the denominator's, e.g.
  ``joule / kelvin / mole`` for ``HM.T``.

.. code-block:: python

   wks.get("HM.T")                   # equilibrium heat capacity, J/mol/K
   wks.get("NP(FCC_L12#1).T")        # change of a phase fraction with temperature
   wks.get("HM.X(AL)")               # change of molar enthalpy with composition


Units
-----

Every property has fixed ``implementation_units`` and ``display_units`` that can
be changed. Put the new units in square brackets on the property object (not in
the string) and the values are converted with `pint <https://pint.readthedocs.io>`_:

.. code-block:: python

   wks.get(v.T["degC"])                   # 926.85
   wks.get(v.X("AL")["%"])                # 20.0
   wks.get(as_property("GM")["kJ/mol"])   # -87.50...
   wks.get(as_property("GM")["J/g"])      # -1671.5... (per gram of the whole system)
   wks.get(as_property("HM.T")["J/g/K"])

Conversions between per-mole and per-gram units of whole-system properties
use the average molar mass of the stable phases. Units that do not match dimensionally raise
``pint.DimensionalityError``, as do properties listed above without units
(for example ``as_property("TC")["K"]``). ``"GM[J/g]"`` inside a string is not
understood.


Metaproperties
--------------

These are not written as strings; create the object and pass it to ``get``.
Import them from ``pycalphad.property_framework``.

.. list-table::
   :header-rows: 1
   :widths: 38 40 40

   * - Object
     - Meaning
     - Notes
   * - ``DormantPhase("LIQUID", wks)(prop)``
     - ``prop`` for a phase that is not allowed to become stable, with its internal
       degrees of freedom minimized at the chemical potentials of the equilibrium
     - ``DormantPhase("LIQUID", wks).driving_force`` is the driving force for the
       phase to form (negative when it is metastable), in ``J / mol``.
   * - ``IsolatedPhase("LIQUID", wks)(prop)``
     - ``prop`` for the phase equilibrated on its own at the same conditions,
       with all other phases suspended
     - Works for phases that are never stable in ``wks``. Its ``driving_force``
       is always 0, because the phase is in equilibrium with itself;
       use ``DormantPhase`` for driving forces.
   * - ``ReferenceState([(phase, conds), ...], wks)(prop)``
     - ``prop`` relative to reference states, e.g. an enthalpy of mixing or formation
     - Each entry is a phase and the conditions of one reference point; for a
       binary enthalpy of mixing, give one entry per pure component. See
       :doc:`../examples/4_Metastability_Reference_States/index`.
   * - ``T0("FCC_L12", "BCC_B2", wks)``
     - Temperature where the two phases have equal molar Gibbs energy at the
       same composition
     - Both phases must be stable somewhere in ``wks``. ``NaN`` where no
       solution is found between 298.15 K and 6000 K.
   * - ``DrivingForce("FCC_L12")``
     - Driving force of a phase in the equilibrium
     - 0 for stable phases and ``NaN`` for phases that are not stable; use
       ``DormantPhase(...).driving_force`` for metastable phases.

A string such as ``"DrivingForce(LIQUID)"`` or ``"T0(FCC_L12,BCC_B2)"`` is read
as a Model property, so it returns ``NaN`` or raises an error instead of
computing the metaproperty.

.. code-block:: python

   from pycalphad.property_framework import DormantPhase, IsolatedPhase, ReferenceState

   wks.get(DormantPhase("LIQUID", wks).driving_force)        # about -5300 J/mol
   wks.get(IsolatedPhase("LIQUID", wks)("GM(LIQUID)"))      # GM of an all-liquid alloy
   ref = ReferenceState([("FCC_L12", {v.X("AL"): 0}), ("FCC_L12", {v.X("AL"): 1})], wks)
   wks.get(ref("HM"))                                        # HM relative to pure fcc Ni and Al


Linear combinations of mole fractions
-------------------------------------

A symbolic expression of overall or per-phase mole fractions, such as
``v.X("AL") - v.X("NI")`` or ``v.X("FCC_L12#1", "AL") - v.X("FCC_L12#2", "AL")``,
is evaluated as written. Other variables (``2*v.T``) and products
(``v.X("AL")*v.X("NI")``) are not supported. These expressions exist mainly for
conditions: a ratio ``v.X("AL")/v.X("NI")`` is turned into the condition form
``X(AL) - c*X(NI)`` and does not return the ratio.
