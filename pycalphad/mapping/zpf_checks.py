from typing import Mapping
import copy
import logging

import numpy as np

from pycalphad import variables as v
from pycalphad.core.constants import COMP_DIFFERENCE_TOL
from pycalphad.core.composition_set import CompositionSet

from pycalphad.mapping.primitives import ZPFLine, Point, ZPFState, MIN_COMPOSITION
import pycalphad.mapping.zpf_equilibrium as zeq

_log = logging.getLogger(__name__)

"""
Two types of checking functions in this module

Simple checks
    simple_check_valid_point
    simple_check_change_in_phases
    simple_check_global_min

    These quickly check a step result and returns
    a bool whether the step result is valid or not

    These are intended for exit/direction finding, where we just need to know if
    an exit/direction is valid

Normal checks
    check_valid_point
    check_axis_values
    check_change_in_phases
    check_global_min
    check_similar_phase_composition

    These will check whether the step result is valid or
    if not valid, then will modify the zpf line or create
    a new node
"""

def simple_check_valid_point(step_results: tuple[Point, list[CompositionSet]], **kwargs):
    """
    Returns True or False for whether step result was successful equilibria

    Parameters
    ----------
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions

    Returns
    -------
    bool whether step result is valid or not (None)
    """
    if step_results is None:
        _log.info("Invalid equilibrium results")

    return step_results is not None

def simple_check_change_in_phases(step_results: tuple[Point, list[CompositionSet]], **kwargs):
    """
    Returns True or False for whether step result resulted in same number of phases

    Parameters
    ----------
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions

    Returns
    -------
    bool whether step result resulted in a phase change
    """
    if step_results is None:
        return False
    new_point, orig_cs = step_results
    num_different_phases = len(set(orig_cs).symmetric_difference(set(new_point.stable_composition_sets)))

    if num_different_phases != 0:
        _log.info(f"Number of stable phases changed from {[cs.phase_record.phase_name for cs in orig_cs]} to {new_point.stable_phases}")

    return num_different_phases == 0

def simple_check_global_min(step_results: tuple[Point, list[CompositionSet]], **kwargs):
    """
    Returns True or False for whether step result is still global min

    Parameters
    ----------
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions

    Returns
    -------
    bool whether step result is global min or not
    """
    if step_results is None:
        return False

    pdens = kwargs.get("pdens", 500)
    tol = kwargs.get("tol", 1e-4)
    system_info = kwargs.get("system_info", None)
    num_candidates = kwargs.get("global_num_candidates", 1)
    new_point, orig_cs = step_results
    global_test_point = zeq.find_global_min_point(new_point, system_info, pdens, tol, num_candidates)

    if global_test_point is not None:
        _log.info(f"Point is not global minimum. Current CS: {new_point.stable_phases}, new CS: {global_test_point.stable_phases}")

    return global_test_point is None

def _on_diagram_edge(point: Point):
    """Whether the point has two or more composition sets and every one of them lacks the same component (below
    2*MIN_COMPOSITION, the composition limit of the starting points): all at a pure component in a binary, on the
    same edge of the diagram in a ternary"""
    comp_sets = point.stable_composition_sets
    if len(comp_sets) < 2:
        return False
    X = np.array([cs.X for cs in comp_sets])
    return bool(np.any(np.all(X < 2*MIN_COMPOSITION, axis=0)))

def simple_check_diagram_edge(step_results: tuple[Point, list[CompositionSet]], **kwargs):
    """
    Returns False if the step result is on an edge of the diagram (see check_diagram_edge)
    """
    if step_results is None:
        return False
    return not _on_diagram_edge(step_results[0])

def check_diagram_edge(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    Where every composition set lacks the same component, an equilibrium between them lies on an edge of the
    diagram (at a pure component in a binary, on a binary edge of a ternary), where a two-phase region has no
    width: at fixed conditions it is a single tie line. A step from there can stay on the edge (e.g. in temperature
    from a pure component's transformation) and trace a line along the edge of the diagram. A ZPF line ends where it
    reaches an edge of the diagram.

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    None - this check does not produce any new nodes
    """
    if step_results is None or len(axis_data["axis_vars"]) < 2:
        return None
    if _on_diagram_edge(step_results[0]):
        _log.info("ZPF line reached an edge of the diagram. Ending ZPF line")
        zpf_line.status = ZPFState.REACHED_LIMIT
    return None

# A ZPF line is smooth between nodes: where the trace of a composition set turns by more than CURVATURE_ANGLE
# (degrees) between consecutive steps, the step is retried smaller, so that tight curves are drawn with enough points.
# Segments shorter than CURVATURE_MIN_SEGMENT (in axis step sizes) are not considered, nor lines whose composition sets
# are within CURVATURE_MIN_WIDTH (mole fraction) of each other: near a critical point or where a two-phase region
# closes into a second-order transition the traces coincide, and the line can have a cusp that smaller steps in the
# same direction would step past
CURVATURE_ANGLE = 20.0
CURVATURE_MIN_SEGMENT = 0.1
CURVATURE_MIN_WIDTH = 0.01

def _trace_location(point: Point, comp_set: CompositionSet, axis_data: Mapping):
    """Position of a composition set's trace in units of the axis step sizes: its mole fraction for each composition
    axis variable, the point's value for each state variable"""
    elements = [str(el) for el in comp_set.phase_record.nonvacant_elements]
    location = []
    for av in axis_data["axis_vars"]:
        if isinstance(av, v.X):
            location.append(comp_set.X[elements.index(str(av.species))] / axis_data["axis_delta"][av])
        else:
            location.append(float(np.squeeze(point.get_property(av))) / axis_data["axis_delta"][av])
    return np.array(location)

def check_curvature(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    Retries the step with a smaller step size where the trace of a composition set turns by more than
    CURVATURE_ANGLE, so that tight curves are resolved. Composition sets whose composition does not change
    (stoichiometric phases, which can turn back at a congruent point) are not considered. At the minimum step size
    the point is accepted.

    2 possible outcomes
        a) No trace turns more than CURVATURE_ANGLE, or the step size is at its minimum -> pass
        b) A trace turns more -> don't add point, reduce axis delta and attempt stepping again

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    None - this check does not produce any new nodes
    """
    if step_results is None or len(zpf_line.points) < 2 or len(axis_data["axis_vars"]) < 2:
        return None
    delta_scale = kwargs.get("delta_scale", 0.5)
    min_delta_ratio = kwargs.get("min_delta_ratio", 0.1)
    if zpf_line.current_delta * delta_scale / axis_data["axis_delta"][zpf_line.axis_var] < min_delta_ratio:
        return None
    new_point, orig_cs = step_results
    first, second = zpf_line.points[-2], zpf_line.points[-1]
    # (orig_cs are the last point's composition sets, in the same order, solved at the new conditions)
    if not (len(first.stable_composition_sets) == len(second.stable_composition_sets) == len(orig_cs)):
        return None
    compositions = [np.asarray(cs.X) for cs in orig_cs]
    if max(np.max(np.abs(a - b)) for a, b in zip(compositions[:-1], compositions[1:])) < CURVATURE_MIN_WIDTH:
        return None
    for cs_first, cs_second, cs_new in zip(first.stable_composition_sets, second.stable_composition_sets, orig_cs):
        if cs_first.phase_record.phase_name != cs_new.phase_record.phase_name:
            return None
        if np.max(np.abs(np.asarray(cs_new.X) - np.asarray(cs_first.X))) < 1e-9:
            continue   # (composition fixed)
        a = _trace_location(first, cs_first, axis_data)
        b = _trace_location(second, cs_second, axis_data)
        c = _trace_location(new_point, cs_new, axis_data)
        u, w = b - a, c - b
        if np.linalg.norm(u) < CURVATURE_MIN_SEGMENT or np.linalg.norm(w) < CURVATURE_MIN_SEGMENT:
            continue
        angle = np.degrees(np.arccos(np.clip(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)), -1, 1)))
        if angle > CURVATURE_ANGLE:
            _log.info(f"Trace of {cs_new.phase_record.phase_name} turns {angle:.0f} degrees, reducing step size from {zpf_line.current_delta} -> {zpf_line.current_delta*delta_scale}")
            zpf_line.current_delta *= delta_scale
            zpf_line.status = ZPFState.ATTEMPT_NEW_STEP
            return None
    return None

def check_valid_point(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    3 possible outcomes
        a) Converged equilibrium -> pass
        b) Failed equilibrium, reduce axis delta -> don"t add point and attempt stepping again
        c) Failed equilibrium and axis delta reached minimum -> end zpf line with unexpected ending

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    None - this check does not produce any new nodes
    However, this will end zpf_line if step_result is invalid
    """
    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]
    delta_scale = kwargs.get("delta_scale", 0.5)
    min_delta_ratio = kwargs.get("min_delta_ratio", 0.1)
    if step_results is None:
        _log.info(f"Invalid equilibrium result, reducing step size from {zpf_line.current_delta} -> {zpf_line.current_delta*delta_scale}")

        zpf_line.current_delta *= delta_scale
        if zpf_line.current_delta / axis_delta[zpf_line.axis_var] < min_delta_ratio:
            _log.info(f"Step size has reached minimum {zpf_line.current_delta}/{axis_delta[zpf_line.axis_var]} = {min_delta_ratio}. Failing ZPF line")

            zpf_line.status = ZPFState.FAILED
            return None
        zpf_line.status = ZPFState.ATTEMPT_NEW_STEP
    return None

# A composition set whose mole fractions change by more than this in one step has (usually) jumped to another
# solution of the equilibrium, rather than followed the ZPF line
COMPOSITION_JUMP = 0.1

def check_composition_jump(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    A step whose equilibrium jumped to another solution (a composition set moving far more than the ZPF line can in
    one step, e.g. into another minimum of its phase) is retried with a smaller step. Following the line, the change
    shrinks with the step; a jump does not. At the minimum step size the point is accepted as before.

    2 possible outcomes
        a) No composition set moved more than COMPOSITION_JUMP, or the step size is at its minimum -> pass
        b) A composition set jumped -> don't add point, reduce axis delta and attempt stepping again

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    None - this check does not produce any new nodes
    """
    if step_results is None or len(zpf_line.points) == 0:
        return None
    axis_delta = axis_data["axis_delta"]
    delta_scale = kwargs.get("delta_scale", 0.5)
    min_delta_ratio = kwargs.get("min_delta_ratio", 0.1)

    # orig_cs are the previous point's composition sets (copied in the same order), solved at the new conditions
    _, orig_cs = step_results
    prev_cs = zpf_line.points[-1].stable_composition_sets
    if len(prev_cs) != len(orig_cs):
        return None
    jump = max(np.max(np.abs(np.asarray(new.X) - np.asarray(prev.X))) for new, prev in zip(orig_cs, prev_cs))
    if jump > COMPOSITION_JUMP and zpf_line.current_delta * delta_scale / axis_delta[zpf_line.axis_var] >= min_delta_ratio:
        _log.info(f"Composition set moved {jump} in one step, reducing step size from {zpf_line.current_delta} -> {zpf_line.current_delta*delta_scale}")
        zpf_line.current_delta *= delta_scale
        zpf_line.status = ZPFState.ATTEMPT_NEW_STEP
    return None

def _check_axis_values_within_limit(new_point_vars: dict[v.StateVariable, float], axis_data: Mapping):
    """
    Checks that axis values are within the axis limits

    NOTE: I think prev_point_vars (unused) and new_point_vars are called this rather
          than conditions since they are taken from v.StateVariable.compute_property
          rather than from the global conditions

    NOTE: There were some issues before with X-C systems where a zpf line containing
          graphite and another stoichiometric phase would stop midway due to the
          composition of graphite being at X(C)=1

          Adding an offset to slightly extend the axis limits seems to help this issue.
          Local tests with the Bengt mpea-05 database on all X-C systems (X = Al, Co, Cr, Fe, Ni, Mn)
          appears to have to have all zpf lines correctly present

          I will also justify this offset with the following:
            1) Stepping will automatically step the stepping variable to be within axis limits
            2) For stoichiometric phases at an endmember, finite precision may cause the composition to be slightly above/below 1 (within numerical precision)
            3) For non-stoichiometric phases, composition will be limited between 0 and 1 due to the constraints in the minimizer

    Parameters
    ----------
    prev_point_vars : dict[v.StateVariable, float]
        Conditions of last point in zpf line
    new_point_vars : dict[v.StateVariable, float]
        Conditions of new point
    axis_data : dict
        Axis variable information from map strategy

    Returns
    -------
    bool for whether new point is within axis limits
    """
    axis_vars, _, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]

    for av in axis_vars:
        # Allow some leeway below and above the axis limits. See note in docstring for details
        offset = 1e-6

        if new_point_vars[av] < min(axis_lims[av]) - offset or new_point_vars[av] > max(axis_lims[av]) + offset:
            _log.info(f"New point outside axis limits. {av} = {new_point_vars[av]}. Limits = {axis_lims[av]}")
            return False
    return True

def _check_axis_values_by_distance(prev_point_vars: dict[v.StateVariable, float], new_point_vars: dict[v.StateVariable, float], axis_data: Mapping, **kwargs):
    """
    Checks that the normalized distance between the previous point and the new point is within reasonable values

    NOTE: a threshold of 3 is quite large since the axis swapping should limit this to 1 (give/take some leeway if the swapping hadn't occured yet)
    NOTE: this should scale with the global_check_interval defined in the map strategy

    Parameters
    ----------
    prev_point_vars : dict[v.StateVariable, float]
        Conditions of last point in zpf line
    new_point_vars : dict[v.StateVariable, float]
        Conditions of new point
    axis_data : dict
        Axis variable information from map strategy

    Returns
    -------
    bool for whether new point is within distance threshold of previous point
    """
    axis_vars, _, _ = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]
    normalize_factor = kwargs.get("normalize_factor", {av: 1 for av in axis_vars})

    dist_threshold = kwargs.get("distance_threshold", 3)
    # Squeezing here since v.MoleFraction.compute_property will return an array instead of a scalar
    distances = [np.squeeze(np.abs((new_point_vars[av] - prev_point_vars[av])/normalize_factor[av])) for av in axis_vars]
    dist = np.amax(distances)
    if dist > dist_threshold:
        av = axis_vars[np.argmax(distances)]
        _log.info(f"Axis variable moved more than threshold distance. {av} = {prev_point_vars[av]} -> {new_point_vars[av]}. Distance = {dist*normalize_factor[av]} > {dist_threshold}*{normalize_factor[av]}")

    return dist < dist_threshold

def check_axis_values(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    Checks both if axis values are within axis limits and if the axis values hadn"t moved to far from previous point

    The two checks are separated from here so that we can use them in _check_global_min and _check_change_in_phases

    3 possible outcomes
        a) Axes are within limits and minimal distance change -> pass
        b) Axes are outside limits -> end zpf line with graceful ending
        c) Distance changed too much -> end zpf line with unexpected ending

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    None - this check does not produce any new nodes
    This will end zpf_line with FAILED or REACHED_LIMIT depending on test that failed
    """
    if step_results is None:
        return None

    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]

    new_point, orig_cs = step_results
    prev_point = zpf_line.points[-1]
    new_point_vars = {av: new_point.get_property(av) for av in axis_vars}
    prev_point_vars = {av: prev_point.get_property(av) for av in axis_vars}

    if not _check_axis_values_by_distance(prev_point_vars, new_point_vars, axis_data, **kwargs):
        _log.info("Variable more than distance threshold. Failing ZPF line")
        zpf_line.status = ZPFState.FAILED
        return None

    if not _check_axis_values_within_limit(new_point_vars, axis_data):
        _log.info("Variable reach axis limit. Ending ZPF line")
        zpf_line.status = ZPFState.REACHED_LIMIT
        return None

def check_change_in_phases(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    If the number of phases changed upon stepping, then attempt to create a new node
        If making the new node was unsuccessful, then we end the zpf line anyways

    3 possible outcomes
        a) No change in phases -> pass
        b) Change in phases, node successfully found -> process new node and end zpf line with graceful ending
        c) Change in phases, node not found -> end zpf line with unexpected ending

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    new_node : Node
        Node from step_results if there is a change in phases

    or

    None : if step_results does not result in a change in phase
           or new node cannot be found (zpf_line will be ended)
    """
    if step_results is None:
        return None

    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]
    do_not_create_node = kwargs.get("do_not_create_node", False)    #For testing/debugging purposes

    new_point, orig_cs = step_results
    new_point_vars = {av: new_point.get_property(av) for av in axis_vars}
    num_different_phases = len(set(orig_cs).symmetric_difference(set(new_point.stable_composition_sets)))
    if num_different_phases != 0:
        _log.info("Number of phases changed")
        # By default, assumed the zpf failed. When we successfully find a new node, then we change this
        zpf_line.status = ZPFState.FAILED

        # Make sure the set of phases only change by one
        # Since we step along a single axis variable, we can only fix one additional phase to satisfy Gibbs phase rule
        if num_different_phases > 1:
            _log.info("Number of phases changes by more than one. Failing ZPF line")
            return None

        if do_not_create_node:
            new_node = None
        else:
            new_node = zeq.create_node_from_different_points(new_point, orig_cs, axis_vars)

        # If the new node was successfully created, then process the node, otherwise, we unexpectedly ended
        if new_node is not None:
            new_node_vars = {av: new_node.get_property(av) for av in axis_vars}
            # Check that new node satisfy axis limits and distance between nodes
            # If not, then the zpf line ends unexpectedly
            check_axis = _check_axis_values_within_limit(new_node_vars, axis_data)
            check_dist =_check_axis_values_by_distance(new_point_vars, new_node_vars, axis_data, **kwargs)
            if check_axis and check_dist:
                _log.info(f"New node found successfully. {new_node.global_conditions}, {new_node.fixed_phases}, {new_node.free_phases}")
                zpf_line.status = ZPFState.NEW_NODE_FOUND
                return new_node

        _log.info("New node could not be found. Failing ZPF line")

    return None

def check_global_min(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    Check if the point is global minimum
    1. Check if a new composition set can be added that can lower free energy
    2. Create a new node with the additional composition set
    3. Check that the new node is valid

    3 possible outcomes
        a) No change in phases -> pass
        b) Change in phases, node successfully found -> process new node and end zpf line with graceful ending
        c) Change in phases, node not found -> end zpf line with unexpected ending

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    new_node : Node
        Node from step_results if global min is found

    or

    None : if step_results is still global min
           or new node cannot be found (zpf_line will be ended)
    """
    if step_results is None:
        return None

    axis_vars, axis_delta, axis_lims = axis_data["axis_vars"], axis_data["axis_delta"], axis_data["axis_lims"]
    global_check_interval = kwargs.get("global_check_interval", 1)
    pdens = kwargs.get("pdens", 500)
    tol = kwargs.get("tol", 1e-4)
    num_candidates = kwargs.get("global_num_candidates", 1)
    system_info = kwargs.get("system_info", None)
    do_not_create_node = kwargs.get("do_not_create_node", False)    #For testing/debugging purposes

    new_point, orig_cs = step_results
    new_point_vars = {av: new_point.get_property(av) for av in axis_vars}
    if len(zpf_line.points) % global_check_interval == 0:
        global_test_point = zeq.find_global_min_point(new_point, system_info, pdens, tol, num_candidates)
        if global_test_point is None and getattr(new_point, "_merged_global_min_candidate", None) is not None \
                and getattr(zpf_line.points[-1], "_merged_global_min_candidate", None) is None:
            # A candidate with a driving force at another composition merged with a stable composition set when they
            # were solved together. This happens past the boundary of a two-phase region of one phase (a miscibility
            # gap), where the stable set is metastable: it starts the solve inside the gap. Where the line has just
            # reached it, a smaller step reaches the boundary, where the node is found. (Where the previous point
            # had such a candidate too, it is not the boundary and the line goes on)
            delta_scale = kwargs.get("delta_scale", 0.5)
            min_delta_ratio = kwargs.get("min_delta_ratio", 0.1)
            if zpf_line.current_delta * delta_scale / axis_delta[zpf_line.axis_var] >= min_delta_ratio:
                _log.info(f"Global min candidate merged with a stable composition set. Reducing step size from {zpf_line.current_delta} -> {zpf_line.current_delta*delta_scale}")
                zpf_line.current_delta *= delta_scale
                zpf_line.status = ZPFState.ATTEMPT_NEW_STEP
                return None
        if global_test_point is not None:
            _log.info(f"Global min detected. {new_point.stable_phases} -> {global_test_point.stable_phases}")
            # By default, assumed the zpf failed. When we successfully find a new node, then we change this
            zpf_line.status = ZPFState.FAILED

            if do_not_create_node:
                new_node = None
            else:
                # (node finding updates the composition sets in place: keep copies of the step point's for a retry)
                pristine = (copy.deepcopy(new_point.fixed_composition_sets), copy.deepcopy(new_point.free_composition_sets))
                new_node = zeq.create_node_from_different_points(global_test_point, new_point.stable_composition_sets, axis_vars)
                new_node = _retry_metastable_node(new_node, new_point, pristine, axis_vars, system_info, pdens, tol, num_candidates)

            if new_node is not None:
                new_node_vars = {av: new_node.get_property(av) for av in axis_vars}
                # Check that new node satisfy axis limits and distance between nodes
                # If not, then the zpf line ends unexpectedly
                check_axis = _check_axis_values_within_limit(new_node_vars, axis_data)
                check_dist =_check_axis_values_by_distance(new_point_vars, new_node_vars, axis_data, **kwargs)
                # check_dist = True
                if check_axis and check_dist:
                    zpf_line.status = ZPFState.NEW_NODE_FOUND
                    _log.info(f"New node found successfully. {new_node.global_conditions}, {new_node.fixed_phases}, {new_node.free_phases}")
                    return new_node

            # A node computed from a point closer to it usually succeeds: retry with a smaller step before failing
            delta_scale = kwargs.get("delta_scale", 0.5)
            min_delta_ratio = kwargs.get("min_delta_ratio", 0.1)
            if zpf_line.current_delta * delta_scale / axis_delta[zpf_line.axis_var] >= min_delta_ratio:
                _log.info(f"New node could not be found. Reducing step size from {zpf_line.current_delta} -> {zpf_line.current_delta*delta_scale}")
                zpf_line.current_delta *= delta_scale
                zpf_line.status = ZPFState.ATTEMPT_NEW_STEP
                return None
            _log.info("New node could not be found. Failing ZPF line")

MAX_NODE_RETRIES = 3

def _retry_metastable_node(new_node, new_point, pristine, axis_vars, system_info, pdens, tol, num_candidates):
    """
    The candidate with the largest driving force at a step point is not necessarily the phase that became stable
    first along the step. A node made with it is then metastable to another candidate, and the strategy would
    discard the whole ZPF line. Instead, rebuild the node from the step point with that candidate (a few times at most).
    """
    for _ in range(MAX_NODE_RETRIES):
        if new_node is None or system_info is None:
            return new_node
        other = zeq._find_global_min_cs(new_node, system_info, pdens, tol, num_candidates, known_states=True)
        if other is None or not zeq._detect_degenerate_phase(new_node, other[0]):
            return new_node
        cs = other[0]
        num_sv = cs.phase_record.num_statevars
        fixed, free = copy.deepcopy(pristine[0]), copy.deepcopy(pristine[1])
        base_point = Point(new_point.global_conditions, new_point.chemical_potentials, fixed, free)
        candidate = CompositionSet(cs.phase_record)
        candidate.update(np.array(cs.dof[num_sv:], dtype=np.float64), 1e-6, np.array(base_point.stable_composition_sets[0].dof[:num_sv], dtype=np.float64))
        if not zeq._detect_degenerate_phase(base_point, candidate):
            return new_node
        retry_point = Point(new_point.global_conditions, new_point.chemical_potentials, fixed, free + [candidate])
        _log.info(f"Node {new_node.stable_phases} is metastable to {cs.phase_record.phase_name}; retrying the node with it")
        retry_node = zeq.create_node_from_different_points(retry_point, base_point.stable_composition_sets, axis_vars)
        if retry_node is None:
            return new_node
        new_node = retry_node
    return new_node

def check_similar_phase_composition(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    If two composition sets are close in composition, then we stop zpf line pre-maturely so that they don"t go on top of each other

    If the composition are the same, it can result in ill-defined matrices in the solver

    2 possible outcomes
        a) Composition sets are separate -> pass
        b) Composition sets are similar -> end zpf line with unexpected ending

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    None : this check does not attempt to make a new node
    However, the zpf line will end if this check fails
    """
    if step_results is None:
        return None

    new_point, orig_cs = step_results
    comp_sets = new_point.stable_composition_sets
    for i in range(len(comp_sets)):
        for j in range(i+1, len(comp_sets)):
            same_comp = np.allclose(comp_sets[i].X, comp_sets[j].X, atol=10*COMP_DIFFERENCE_TOL)
            same_phase = comp_sets[i].phase_record.phase_name == comp_sets[j].phase_record.phase_name
            if same_comp and same_phase:
                zpf_line.status = ZPFState.REACHED_LIMIT
                _log.info(f"Two composition sets have the same composition. Ending ZPF line. {comp_sets[i]} = {comp_sets[j]}")
                return None
    return None

def check_circular_loop(zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]], axis_data: Mapping, **kwargs):
    """
    If the composition of the new result loops back to beginning of ZPF line, then
    we are likely in a loop. One case where this may occur is in ternary systems
    with a two-phase equilibria between a high melting point phase and liquid

    This stops the zpf line if the distance from the step result to the first point
    is smaller than the distance to the previous point. The only times this should
    occur is if the zpf line rapidly switched directions or if it loops in on itself

    Parameters
    ----------
    zpf_line : ZPFLine
        ZPFLine that the point is stepping in
    step_results : [Point, [CompositionSet]]
        Results from zpf_equilibrium.update_equilibrium_with_new_conditions
    axis_data : dict
        Axis variable data from a map strategy class

    Returns
    -------
    None : this check does not attempt to make a new node
    However, the zpf line will end if this check fails
    """
    if len(zpf_line.points) < 2:
        return None

    # Potential and composition variables are on different scales, so distances are normalized (by the step size)
    normalize_factor = kwargs.get("normalize_factor", {av: 1 for av in axis_data['axis_vars']})
    scale = np.array([normalize_factor[var] for var in axis_data['axis_vars']], dtype=float)
    x_curr = np.array([step_results[0].get_property(var) for var in axis_data['axis_vars']]) / scale
    x_first = np.array([zpf_line.points[0].get_property(var) for var in axis_data['axis_vars']]) / scale
    x_prev = np.array([zpf_line.points[-1].get_property(var) for var in axis_data['axis_vars']]) / scale
    vfirst = x_first - x_curr
    vprev = x_curr - x_prev
    dist_first = np.sqrt(np.sum(vfirst**2))
    dist_prev = np.sqrt(np.sum(vprev**2))
    # if distance to first point is smaller than to previous point, then stop zpf line
    if dist_first < dist_prev:
        zpf_line.status = ZPFState.REACHED_LIMIT

    return None
