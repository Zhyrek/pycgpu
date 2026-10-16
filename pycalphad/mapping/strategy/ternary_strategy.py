from typing import Union
import logging
import itertools
import copy

import numpy as np

from pycalphad import Database, Workspace, variables as v
from pycalphad.core.composition_set import CompositionSet
from pycalphad.property_framework.computed_property import LinearCombination

from pycalphad.mapping.primitives import ZPFLine, Node, Point, ExitHint, Direction, MIN_COMPOSITION, ZPFState, _eq_compset
from pycalphad.mapping.starting_points import point_from_equilibrium
import pycalphad.mapping.zpf_equilibrium as zeq
import pycalphad.mapping.utils as map_utils
from pycalphad.mapping.strategy.strategy_base import MapStrategy
from pycalphad.mapping.strategy.step_strategy import StepStrategy
from pycalphad.mapping.strategy.strategy_data import PhaseRegionData, get_invariant_data_from_tieline_strategy, get_tieline_data_from_tieline_strategy

_log = logging.getLogger(__name__)

def _get_delta_cs_var(point: Point, comp_sets: list[CompositionSet], axis_vars: list[v.StateVariable], normalize=True):
    """
    Given two composition sets, get the vector using axis_vars as coordinates

    Seems a bit strange to have the comp sets as an input when we already have the point, but it's just to have some
    flexibility on which comp sets to treat at 0 or 1
    """
    dx = point.get_local_property(comp_sets[0], axis_vars[0]) - point.get_local_property(comp_sets[1], axis_vars[0])
    dy = point.get_local_property(comp_sets[0], axis_vars[1]) - point.get_local_property(comp_sets[1], axis_vars[1])
    norm_factor = 1 if not normalize else np.sqrt(dx*dx + dy*dy)
    return [dx/norm_factor, dy/norm_factor]

def _get_norm(point: Point, axis_vars: list[v.StateVariable]):
    """
    Creates normal vector of a tieline

    Assumes point has two phases
    """
    vec = _get_delta_cs_var(point, point.stable_composition_sets, axis_vars)
    return [-vec[1], vec[0]]

def _create_linear_comb_conditions(point: Point, axis_vars: list[v.StateVariable], normal: list[float] = None):
    """
    Creates a linear combination along a normal vector

    If normal is not given, then the normal will be the normal of the tieline defined by the point
    """
    # Get normal and axis variable to step in (this will be along the maximum of the normal vector)
    if normal is None:
        normal = _get_norm(point, axis_vars)

    # Create a linear combination that forces the stepping to be along the normal
    c = normal[1]*point.global_conditions[axis_vars[0]] - normal[0]*point.get_property(axis_vars[1])
    lc = LinearCombination(normal[1]*axis_vars[0] - normal[0]*axis_vars[1])
    return lc, c

def _sort_point(point: Point, axis_vars: list[v.StateVariable]):
    """
    Given a point with 2 free phases, get derivative at both composition sets to
    test with CS to fix and which direction to start

    NOTE: unlike the binary version of this function, normal is defined from the tieline
          rather than the axis variables
    """
    _log.info(f"Sorting point {point.fixed_phases}, {point.free_phases}, {point.global_conditions}")

    # Free all phases in point, we will fix the phase at the end
    free_point = map_utils._generate_point_with_free_cs(point)

    # Get normal and axis variable to step in (this will be along the maximum of the normal vector)
    normal = _get_norm(free_point, axis_vars)
    av = axis_vars[np.argmax(np.abs(normal))]
    _log.info(f"Point {point.fixed_phases}, {point.free_phases} with normal {normal}. Testing derivative in {av}")

    # This is here to have derivative with respect to the axis variable to follow the direction of the normal
    #  NOTE: the derivative we compute is not the directional derivative with respect to the normal
    #  but rather the derivative with respect to the axis variable, with the normal as a fixed condition
    #  Since we lose the direction information of the normal when taking the derivative, mult_factor is here
    #  to retain that information
    mult_factor = np.sign(normal[axis_vars.index(av)])

    # Create a linear combination that forces the stepping to be along the normal
    lc, c = _create_linear_comb_conditions(point, axis_vars, normal)

    # Create temporary set of conditions that replaces the fixed axis variable with the linear combination
    del free_point.global_conditions[axis_vars[1-axis_vars.index(av)]]
    free_point.global_conditions[lc] = c

    # Test phase composition derivative for each phase stepping along the axis variable fixed along the normal
    stable_cs = free_point.stable_composition_sets
    cs_options = [[stable_cs[0], stable_cs[1]], [stable_cs[1], stable_cs[0]]]
    options_tests = []
    for cs_list in cs_options:
        for av_test in axis_vars:
            v_num = v.X(cs_list[1].phase_record.phase_name, av_test.species)
            der = mult_factor*zeq.compute_derivative(free_point, v_num, av, free_den = False)
            new_point = map_utils._generate_point_with_fixed_cs(point, cs_list[0], cs_list[1])
            options_tests.append((der, new_point, av_test))

    # Best index is the one with the largest derivative
    best_index = -1
    best_der = -1
    for i in range(len(options_tests)):
        _log.info(f"Option: Axis var {options_tests[i][2]}, derivative {options_tests[i][0]}, point {options_tests[i][1].fixed_phases}, {options_tests[i][1].free_phases}")
        if abs(options_tests[i][0]) > best_der:
            best_index = i
            best_der = abs(options_tests[i][0])

    if best_index == -1:
        return options_tests[0][0], options_tests[0][1], options_tests[0][2], normal
    else:
        return options_tests[best_index][0], options_tests[best_index][1], options_tests[best_index][2], normal

class TernaryStrategy(MapStrategy):
    def __init__(self, dbf: Database, components: list[str], phases: list[str], conditions: dict[v.StateVariable, Union[float, tuple[float]]], **kwargs):
        super().__init__(dbf, components, phases, conditions, **kwargs)
        # TODO: This assumes pure elements and will likely change with the generalize component support
        unlisted_element = list(set(self.components) - {'VA'} - set([str(av.species) for av in self.axis_vars]))[0]
        self.all_vars = self.axis_vars + [v.X(unlisted_element)]
        # Coverage check (see _add_uncovered_starting_points): equilibria on a grid of the axis step sizes, and the
        # distance (mole fraction) within which a sample's phase compositions count as on a ZPF line or tie triangle
        self.COVERAGE_CHECK = kwargs.get("COVERAGE_CHECK", True)
        self.COVERAGE_TOL = kwargs.get("COVERAGE_TOL", 0.01)
        # Starting points are added one at a time; after this many in a row that add no coverage (their ZPF lines
        # fail at once), no more are added
        self.COVERAGE_MAX_UNPRODUCTIVE = kwargs.get("COVERAGE_MAX_UNPRODUCTIVE", 5)
        self._coverage_samples = None
        self._coverage_seeded = set()
        self._coverage_last_uncovered = None
        self._coverage_unproductive = 0

    def generate_automatic_starting_points(self):
        """
        Searches axis limits to find starting points

        Here, we do a step mapping along the axis bounds and grab all the nodes
        The nodes of a step map is distinguished from starting points in that they have a parent
        """
        map_kwargs = self._constant_kwargs()

        # Iterate through axis variables, and set conditions to fix axis variable at min only
        for av in self.axis_vars:
            conds = copy.deepcopy(self.conditions)
            conds[av] = np.amin(self.axis_lims[av])

            # Adjust composition conditions to be slightly above 0 for numerical stability
            if isinstance(av, v.X):
                if conds[av] == 0:
                    conds[av] = MIN_COMPOSITION

            # Step map
            step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
            step.do_map()
            self.add_starting_points_from_step(step)

        # Additional step where we switch axis conditions to the unlisted variable
        conds = copy.deepcopy(self.conditions)
        conds[self.all_vars[-1]] = MIN_COMPOSITION
        del conds[self.axis_vars[0]]

        # Step map
        step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
        step.do_map()
        self.add_starting_points_from_step(step)

    def _compute_coverage_samples(self):
        """
        Equilibria on a grid of compositions (the axis step sizes, offset by half a step from the limits): for each
        two- or three-phase result, its conditions and its phase compositions
        """
        conds = {key: val for key, val in self.conditions.items() if key not in self.axis_vars}
        values = {}
        for av in self.axis_vars:
            lo, hi = np.amin(self.axis_lims[av]), np.amax(self.axis_lims[av])
            values[av] = np.arange(lo + self.axis_delta[av] / 2, hi, self.axis_delta[av])
        conds.update({av: values[av] for av in self.axis_vars})
        conds.setdefault(v.N, 1)
        samples = []
        try:
            wks = Workspace(self.dbf, self.components, self.phases, conds, models=self.models, phase_record_factory=self.phase_records)
        except Exception as err:
            _log.info(f"Coverage check equilibria failed: {err}")
            return samples
        GM = np.squeeze(wks.eq.GM); NP = np.squeeze(wks.eq.NP); X = np.squeeze(wks.eq.X)
        GM = GM.reshape(len(values[self.axis_vars[0]]), len(values[self.axis_vars[1]]))
        NP = NP.reshape(GM.shape + (-1,)); X = X.reshape(GM.shape + NP.shape[-1:] + (-1,))
        for i, j in np.ndindex(GM.shape):
            if not np.isfinite(GM[i, j]):
                continue
            stable = np.flatnonzero(NP[i, j] > 1e-6)
            if len(stable) in (2, 3):
                point_conds = {key: val for key, val in conds.items() if key not in self.axis_vars}
                point_conds[self.axis_vars[0]] = float(values[self.axis_vars[0]][i])
                point_conds[self.axis_vars[1]] = float(values[self.axis_vars[1]][j])
                samples.append((point_conds, [np.asarray(X[i, j, k], dtype=float) for k in stable]))
        return samples

    def _is_covered(self, compositions: list[np.ndarray]):
        """
        Whether phase compositions of a sample lie on a ZPF line found so far (two phases: on its two traces) or are
        the corners of a tie triangle found so far (three phases), within COVERAGE_TOL
        """
        def dist_to_trace(x, trace):
            if len(trace) == 1:
                return np.linalg.norm(x - trace[0])
            a, b = trace[:-1], trace[1:]
            d = b - a
            L = np.einsum('ij,ij->i', d, d)
            t = np.clip(np.einsum('ij,ij->i', x - a, d) / np.where(L > 0, L, 1), 0, 1)
            return np.min(np.linalg.norm(x - (a + t[:, None] * d), axis=1))
        tol = self.COVERAGE_TOL
        if len(compositions) == 2:
            for zpf_line in self.zpf_lines:
                if len(zpf_line.points) == 0 or len(zpf_line.points[0].stable_composition_sets) != 2:
                    continue
                traces = [np.array([np.asarray(p.stable_composition_sets[k].X, dtype=float) for p in zpf_line.points]) for k in range(2)]
                for x0, x1 in (compositions, compositions[::-1]):
                    if dist_to_trace(x0, traces[0]) < tol and dist_to_trace(x1, traces[1]) < tol:
                        return True
        else:
            for node in self.node_queue.nodes:
                corners = [np.asarray(cs.X, dtype=float) for cs in node.stable_composition_sets]
                if len(corners) == 3 and all(min(np.linalg.norm(x - c) for c in corners) < tol for x in compositions):
                    return True
        return False

    def _acceptable_starting_point(self, point: Point, conds):
        """
        A starting point that is not inside a region found so far (see _inside_map), not two composition sets in one
        state, and the global minimum (an equilibrium calculation can return a metastable result, which would start
        metastable ZPF lines)
        """
        if self._inside_map(conds):
            _log.info(f"Starting point {point.stable_phases} is inside a region found already")
            return False
        if any(_eq_compset(a, b) for a, b in itertools.combinations(point.stable_composition_sets, 2)):
            _log.info(f"Starting point {point.stable_phases} has two composition sets in one state")
            return False
        # (the global min check of the mapping's steps: find_global_min_point)
        free_point = map_utils._generate_point_with_free_cs(point)
        if zeq.find_global_min_point(free_point, self.system_info, self.GLOBAL_MIN_PDENS, self.GLOBAL_MIN_TOL, self.GLOBAL_MIN_NUM_CANDIDATES) is not None:
            _log.info(f"Starting point {point.stable_phases} is not the global minimum")
            return False
        return True

    def _inside_map(self, conds):
        """
        Whether the overall composition of conds lies inside a tie triangle or two-phase region found so far (the area
        swept by a ZPF line's tie lines): there, the mapping has already decided the phases, and a sample with other
        phases is not a gap but a metastable equilibrium
        """
        def xy(cs):
            X = np.asarray(cs.X, dtype=float)
            return [X[cs.phase_record.nonvacant_elements.index(str(av.species))] for av in self.axis_vars]
        p = np.array([float(conds[av]) for av in self.axis_vars])
        def inside(V):
            V = np.asarray(V)
            if np.any(p < V.min(axis=0)) or np.any(p > V.max(axis=0)):
                return False
            x, y = p; result = False
            for i in range(len(V)):
                (x1, y1), (x2, y2) = V[i], V[(i + 1) % len(V)]
                if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                    result = not result
            return result
        for node in self.node_queue.nodes:
            if len(node.stable_composition_sets) == 3 and inside([xy(cs) for cs in node.stable_composition_sets]):
                return True
        for zpf_line in self.zpf_lines:
            if len(zpf_line.points) < 2 or len(zpf_line.points[0].stable_composition_sets) != 2:
                continue
            ties = [[xy(cs) for cs in pt.stable_composition_sets] for pt in zpf_line.points]
            for (a0, b0), (a1, b1) in zip(ties[:-1], ties[1:]):
                if inside([a0, b0, b1, a1]):
                    return True
        return False

    def _add_uncovered_starting_points(self):
        """
        The strategy finds the regions connected to its starting points on the edges of the diagram (and to each
        other through nodes). A region not connected to them (e.g. a ternary compound surrounded by liquid) or cut off
        by a failed ZPF line or node is not found. So once the mapping has run out of nodes, the equilibria on a grid
        of the axis step sizes are compared with what was found, and a two- or three-phase equilibrium that is not on a
        ZPF line or tie triangle, at a composition outside every region found, is added as a starting point. One is
        added at a time, since the region it starts usually covers others
        """
        if not self.COVERAGE_CHECK or self._coverage_unproductive >= self.COVERAGE_MAX_UNPRODUCTIVE:
            return False
        if self._coverage_samples is None:
            self._coverage_samples = self._compute_coverage_samples()
        uncovered = [k for k, (conds, compositions) in enumerate(self._coverage_samples) if not self._is_covered(compositions)]
        if self._coverage_last_uncovered is not None:
            self._coverage_unproductive = 0 if len(uncovered) < self._coverage_last_uncovered else self._coverage_unproductive + 1
        self._coverage_last_uncovered = len(uncovered)
        for k in uncovered:
            if self._coverage_unproductive >= self.COVERAGE_MAX_UNPRODUCTIVE:
                _log.info("Coverage check: the last starting points added no coverage. Not adding more")
                break
            if k in self._coverage_seeded:
                continue
            self._coverage_seeded.add(k)
            conds, compositions = self._coverage_samples[k]
            if self._inside_map(conds):
                continue
            point = point_from_equilibrium(self.dbf, self.components, self.phases, copy.deepcopy(conds), models=self.models, phase_record_factory=self.phase_records)
            if point is None or len(point.stable_composition_sets) not in (2, 3):
                continue
            if not self._acceptable_starting_point(point, conds):
                continue
            _log.info(f"Coverage check: {len(compositions)} phase equilibrium at {conds} is not on the map. Adding starting point")
            exit_hint, direction, err_reason = self._validate_custom_starting_point(point, None)
            if exit_hint == ExitHint.NORMAL:
                self.node_queue.add_node(self._create_node_from_point(point, None, None, None, exit_hint), True)
            else:
                for d in (Direction.POSITIVE, Direction.NEGATIVE):
                    self.node_queue.add_node(self._create_node_from_point(point, None, None, d, ExitHint.POINT_IS_EXIT), True)
            return True
        return False

    def add_starting_points_from_step(self, step: StepStrategy):
        """
        Adds all 2-phase and 3-phase regions from step as starting points
        We also do a global min check to make sure these phase regions are truly the phases they say they are
        """
        for zpf_line in step.zpf_lines:
            if len(zpf_line.stable_phases) == 2 or len(zpf_line.stable_phases) == 3:
                num_phases = len(zpf_line.stable_phases)

                # Iterate through the zpf line until we find a point with the same number of phases as the
                # zpf line itself. This seems redundant, but it is because the first point of the zpf line
                # is likely going to be a node or a node parent, where the number of phases may differ by 1
                p_index = 0
                while len(zpf_line.points[p_index].stable_phases) != num_phases:
                    p_index += 1

                # If we do find a good point on the zpf line, then we can process it to be a starting point
                if len(zpf_line.points[p_index].stable_phases) == num_phases:
                    new_point = zpf_line.points[p_index]
                    if self.all_vars[-1] in new_point.global_conditions:
                        del new_point.global_conditions[self.all_vars[-1]]
                        new_point.global_conditions[self.axis_vars[0]] = new_point.get_property(self.axis_vars[0])
                    _log.info(f"Adding node {new_point.fixed_phases}, {new_point.free_phases}, {new_point.global_conditions}")

                    # New point will be make with all composition sets as free, since we set
                    # which CS to fix when finding an exit
                    free_point = map_utils._generate_point_with_free_cs(new_point)

                    cs_result = zeq._find_global_min_cs(free_point, system_info=self.system_info, pdens=self.GLOBAL_MIN_PDENS, tol=self.GLOBAL_MIN_TOL, num_candidates=self.GLOBAL_MIN_NUM_CANDIDATES)
                    if cs_result is None:
                        new_node = self._create_node_from_point(free_point, None, None, None)
                        self.node_queue.add_node(new_node)

    def _validate_custom_starting_point(self, point: Point, direction: Direction):
        """
        Modifies exit hint and direction based off number of composition sets
            Single phase -> cannot be added
            Two phase    -> point is exit, direction stays the same as input
            Three phase  -> normal exit finding strategy, no direction needed

        This is the same as BinaryStrategy
        """
        if len(point.stable_composition_sets) <= 1:
            return None, None, "Single phase detected"
        elif len(point.stable_composition_sets) == 2:
            return ExitHint.POINT_IS_EXIT, direction, None
        elif len(point.stable_composition_sets) == 3:
            return ExitHint.NORMAL, None, None
        else:
            return None, None, "More than three phases detected"

    def _find_exits_from_node(self, node: Node):
        """
        A node in a ternary system has three exits, which are combinations of 2 CS in the node
        Since the node is found from one pair of CS, one of the exits are already accounted for, so
        practically, it's only 2 exits
            However, if there are multiple starting points, a node may be found from multiple zpf lines
            thus a node may only have 1 or even 0 exits - (does this mean some exits are repeated if a node is found twice?)

        I believe this is the same as the binary strategy
        """
        exits, exit_dirs = super()._find_exits_from_node(node)
        if node.exit_hint == ExitHint.POINT_IS_EXIT:
            return exits, exit_dirs

        for cs_1, cs_2 in itertools.combinations(node.stable_composition_sets, 2):
            new_conds = {key: node.get_property(key) for key, val in node.global_conditions.items()}

            # Create point with free composition sets (and ignore whether cs_1 or cs_2 is actually fixed)
            # This will be modified in _determine_start_direction
            candidate_point = Point.with_copy(new_conds, node.chemical_potentials, [], [cs_1, cs_2])

            if not node.has_point_been_encountered(candidate_point, False):
                _log.info(f"Found candidate exit: {candidate_point.fixed_phases}, {candidate_point.free_phases}, {candidate_point.global_conditions}")
                exits.append(candidate_point)
                # This function is only responsible for finding exit points
                # The _determine_start_direction will take the point and find the direction and axis variable
                exit_dirs.append(None)

        return exits, exit_dirs

    def _determine_start_direction(self, node: Node, exit_point: Point, proposed_direction: Direction):
        """
        For stepping, only one direction is possible from a node since we either step positive or negative

        If a direction cannot be found, then we force add a starting point just past the exit_point
        """
        # Sort exit point to fix composition set that varies the least
        der, exit_point, axis_var, normal = _sort_point(exit_point, self.axis_vars)

        free_cs = exit_point.free_composition_sets[0]
        # If node is invariant, then we can get the other cs to know which direction to step away from
        # It's possible that the node is not invariant (ex. as a starting point), in which case, we just set the norm_delta_dot to 1
        other_cs = [cs for cs in node.stable_composition_sets if not _eq_compset(cs, exit_point.fixed_composition_sets[0]) and not _eq_compset(cs, exit_point.free_composition_sets[0])]
        if len(node.stable_composition_sets) == 3 and len(other_cs) > 0:
            delta_var = _get_delta_cs_var(exit_point, [free_cs, other_cs[0]], self.axis_vars)
            norm_delta_dot = np.dot(delta_var, normal)
        else:
            norm_delta_dot = 1


        if proposed_direction is None:
            # If the derivative of the axis variable is positive and the normal is the same direction as the delta
            # Or if derivative is negative and normal is opposite direction, then stepping is Positive direction
            # would be preferred. Same goes for the two other cases where Negative direction is preferred
            # We add the other direction just in case the first proposed direction fails
            if norm_delta_dot * der > 0:
                directions = [Direction.POSITIVE, Direction.NEGATIVE]
            else:
                directions = [Direction.NEGATIVE, Direction.POSITIVE]
        else:
            directions = [proposed_direction]

        for d in directions:
            dir_results = self._test_direction(exit_point, axis_var, d)
            if dir_results is not None:
                av_delta, other_av_delta = dir_results
                _log.info(f"Found direction: {axis_var, d, av_delta} for point {exit_point.fixed_phases}, {exit_point.free_phases}, {exit_point.global_conditions}")
                return exit_point, axis_var, d, av_delta

        # Steps in the axis variable that the line changes most in can fail in both directions, e.g. from an exit
        # point next to an edge of the diagram (a step towards the edge leaves it), so try the other axis variable
        other_axis_var = self.axis_vars[1 - self.axis_vars.index(axis_var)]
        for d in [Direction.POSITIVE, Direction.NEGATIVE]:
            dir_results = self._test_direction(exit_point, other_axis_var, d)
            if dir_results is not None:
                av_delta, other_av_delta = dir_results
                _log.info(f"Found direction: {other_axis_var, d, av_delta} for point {exit_point.fixed_phases}, {exit_point.free_phases}, {exit_point.global_conditions}")
                return exit_point, other_axis_var, d, av_delta

        self._add_starting_point_at_new_condition(exit_point, normal, Direction.POSITIVE if norm_delta_dot > 0 else Direction.NEGATIVE)

        return None
    
    def _add_starting_point_at_new_condition(self, point: Point, normal: list[float], direction: Direction):
        """
        If we made it here, then no direction has worked. This could be a case where
        the two CS of the exit point are stoichiometric, so there is no ZPF line leading
        to the next node
        So we take a step from the normal of the exit away from the third CS of the invariant
        and add a new starting point. We also add the current exit as a parent, so we don't
        search in that direction again
        """
        free_point = map_utils._generate_point_with_free_cs(point)
        copy_conds = copy.deepcopy(free_point.global_conditions)
        # Move by small amount (1e-3 seem to be a good value, too small, and we may fail to detect a possible third phase and too large, and we may step over a potential node)
        for av, norm_dir in zip(self.axis_vars, normal):
            copy_conds[av] += 1e-3*norm_dir*direction.value

        _log.info(f"Attemping to add point at {copy_conds}")
        new_point = point_from_equilibrium(self.dbf, self.components, self.phases, copy_conds, models=self.models, phase_record_factory=self.phase_records)
        if new_point is not None:
            # If new point is an invariant, then we add it as a starting point (point is added as a parent to prevent repeated exit calculations)
            # If new point is 2 phase, then we add it as a starting point the node being the exit
            _log.info(f"Adding starting point: {new_point.stable_phases}")
            success = False
            if len(new_point.stable_composition_sets) == 3:
                new_node = self._create_node_from_point(new_point, point, None, None)
                success = self.node_queue.add_node(new_node)
            elif len(new_point.stable_composition_sets) == 2:
                new_node = self._create_node_from_point(new_point, None, None, None, ExitHint.POINT_IS_EXIT)
                success = self.node_queue.add_node(new_node)
            else:
                _log.info("Point could not be starting point. Needs 2 or 3 phases")
            if not success:
                _log.info("Point has already been added")
        else:
            _log.info("Could not find a new node from conditions")

    def _process_new_node(self, zpf_line: ZPFLine, new_node: Node):
        """
        Post global min check after finding node to ensure it's not a metastable node

        This is the same as _process_new_node in the BinaryStrategy
        """
        _log.info("Checking if new node is metastable")
        cs_result = zeq._find_global_min_cs(new_node, system_info=self.system_info, pdens=self.GLOBAL_MIN_PDENS, tol=self.GLOBAL_MIN_TOL, num_candidates=self.GLOBAL_MIN_NUM_CANDIDATES)
        if cs_result is not None:
            if not zeq._detect_degenerate_phase(new_node, cs_result[0]):
                cs_result = None
        if cs_result is None:
            _log.info("Global eq check on new node passed")
            super()._process_new_node(zpf_line, new_node)
        else:
            _log.info("Global eq check failed. New node is metastable. Removing current zpf line.")
            self.zpf_lines.pop(-1)

    def get_invariant_data(self, x: v.StateVariable, y: v.StateVariable, global_x: bool = False, global_y: bool = False) -> list[PhaseRegionData]:
        """
        Create a dictionary of data for invariant plotting.

        Parameters
        ----------
        x : v.StateVariable
            The state variable to be used for the x-axis.
        y : v.StateVariable
            The state variable to be used for the y-axis.
        global_x : bool
            Whether variable x applies to the global system
        global_y : bool
            Whether variable y applies to the global system

        Returns
        -------
        list of StrategyData pertaining to each invariant
        """
        return get_invariant_data_from_tieline_strategy(self, x, y, global_x, global_y)

    
    def get_tieline_data(self, x: v.StateVariable, y: v.StateVariable, global_x: bool = False, global_y: bool = False) -> list[PhaseRegionData]:
        """
        Create a dictionary of data for plotting ZPF lines.

        Parameters
        ----------
        x : v.StateVariable
            The state variable to be used for the x-axis.
        y : v.StateVariable
            The state variable to be used for the y-axis.
        global_x : bool
            Whether variable x applies to the global system
        global_y : bool
            Whether variable y applies to the global system

        Returns
        -------
        list of Strategy pertaining to each tieline
        """
        return get_tieline_data_from_tieline_strategy(self, x, y, global_x, global_y)
