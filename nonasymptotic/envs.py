from sympy.combinatorics.graycode import GrayCode
from shapely import Polygon, MultiPolygon
from scipy.spatial import ConvexHull
import networkx as nx
import numpy as np
import time

import copy
from enum import Enum
from abc import ABC, abstractmethod


class Environment(ABC):

    @abstractmethod
    def sample_from_env(self):
        pass

    @abstractmethod
    def arclength_to_curve_point(self, t_normed):
        pass

    @abstractmethod
    def is_motion_valid(self, start, goal):
        pass

    @abstractmethod
    def is_prm_epsilon_delta_complete(self, prm, tol):
        pass


class StraightLine(Environment):
    """
    Constructs a straight line of length 1 in a box environment with
    the given delta_clearance.
    """

    def __init__(self, dim, delta_clearance):
        assert dim >= 2
        assert delta_clearance >= 0

        # define the box bounds. first dim will be the one containing the line. the remaining
        # will just be delta-tube in the l_infinity norm.
        self.bounds_lower = np.array([-delta_clearance] + [-delta_clearance] * dim)
        self.bounds_upper = np.array([1.0 + delta_clearance] + [delta_clearance] * dim)

        self.dim = dim
        self.rng = np.random.default_rng()

    def sample_from_env(self):
        return self.rng.uniform(self.bounds_lower, self.bounds_upper)

    def arclength_to_curve_point(self, t_normed):
        t = np.clip(t_normed, 0.0, 1.0)
        point_on_curve = np.zeros(self.dim)
        point_on_curve[0] += t
        return point_on_curve

    def is_motion_valid(self, start, goal):
        # take advantage of fact that shape is convex
        start_valid = np.all(start >= self.bounds_lower) and np.all(start <= self.bounds_upper)
        goal_valid = np.all(start >= self.bounds_lower) and np.all(start <= self.bounds_upper)
        return start_valid and goal_valid

    def is_prm_epsilon_delta_complete(self, prm, tol, timeout=60.0):
        rng = np.random.default_rng()
        conn_r = prm.conn_r
        prm_points_to_cvx_hull = {}

        length_tri_points = [(0.0, conn_r), (0.0, 1.0), (1.0 - conn_r, 1.0)]
        length_space_to_cover = Polygon(length_tri_points)
        p1s = np.array(length_tri_points)
        p2s = np.roll(p1s, 1)
        p2s_min_p1s = p2s - p1s

        timeout_time = time.process_time() + timeout

        while True:
            # sample a point query new solutions and add convex sets
            unit_square_sample = rng.uniform(low=[0.0, 0.0], high=[1.0, 1.0])
            unit_triangle_sample = np.sort(unit_square_sample)
            length_space_sample = ((1.0 - conn_r) * (unit_triangle_sample - np.array([0.0, 1.0]))) + np.array(
                [0.0, 1.0])

            sample_sols_in_and_outs = self._find_valid_prm_entry_exit_points(length_space_sample, prm, tol)

            # if there are no sols, then the prm does _not_ have a valid path.
            if sample_sols_in_and_outs.size <= 0:
                return False

            # compute edge projection of the sampled query to an edge and then perform the same query search.
            # I was lazy and looked up the closed form expression:
            # https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter05/section05.html
            projection_to_sides = (
                    np.sum((length_space_sample - p1s) * p2s_min_p1s, axis=1)
                    * p2s_min_p1s
                    / np.sum(p2s_min_p1s * p2s_min_p1s, axis=1)
                    + p1s
            )
            proj_dists = np.linalg.norm(projection_to_sides - length_space_sample, axis=1)
            proj_sample = projection_to_sides[np.argmax(proj_dists)]

            proj_sols_in_and_outs = self._find_valid_prm_entry_exit_points(proj_sample, prm, tol)
            if proj_sols_in_and_outs.size <= 0:
                return False

            # add query points to convex set system
            for query_point, sols_in_and_outs in zip([length_tri_points, proj_sample],
                                                     [sample_sols_in_and_outs, proj_sols_in_and_outs]):
                for prm_in, prm_out in sols_in_and_outs:
                    try:
                        prm_points_to_cvx_hull[(prm_in, prm_out)].add_points([query_point])
                    except KeyError:
                        prm_points_to_cvx_hull[(prm_in, prm_out)] = ConvexHull([query_point], incremental=True)

            # compute new union polygon...  and I think we're forced to use Shapely since CGAL is missing bindings to do
            # the same computations.
            # it appears that shapely also has a covers function... ...
            convex_sets = map(lambda hull: Polygon(hull.vertices), prm_points_to_cvx_hull.values())
            ranges = MultiPolygon(convex_sets)

            # continue until timeout (where we confirm or deny? decide). or if we're covered return true
            # or if we don't have an admissible solution, return false.
            if ranges.covers(length_space_to_cover):
                return True

            if time.process_time() > timeout_time:
                print('Epsilon-Delta check timed out.')
                return True

    def _find_valid_prm_entry_exit_points(self, length_space_sample, prm, tol):
        # use prm to find the valid queries (with respect to epsilon delta completeness)
        t_start = length_space_sample[0]
        t_goal = length_space_sample[1]
        start = self.arclength_to_curve_point(t_start)
        goal = self.arclength_to_curve_point(t_goal)
        path_length = t_start - t_goal
        sols_in_and_outs, sols_distances = prm.query_all_graph_connections(start, goal)
        valid_pairs = (sols_distances
                       + np.linalg.norm(start - sols_in_and_outs[0], axis=1)
                       + np.linalg.norm(goal - sols_in_and_outs[1], axis=1)) <= path_length * (1 + tol)
        # store valid queries in some structure that encodes the multiset
        # compute new convex hulls.
        valid_sols_in_and_outs = np.swapaxes(sols_in_and_outs[:, valid_pairs, :], 0, 1)
        return valid_sols_in_and_outs


class GrayCodeWalls(Environment):
    def __init__(self, dim, length, thickness=0.0):
        assert dim >= 2
        assert length > 0

        # set up the snaking path through space using gray codes
        gray_code = GrayCode(dim - 1)
        gray_coords = []
        for gray_str in gray_code.generate_gray():
            gray_coords.append(tuple(list(map(lambda s: int(s), [*gray_str]))))

        gray_coords_reversed = copy.deepcopy(gray_coords)
        gray_coords_reversed.reverse()

        self.no_walls_linear_list = []  # we'll hold onto this because the ordering start -> end is helpful
        for i in range(length):
            if i % 2 == 0:
                self.no_walls_linear_list += self._prepend_coord(i, gray_coords)
            else:
                self.no_walls_linear_list += self._prepend_coord(i, gray_coords_reversed)

        no_walls_edge_list = list(zip(self.no_walls_linear_list, self.no_walls_linear_list[1:]))

        self.no_walls_graph = nx.DiGraph()
        self.no_walls_graph.add_nodes_from(self.no_walls_linear_list)
        self.no_walls_graph.add_edges_from(
            no_walls_edge_list)  # we could store the complement, (edge = wall) but that's larger memory

        # initialize all the random number generators that we'll need for sampling
        # we pre-process all probability values to speed up computation later
        self.cube_center_lengths = np.ones(dim) * (1.0 - 2 * thickness)

        self.open_wall_lengths = copy.deepcopy(self.cube_center_lengths)
        self.open_wall_lengths[0] = thickness

        vol_center = np.prod(self.cube_center_lengths)
        vol_open_wall = np.prod(self.open_wall_lengths)
        vol_end_passage = vol_center + vol_open_wall
        vol_mid_passage = vol_center + 2 * vol_open_wall

        self.n_nodes = len(self.no_walls_linear_list)
        total_volume = (self.n_nodes - 2) * vol_mid_passage + 2 * vol_end_passage

        self.rng = np.random.default_rng()

        # probability a sample falls in an end passage/mid passage
        prob_end_passage = vol_end_passage / total_volume  # this is the probability of one _specific_ end passage
        prob_mid_passage = vol_mid_passage / total_volume

        self.cuboid_pmf_vect = np.array(
            [prob_end_passage]
            + [prob_mid_passage] * (self.n_nodes - 2)
            + [prob_end_passage]
        )

        # probability a sample falls in cube center/space opened by absence of wall
        prob_center_in_end_passage = vol_center / vol_end_passage
        prob_open_wall_in_end_passage = vol_open_wall / vol_end_passage

        prob_center_in_mid_passage = vol_center / vol_mid_passage
        prob_open_wall_in_mid_passage = vol_open_wall / vol_mid_passage

        self.mid_passage_regions_pmf_vect = np.array(
            [prob_open_wall_in_mid_passage]
            + [prob_center_in_mid_passage]
            + [prob_open_wall_in_mid_passage])  # vector is ordered in transversal of the hamiltonion path

        self.end_passage_regions_pmf_vect = np.array(
            [prob_center_in_end_passage] + [prob_open_wall_in_end_passage]
        )  # can do the order here, so it's center -> open wall

        # we'll need these in later computations
        self.dim = dim
        self.length = length
        self.thickness = thickness

        # helpful quantity for solution curve computations
        self.curve_arclength = 0.5 * (1 + 2 * (len(self.no_walls_linear_list) - 2) + 1)

    def distance_to_wall(self, x):
        assert x.shape == (self.dim,)

        # begin by rounding to know which hypercube we are in
        cube_coords = np.floor(x).astype('int64')

        return self._distance_to_wall_in_cube(cube_coords, x)

    def _distance_to_wall_in_cube(self, cube_coords, x):
        # sometimes, for a measure zero set of points, it's useful to manually specify
        # exactly which cube we're in. but we need to be careful when doing this, which is
        # why this is given as only an internal helper method.

        # translate to cube center
        x_c = x - (cube_coords + 0.5)
        # get neighbors so we know there are free passageways
        neighbors = (list(self.no_walls_graph.predecessors(tuple(cube_coords)))
                     + list(self.no_walls_graph.successors(tuple(cube_coords))))
        # fill in coordinates of walls, accounting for neighbors with no walls
        walls_low = np.ones(self.dim) * -0.5
        walls_high = np.ones(self.dim) * 0.5
        for i in range(len(neighbors)):
            walls_low, walls_high = self._unblock_wall(
                cube_coords, np.array(neighbors[i]), walls_low, walls_high)
        walls_low_dists = x_c - walls_low
        walls_high_dists = walls_high - x_c
        return min(np.min(walls_low_dists), np.min(walls_high_dists)) - self.thickness

    def sample_from_env(self):
        # first, sample which cuboid we will land
        sampled_cuboid_coords_ix = self.rng.choice(len(self.no_walls_linear_list), p=self.cuboid_pmf_vect)
        sampled_cuboid_coords = self.no_walls_linear_list[sampled_cuboid_coords_ix]

        # next, sample if we are in the center or (one of/only) open wall in selected cuboid
        if ((sampled_cuboid_coords == self.no_walls_linear_list[0])
                or (sampled_cuboid_coords == self.no_walls_linear_list[-1])):
            sampled_region_ix = self.rng.choice(a=2, p=self.end_passage_regions_pmf_vect)
            sampled_region = [EndCuboidRegions.CENTER, EndCuboidRegions.PASSAGE][sampled_region_ix]

        else:
            sampled_region_ix = self.rng.choice(a=3, p=self.mid_passage_regions_pmf_vect)
            sampled_region = [MidCuboidRegions.PASSAGE1,
                              MidCuboidRegions.CENTER,
                              MidCuboidRegions.PASSAGE2][sampled_region_ix]

        # sample point from the defined region (sample from a box then do coord scale)
        # and then transform back
        unit_cube_sample = self.rng.uniform(size=(self.dim,))

        return self._transform_sample_to_global_frame(unit_cube_sample, sampled_cuboid_coords, sampled_region)

    def start_point(self):
        # the start point is always the same: [0.5]*d
        return np.array([0.5] * self.dim)

    def end_point(self):
        return np.array(self.no_walls_linear_list[-1]) + 0.5 * np.ones(self.dim)

    def arclength_to_curve_point(self, t_normed):
        # set to scale from [0, 1] to the true geometric length of the curve
        t = np.clip(t_normed, 0, 1) * self.curve_arclength

        # compute number of half-edges traversed (including the current one)
        t_leg = 1 if t <= 0.0 else np.ceil(t / 0.5)

        # then, find the cube, check to see how much progress t has made in the cube, and then work out the coordinate
        # from there.
        if t_leg == 1:
            t_point = np.ones(self.dim) * 0.5
            t_point[-1] += t
        elif t_leg == 1 + 2 * (len(self.no_walls_linear_list) - 2) + 1:
            cube_coords = np.array(self.no_walls_linear_list[-1])
            t_backup = 0.5 - (t - (t_leg - 1) * 0.5)
            t_point = cube_coords + np.ones(self.dim) * 0.5
            t_point[-1] += t_backup

        else:
            cube = self.no_walls_linear_list[int(t_leg / 2)]
            t_point = np.array(cube) + np.ones(self.dim) * 0.5
            if t_leg % 2 == 0:  # if we are on the first (entrance) leg of the cube
                t_backup = 0.5 - (t - (t_leg - 1) * 0.5)
                pred_cube = self.no_walls_graph.predecessors(cube).__next__()
                dir_to_entrance = np.array(pred_cube) - np.array(cube)
                t_point += t_backup * dir_to_entrance

            else:  # if we are on the second (exit) leg of the cube
                t_forward = t - (t_leg - 1) * 0.5
                succ_cube = self.no_walls_graph.successors(cube).__next__()
                dir_to_exit = np.array(succ_cube) - np.array(cube)
                t_point += t_forward * dir_to_exit

        return t_point

    def get_curve_arclength(self):
        return self.curve_arclength

    def is_motion_valid(self, start, goal):
        # we're doing this as an exact computation, so then we can have exact experiment results.
        # this will also save computational power in the long run.

        start_cube = np.floor(start).astype('int64')
        goal_cube = np.floor(goal).astype('int64')

        # check if we are searching forward or backward.
        searching_forward = [tuple(goal_cube)] in nx.dfs_successors(self.no_walls_graph, tuple(start_cube)).values()

        while np.any(start_cube != goal_cube):
            # find line constants. We'll parameterize it as a line l: [0, 1] \to \{env\} as a fun
            # of [0, 1], where l(0) = start and l(1) = goal
            dir_vec = goal - start

            # find the next cube so we can find the shared opening.
            next_cube = self.no_walls_graph.successors(tuple(start_cube)).__next__() if searching_forward \
                else self.no_walls_graph.predecessors(tuple(start_cube)).__next__()
            next_cube = np.array(next_cube)

            # find the search direction between cubes
            next_cube_dir = next_cube - start_cube
            next_cube_dir_ind = np.where(next_cube_dir)

            # compute shared opening on path:
            start_cube_center = start_cube + np.ones(self.dim) * 0.5
            next_cube_center = goal_cube + np.ones(self.dim) * 0.5
            between_cubes_point = (start_cube_center + next_cube_center) / 2

            # solve for point at cube opening.
            t = (between_cubes_point[next_cube_dir_ind] - start[next_cube_dir_ind]) / dir_vec[next_cube_dir_ind]
            if t <= 0.0 or np.any(np.isnan(t)):
                # we are only in while loop if we are spanning multiple blocks. so if t = 0 or nan, then line does
                # not travel through the exit of this block on the path, so it must have collided somewhere else
                return False
            motion_point_at_opening = dir_vec * t + start

            # check if start and mp@opening point are not in collision. this is a slightly
            # a different computation than distance_to_wall, which returns differently at a measure 0
            # set of points (but that set matters here)
            if (self._distance_to_wall_in_cube(start_cube, start) < 0.0
                    or self._distance_to_wall_in_cube(start_cube, motion_point_at_opening) < 0.0):
                return False

            start = motion_point_at_opening
            start_cube = next_cube  # this needs to be done so we don't get trapped in the same cube

        # if start and goal are in the same cube, then return true because cubes are convex
        return self.distance_to_wall(start) >= 0.0 and self.distance_to_wall(goal) >= 0.0

    def is_prm_epsilon_delta_complete(self, prm, tol):
        raise NotImplementedError('Not done yet!')

    def _transform_sample_to_global_frame(self, unit_sample, cuboid_coords, region):

        if region == EndCuboidRegions.CENTER or region == MidCuboidRegions.CENTER:
            region_sample = unit_sample * self.cube_center_lengths

            # since the center case is all the same, we can transform back immediately.
            # translate coords to the center of the cube
            region_sample -= self.cube_center_lengths / 2

            # then translate to the global coord frame using the selected cuboid

        else:
            region_sample = unit_sample * self.open_wall_lengths  # the short side is in the 0th dimension

            # exchange axes to reflect to the correct orientation
            ix = 0 if region == EndCuboidRegions.PASSAGE or region == MidCuboidRegions.PASSAGE1 else 1
            neighbors = (list(self.no_walls_graph.predecessors(cuboid_coords)) +
                         list(self.no_walls_graph.successors(cuboid_coords)))
            adjoined_cuboid_coords = neighbors[ix]

            # find direction (e.g. dimension) cuboid of open wall (sign says if positive or negative direction)
            open_wall_dir = np.array(adjoined_cuboid_coords) - np.array(cuboid_coords)
            open_wall_dir_ind = np.where(open_wall_dir)
            open_wall_dir_sign = open_wall_dir[open_wall_dir_ind]

            # swap indices to reflect to the right orientation
            region_sample[0], region_sample[open_wall_dir_ind] = region_sample[open_wall_dir_ind], region_sample[0]

            # transform to local center coordinates of cuboid
            translation = np.ones(self.dim) * -0.5 + self.thickness
            translation[open_wall_dir_ind] = 0.5 - self.thickness if open_wall_dir_sign > 0 else -0.5
            region_sample += translation

        # translate to global coordinates and then return
        return region_sample + np.array(cuboid_coords) + 0.5

    @staticmethod
    def _unblock_wall(cube_coords, neighbor_coords, walls_low, walls_high):
        # remove the wall (by replacing coordinate with np.inf for passageways by neighbor_coords
        neighbor_diff = neighbor_coords - cube_coords
        neighbor_diff_ind = np.where(neighbor_diff)
        neighbor_diff_sign = neighbor_diff[neighbor_diff_ind]

        if neighbor_diff_sign > 0:
            walls_high[neighbor_diff_ind] = np.inf
        else:
            walls_low[neighbor_diff_ind] = -np.inf

        return walls_low, walls_high

    @staticmethod
    def _prepend_coord(coord, list_of_coords):
        # prepend coord to all tuples given in a list of tuples
        return list(
            map(
                lambda tup: (coord,) + tup,
                list_of_coords
            )
        )


class MidCuboidRegions(Enum):
    PASSAGE1 = 0
    CENTER = 1
    PASSAGE2 = 2


class EndCuboidRegions(Enum):
    PASSAGE = 3
    CENTER = 4


if __name__ == '__main__':
    walls = GrayCodeWalls(2, 2, 0.125)
    print(walls.no_walls_linear_list)
