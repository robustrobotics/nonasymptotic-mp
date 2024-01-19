import copy
import time
import heapq
from abc import ABC, abstractmethod
from itertools import count
from enum import Enum

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely import Point, Polygon, unary_union, LineString, MultiPolygon, difference
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate, nearest_points
from shapely.plotting import plot_polygon, plot_points, plot_line
from sympy.combinatorics.graycode import GrayCode


class Environment(ABC):

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

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

    @abstractmethod
    def distance_to_path(self, query_points):
        pass


class StraightLine(Environment):
    """
    Constructs a straight line of length 1 in a box environment with
    the given delta_clearance.
    """

    def __init__(self, dim, delta_clearance, seed=None):
        super().__init__(seed)
        assert dim >= 2
        assert delta_clearance >= 0

        # define the box bounds. first dim will be the one containing the line. the remaining
        # will just be delta-tube in the l_infinity norm.
        self.bounds_lower = np.array([-delta_clearance] + [-delta_clearance] * (dim - 1))
        self.bounds_upper = np.array([1.0 + delta_clearance] + [delta_clearance] * (dim - 1))
        self.line_dir = np.array([1.0] + [0.0] * (dim - 1)).reshape(1, -1)

        self.dim = dim

    def sample_from_env(self):
        return self.rng.uniform(self.bounds_lower, self.bounds_upper)

    def arclength_to_curve_point(self, t_normed):
        t = np.clip(t_normed, 0.0, 1.0)
        point_on_curve = np.zeros(self.dim)
        point_on_curve[0] += t
        return point_on_curve

    def is_motion_valid(self, start, goal):
        assert start.shape == goal.shape

        # take advantage of fact that shape is convex
        if start.ndim == 1:
            start_valid = np.all(start >= self.bounds_lower) and np.all(start <= self.bounds_upper)
            goal_valid = np.all(goal >= self.bounds_lower) and np.all(goal <= self.bounds_upper)
            return start_valid and goal_valid
        else:
            start_valid = np.all(start >= self.bounds_lower, axis=1) & np.all(start <= self.bounds_upper, axis=1)
            goal_valid = np.all(goal >= self.bounds_lower, axis=1) & np.all(goal <= self.bounds_upper, axis=1)
            return start_valid & goal_valid

    def distance_to_path(self, points):
        proj_points_clipped = np.clip(points[:, 0], 0.0, 1.0).reshape(-1, 1)
        return np.linalg.norm(points - proj_points_clipped * self.line_dir, axis=1)

    def is_prm_epsilon_delta_complete(self, prm, tol, n_samples_per_check=100, timeout=60.0, timeout_area_tol=1e-10,
                                      vis=False):
        conn_r = prm.conn_r
        prm_points_to_cvx_hull = {}

        length_tri_points = np.array([(0.0, conn_r), (0.0, 1.0), (1.0 - conn_r, 1.0)])
        length_space_to_cover = Polygon(length_tri_points)

        order_vec = np.array([-np.sqrt(2), np.sqrt(2)]) / 2

        timeout_time = time.process_time() + timeout

        # a fix: https://stackoverflow.com/questions/39504333/
        # python-heapq-heappush-the-truth-value-of-an-array-with-more-than-one-element-is
        # add in the corners of the triangle first
        heap_tiebreaker = count()
        vertex_heap = list(map(
            lambda pt: (np.inner(order_vec, pt), next(heap_tiebreaker), pt),
            length_tri_points
        ))
        heapq.heapify(vertex_heap)

        def _find_valid_prm_entry_exit_points(sample_query):
            # use prm to find the valid queries (with respect to epsilon delta completeness)
            t_start = sample_query[0]
            t_goal = sample_query[1]
            start = self.arclength_to_curve_point(t_start)
            goal = self.arclength_to_curve_point(t_goal)
            path_length = t_goal - t_start
            sols_in_and_outs, sols_distances, sols_ids = prm.query_all_graph_connections(start, goal)
            valid_pairs = (sols_distances
                           + np.linalg.norm(start - sols_in_and_outs[:, 0, :], axis=1)
                           + np.linalg.norm(goal - sols_in_and_outs[:, 1, :], axis=1)) <= path_length * (1 + tol)
            # store valid queries in some structure that encodes the multiset
            # compute new convex hulls.
            return sols_in_and_outs[valid_pairs, :, :], sols_distances[valid_pairs], sols_ids[valid_pairs]

        # some helpers since we'll be processing lots of points in the same way

        # some pre-processing for the next set adding helper (common computation)
        # this will define a long enough shadow that gives the effect of an open half-plane when
        # taken with the intersection
        ray_slope1 = -np.array([2 + tol, tol])
        ray_slope1 *= 2 / np.linalg.norm(ray_slope1)

        ray_slope2 = np.array([tol, 2 + tol])
        ray_slope2 *= 2 / np.linalg.norm(ray_slope2)

        def _add_to_set_system_with_ray_shooting(query_point, sols_in_and_outs_coords, sol_dists, sols_ids):
            # x opt 1: if the query point is already in the set, then we can just toss it.
            # x opt 2: save the conn_r bounding boxes, since they are used every time.
            # x opt 3: if the key existed before and the query was inside the set, its not a new coverset
            # san 4: ensure that when subselecting from the same numpy array that point pairs hash the same
            # TODO: use more stable identifiers
            # TODO: debug inverted query problem.
            # this may be all we can do in terms of optimization... unless we start subselecting nearest neighbors...

            _new_cover_sets = []
            for id_io, (nd_prm_in, nd_prm_out), prm_dist in zip(sols_ids, sols_in_and_outs_coords, sol_dists):
                u_1, u_2 = nd_prm_in[0], np.linalg.norm(nd_prm_in[1:])
                v_1, v_2 = nd_prm_out[0], np.linalg.norm(nd_prm_out[1:])

                id_io = tuple(id_io)

                try:
                    cvx_hull, conn_r_bounding_box = prm_points_to_cvx_hull[id_io]
                    if cvx_hull.covers(Point(query_point)):
                        continue

                    # the query point + its shadow
                    query_shadow = Polygon([query_point, query_point + ray_slope1, query_point + ray_slope2])

                    # take in union of query shadow and then clip off the bounding box defined by conn_r
                    cvx_hull = cvx_hull.union(query_shadow).intersection(conn_r_bounding_box).convex_hull

                except KeyError:
                    # compute radius bounding box
                    t_1_low = u_1 - np.sqrt(conn_r ** 2 - u_2 ** 2)
                    t_1_high = u_1 + np.sqrt(conn_r ** 2 - u_2 ** 2)

                    t_2_low = v_1 - np.sqrt(conn_r ** 2 - v_2 ** 2)
                    t_2_high = v_1 + np.sqrt(conn_r ** 2 - v_2 ** 2)
                    conn_r_bounding_box = Polygon([(t_1_low, t_2_low),
                                                   (t_1_high, t_2_low),
                                                   (t_1_high, t_2_high),
                                                   (t_1_low, t_2_high)])

                    # in this situation, to get as much bang for buck as possible, we can
                    # take the L_1 inner approximation

                    # TODO deal with empty intersection of inner approx
                    # if we DO NOT contain the the point (u_1, v_1) in the admissible set
                    # inner_approx_p1 = np.array([u_1, (u_2 + (1 + tol)*u_1 + v_2 + prm_dist - v_1) / tol])
                    # inner_approx_p2 = np.array([(u_2 + v_2 + prm_dist - (1 + tol) * v_1 + u_1) / -tol, v_1])

                    # if we DO contain the the point (u_1, v_1) in the admissible set
                    # inner_approx_p1 = np.array([u_1, (u_2 + (1 + tol)*u_1 + v_2 + prm_dist + v_1) / (tol + 2)])
                    # inner_approx_p2 = np.array([(u_2 + v_2 + prm_dist - (1 + tol) * v_1 - u_1) / -(2 + tol), v_1])

                    # in admissible set:
                    vertex_line_proj_admissible = prm_dist + u_2 + v_2 <= (1 + tol) * (v_1 - u_1)

                    if not vertex_line_proj_admissible:
                        inner_approx_p1 = np.array([u_1, (u_2 + (1 + tol) * u_1 + v_2 + prm_dist - v_1) / tol])
                        inner_approx_p2 = np.array([(u_2 + v_2 + prm_dist - (1 + tol) * v_1 + u_1) / -tol, v_1])

                        inner_approx_ray_p1 = inner_approx_p1 + ray_slope2
                        inner_approx_ray_p2 = inner_approx_p2 + ray_slope1
                    else:
                        inner_approx_p1 = np.array([u_1, (u_2 + (1 + tol) * u_1 + v_2 + prm_dist + v_1) / (tol + 2)])
                        inner_approx_p2 = np.array([(u_2 + v_2 + prm_dist - (1 + tol) * v_1 - u_1) / -(2 + tol), v_1])

                        inner_approx_ray_p1 = inner_approx_p1 + ray_slope1
                        inner_approx_ray_p2 = inner_approx_p1 + ray_slope2

                    # the ray we add depends on arrangement of approx_p1 and approx_p2
                    # inner_approx_ray_p1, inner_approx_ray_p2 = (inner_approx_p1 + ray_slope1,
                    #                                             inner_approx_p2 + ray_slope2) \
                    #     if inner_not_flip_rays else (inner_approx_p1 + ray_slope2,
                    #                                  inner_approx_p2 + ray_slope1)

                    inner_approx_open = Polygon([inner_approx_ray_p1,
                                                 inner_approx_p1,
                                                 inner_approx_p2,
                                                 inner_approx_ray_p2])

                    if inner_approx_open.covers(Point(query_point)):
                        cvx_hull = inner_approx_open.intersection(conn_r_bounding_box)
                        # plt.figure()
                        # fig, ax = plt.subplots()
                        # plot_polygon(conn_r_bounding_box, ax, color='green')
                        # plot_polygon(inner_approx_open.intersection(conn_r_bounding_box), color='purple')
                        # plt.show()

                    else:
                        # the query point + its shadow
                        query_shadow = Polygon([query_point, query_point + ray_slope1, query_point + ray_slope2])
                        cvx_hull = inner_approx_open.union(query_shadow).intersection(conn_r_bounding_box).convex_hull
                        # plt.figure()
                        # fig, ax = plt.subplots()
                        # plot_polygon(conn_r_bounding_box, ax, color='green')
                        # plot_polygon(query_shadow.intersection(conn_r_bounding_box), color='red')
                        # plot_polygon(inner_approx_open.intersection(conn_r_bounding_box), color='purple')
                        # plt.show()

                prm_points_to_cvx_hull[id_io] = (cvx_hull, conn_r_bounding_box)
                _new_cover_sets.append(cvx_hull)

                return _new_cover_sets

        def _process_query(query_point):
            # returns True if successfully queried, False if the PRM does not support the query with the
            # the required tolerance
            query_sol_ios, query_sol_dists, sols_ids = _find_valid_prm_entry_exit_points(query_point)

            if query_sol_ios.size <= 0:
                print('not e-d complete! failing query: %s' % str(query_point))

                if vis:
                    fig, axs = plt.subplots()
                    plot_polygon(length_space_to_cover, ax=axs, color='red')
                    plot_polygon(cover_union, ax=axs, color='blue')
                    plot_points(Point(query_point), ax=axs, color='green')
                    plt.show()
                return False, None

            _new_cover_sets = _add_to_set_system_with_ray_shooting(
                query_point, query_sol_ios, query_sol_dists, sols_ids
            )

            return True, _new_cover_sets

        cover_union = Polygon()
        while True:
            # if the heap is empty, sample a random point and see if the we can grow from there.
            if not vertex_heap:
                # sample a point query new solutions and add convex sets
                mp_left = difference(length_space_to_cover, cover_union)
                sample_pt = self._random_point_in_mpolygon(mp_left)
                length_space_sample = np.array(sample_pt.coords).flatten()

                # if vis:
                #     fig, axs = plt.subplots()
                #     plot_polygon(mp_left, ax=axs, color='blue')
                #     plot_points(sample_pt, ax=axs, color='green')
                #     plt.show()

                heapq.heappush(
                    vertex_heap,
                    (np.inner(length_space_sample, order_vec), next(heap_tiebreaker), length_space_sample)
                )

                # project over to boundary so we can try to get a deterministic completeness instead of sampling around
                proj_sample = np.array(nearest_points(mp_left, sample_pt)[0].coords).flatten()
                heapq.heappush(
                    vertex_heap,
                    (np.inner(proj_sample, order_vec), next(heap_tiebreaker), proj_sample)
                )

            for i in range(n_samples_per_check):

                try:
                    _, _, prm_query = heapq.heappop(vertex_heap)
                except IndexError:
                    break

                if Point(prm_query).within(cover_union):
                    continue

                query_successful, new_cover_sets = _process_query(prm_query)

                if not query_successful:
                    return False

                new_cover_sets_union = unary_union(new_cover_sets)

                try:
                    new_cover_pts_coords = np.array(
                        new_cover_sets_union.intersection(length_space_to_cover).boundary.coords
                    )[:-1]  # last point repeats the first
                    new_cover_points = np.array(list(
                        map(lambda cds: Point(cds), new_cover_pts_coords)
                    ))

                    # add points that are not in the current cover union to the heap
                    indices_to_add = (np.logical_not(cover_union.covers(new_cover_points)) &
                                      length_space_to_cover.covers(new_cover_points))

                    for pt_coords in new_cover_pts_coords[indices_to_add]:
                        heapq.heappush(
                            vertex_heap, (np.inner(pt_coords, order_vec), next(heap_tiebreaker), pt_coords)
                        )

                    cover_union = cover_union.union(new_cover_sets_union)

                except AttributeError:
                    # here, we catch the error if we query a new point but it doesn't grow the
                    # current coverage area at all.
                    continue

            # continue until timeout (where we confirm or deny? decide). or if we're covered return true
            # or if we don't have an admissible solution, return false.
            cover_frac = cover_union.intersection(length_space_to_cover).area / length_space_to_cover.area
            if cover_frac > 1.0 - timeout_area_tol:
                print('covered!')
                return True

            elif time.process_time() > timeout_time:
                print('timed out; covered fraction: %f' % cover_frac)
                if vis:
                    fig, axs = plt.subplots()
                    plot_polygon(length_space_to_cover, ax=axs, color='red')
                    plot_polygon(cover_union, ax=axs, color='blue')
                    plt.show()
                return False

    def _random_point_in_mpolygon(self, mpolygon):
        "Return list of k points chosen uniformly at random inside the polygon."
        # This is a helper method in this class so we can share the random seed.
        # someone wrote this so we didn't have to:
        # https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
        # (but I had to correct it because it wrote down the matrix slightly wrong).
        areas = []
        transforms = []
        for t in triangulate(mpolygon):
            areas.append(t.area)
            (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
            transforms.append([x1 - x0, x2 - x0, y1 - y0, y2 - y0, x0, y0])
        areas = np.array(areas)
        areas /= np.sum(areas)
        transform = self.rng.choice(transforms, p=areas)
        x, y = self.rng.uniform(size=(2,))
        if x + y > 1:
            p = Point(1 - x, 1 - y)
        else:
            p = Point(x, y)

        tp = affine_transform(p, transform)
        return tp


class GrayCodeWalls(Environment):
    def __init__(self, dim, length, thickness=0.0, seed=None):
        super().__init__(seed)
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

        # NOTE: currently not vectorized to optimize compute
        assert start.ndim == 1 and goal.ndim == 1

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

    def distance_to_path(self, queries):
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
