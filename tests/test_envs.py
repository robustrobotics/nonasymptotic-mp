from nonasymptotic.envs import GrayCodeWalls

from sympy.combinatorics.graycode import GrayCode
import networkx as nx
import numpy as np

from pytest import approx


class Test3dGrayCodeEnv:
    env = GrayCodeWalls(3, 4, 0)
    gray_coords = [
        tuple(
            map(lambda s: int(s), [*gray_str])
        )
        for gray_str in GrayCode(3).generate_gray()
    ]

    start = (0, 0, 0)
    end = (3, 0, 0)

    def test_graph_connects_start_and_end(self):
        assert nx.has_path(self.env.no_walls_graph, self.start, self.end)

    def test_graph_is_linear(self):
        for node, deg in self.env.no_walls_graph.degree:
            if node == self.start or node == self.end:
                assert deg == 1
            else:
                assert deg == 2, "node %s should have degree 2" % str(node)

    def test_cube_centers(self):
        for gc in self.gray_coords:
            gc_ = np.array(gc)
            cube_center = np.ones(3) * 0.25 + 0.5 * gc_
            assert self.env.distance_to_wall(cube_center) == approx(0.25), "node %s transform is off" % str(gc)

    def test_walls_are_where_we_want_them_end_cubes(self):
        # (0, 0, 0) -> (0, 0, 1)
        errors = []

        # away from z axis
        afz = np.array([0.25, 0.25, 0.4])
        if self.env.distance_to_wall(afz) != approx(0.25):
            errors.append("wall incorrectly placed in only path from start")

        # close to z axis
        c2z = np.array([0.25, 0.25, 0.1])
        if self.env.distance_to_wall(c2z) != approx(0.10):
            errors.append("distance mis-measured at start close to z axis: %f" % self.env.distance_to_wall(c2z))

        assert not errors

    def test_walls_are_where_we_want_them_within_one_gray_unit(self):
        # testing (0, 0, 1) <-> (0, 0, 0), (0, 1, 1)
        errors = []
        # close to cube in -z dir
        c2y = np.array([0.6, 0.25, 0.15])
        if self.env.distance_to_wall(c2y) != approx(0.10):
            errors.append("wall incorrectly placed in -z direction of (0, 1, 0)")

        # close to cube in y dir
        c2z = np.array([0.25, 0.45, 0.95])
        if self.env.distance_to_wall(c2z) != approx(0.05):
            errors.append("wall incorrectly placed in y direction of (0, 1, 0)")

        assert not errors

    def test_walls_are_where_we_want_them_across_gray_units(self):

        # ensure that (0, 1, 0) -> (1, 1, 0) are properly linked up

        c2x = np.array([0.499, 0.74, 0.25])
        assert self.env.distance_to_wall(c2x) == approx(
            0.24), "wall incorrectly placed in x dir of (0, 1, 0), %s" % str(list(
            self.env.no_walls_graph.neighbors((0, 1, 0))))
