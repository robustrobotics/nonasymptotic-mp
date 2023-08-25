from nonasymptotic.envs import GrayCodeWalls, MidCuboidRegions, EndCuboidRegions

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
        errors = []
        for node, deg in self.env.no_walls_graph.degree:
            if node == self.start or node == self.end:
                if deg != 1:
                    errors.append("node %s should have degree 1" % str(node))
            else:
                if deg != 2:
                    errors.append("node %s should have degree 2" % str(node))

        assert not errors

    def test_cube_centers(self):
        errors = []
        for gc in self.gray_coords:
            gc_ = np.array(gc)
            cube_center = np.ones(3) * 0.5 + gc_
            if self.env.distance_to_wall(cube_center) != approx(0.5):
                errors.append("node %s transform is off" % str(gc))

        assert not errors

    def test_walls_are_where_we_want_them_end_cubes(self):
        # (0, 0, 0) -> (0, 0, 1)
        errors = []

        # away from z axis
        afz = np.array([0.5, 0.5, 0.8])
        if self.env.distance_to_wall(afz) != approx(0.5):
            errors.append("wall incorrectly placed in only path from start")

        # close to z axis
        c2z = np.array([0.5, 0.5, 0.1])
        if self.env.distance_to_wall(c2z) != approx(0.10):
            errors.append("distance mis-measured at start close to z axis: %f" % self.env.distance_to_wall(c2z))

        assert not errors

    def test_walls_are_where_we_want_them_within_one_gray_unit(self):
        # testing (0, 0, 1) <-> (0, 0, 0), (0, 1, 1)
        errors = []
        # close to cube in -z dir
        c2y = np.array([0.5, 0.5, 1.1])
        if self.env.distance_to_wall(c2y) != approx(0.5):
            errors.append("wall incorrectly placed in -z direction of (0, 0, 1)")

        # close to cube in y dir
        c2z = np.array([0.5, 0.9, 1.5])
        if self.env.distance_to_wall(c2z) != approx(0.5):
            errors.append("wall incorrectly placed in y direction of (0, 0, 1)")

        assert not errors

    def test_walls_are_where_we_want_them_across_gray_units(self):

        # ensure that (0, 1, 0) -> (1, 1, 0) are properly linked up

        c2x = np.array([0.95, 1.5, 0.5])
        assert self.env.distance_to_wall(c2x) == approx(0.5), \
            ("wall incorrectly placed in x dir of (0, 1, 0), %s" %
             str(
                 list(self.env.no_walls_graph.neighbors((0, 1, 0)))
             ))

    def test_end_block_sampling_transform(self):
        # testing end block (3, 0, 0) <- (3, 0, 1)
        cube_coords = (3, 0, 0)
        neighbor_coords = (3, 0, 1)
        errors = []

        sample_corner = np.zeros(3)
        transformed_corner_in_center = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, EndCuboidRegions.CENTER
        )

        if transformed_corner_in_center != approx(np.array(cube_coords, dtype='float')):
            errors.append('incorrect transform for center: is %s, should be %s'
                          % (str(transformed_corner_in_center), str(np.array(cube_coords)))
                          )

        transformed_corner_in_passage = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, EndCuboidRegions.PASSAGE
        )

        if transformed_corner_in_passage != approx(np.array(neighbor_coords, dtype='float')):
            errors.append('incorrect transform for end passage: is %s, should be %s'
                          % (str(transformed_corner_in_center), str(np.array(neighbor_coords)))
                          )

    def test_mid_block_sampling_transform(self):
        # looking at connection (2, 0, 1) -> (2, 1, 1) -> (2, 1, 0)
        errors = []

        pred_coords = (2, 0, 1)
        cube_coords = (2, 1, 1)
        succ_coords = (2, 1, 0)

        sample_corner = np.zeros(3)

        transformed_corner_in_prev_passage = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, MidCuboidRegions.PASSAGE1
        )

        if transformed_corner_in_prev_passage != approx(np.array(cube_coords, dtype='float')):
            errors.append(
                'incorrect transform for successor passage: is %s, should be %s'
                % (str(transformed_corner_in_prev_passage), str(np.array(cube_coords)))
            )

        transformed_corner_in_center = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, MidCuboidRegions.CENTER
        )

        if transformed_corner_in_center != approx(np.array(cube_coords, dtype='float')):
            errors.append(
                'incorrect transform for center: is %s, should be %s'
                % (str(transformed_corner_in_center), str(np.array(cube_coords)))
            )

        transformed_corner_in_succ_passage = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, MidCuboidRegions.PASSAGE2
        )

        if transformed_corner_in_succ_passage != approx(np.array(succ_coords, dtype='float')):
            errors.append(
                'incorrect transform for center: is %s, should be %s'
                % (str(transformed_corner_in_center), str(np.array(succ_coords)))
            )
