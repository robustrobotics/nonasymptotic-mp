from nonasymptotic.envs import GrayCodeWalls, MidCuboidRegions, EndCuboidRegions, StraightLine
from nonasymptotic.prm import SimpleRadiusPRM

from sympy.combinatorics.graycode import GrayCode
import networkx as nx
import numpy as np

from pytest import approx


class Test3dGrayCodeEnvNoThickness:
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

    def test_end_block_sampling_transform_no_thickness(self):
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
                          % (str(transformed_corner_in_passage), str(np.array(neighbor_coords)))
                          )

        assert not errors

    def test_mid_block_sampling_transform_no_thickness(self):
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
                'incorrect transform for predecessor passage: is %s, should be %s'
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

        if transformed_corner_in_succ_passage != approx(np.array(cube_coords, dtype='float')):
            errors.append(
                'incorrect transform for successor passage: is %s, should be %s'
                % (str(transformed_corner_in_succ_passage), str(np.array(cube_coords)))
            )

        assert not errors


class Test3dGrayCodeEnvWithThickness:
    env = GrayCodeWalls(3, 4, 0.1)

    def test_end_block_sampling_transform(self):
        # testing end block (0, 0, 0) -> (0, 0, 1)
        cube_coords = (0, 0, 0)
        neighbor_coords = (0, 0, 1)
        errors = []

        sample_corner = np.zeros(3)
        sample_center = np.ones(3) * 0.5

        transformed_corner_in_center = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, EndCuboidRegions.CENTER
        )

        if transformed_corner_in_center != approx(np.ones(3) * 0.1):
            errors.append('incorrect transform for corner of center: is %s, should be %s'
                          % (str(transformed_corner_in_center), str(np.array(cube_coords)))
                          )

        transformed_corner_in_passage = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, EndCuboidRegions.PASSAGE
        )

        if transformed_corner_in_passage != approx(np.array([0.1, 0.1, 0.9])):
            errors.append('incorrect transform for corner of end passage: is %s, should be %s'
                          % (str(transformed_corner_in_passage), str(np.array(np.array([0.1, 0.1, 0.9]))))
                          )

        transformed_center_in_center = self.env._transform_sample_to_global_frame(
            sample_center, cube_coords, EndCuboidRegions.CENTER
        )

        if transformed_center_in_center != approx(np.ones(3) * 0.5):
            errors.append('incorrect transform for center of center: is %s, should be %s'
                          % (str(transformed_center_in_center), str(np.ones(3) * 0.5))
                          )

        transformed_center_in_passage = self.env._transform_sample_to_global_frame(
            sample_center, cube_coords, EndCuboidRegions.PASSAGE
        )

        if transformed_center_in_passage != approx(np.array([0.5, 0.5, 0.95])):
            errors.append('incorrect transform for center of end passage: is %s, should be %s'
                          % (str(transformed_center_in_passage), str(np.array([0.5, 0.5, 0.95])))
                          )

        assert not errors

    def test_mid_block_sampling_transform(self):
        # looking at connection (2, 0, 0) -> (2, 0, 1) -> (2, 1, 1)
        #                       ^ pred ^                   ^ succ ^
        errors = []

        cube_coords = (2, 0, 1)

        sample_corner = np.zeros(3)
        sample_center = np.ones(3) * 0.5

        transformed_corner_in_prev_passage = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, MidCuboidRegions.PASSAGE1
        )

        if transformed_corner_in_prev_passage != approx(np.array([0.1, 0.1, 0.0]) + np.array(cube_coords)):
            errors.append(
                'incorrect transform for corner of predecessor passage: is %s, should be %s'
                % (str(transformed_corner_in_prev_passage), str(np.array([0.1, 0.1, 0.0]) + np.array(cube_coords)))
            )

        transformed_corner_in_center = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, MidCuboidRegions.CENTER
        )

        if transformed_corner_in_center != approx(np.ones(3) * 0.1 + np.array(cube_coords)):
            errors.append(
                'incorrect transform for corner of center: is %s, should be %s'
                % (str(transformed_corner_in_center), str(np.ones(3) + np.array(cube_coords)))
            )

        transformed_corner_in_succ_passage = self.env._transform_sample_to_global_frame(
            sample_corner, cube_coords, MidCuboidRegions.PASSAGE2
        )

        if transformed_corner_in_succ_passage != approx(np.array([0.1, 0.9, 0.1]) + np.array(cube_coords)):
            errors.append(
                'incorrect transform for corner of successor passage: is %s, should be %s'
                % (str(transformed_corner_in_succ_passage), str(np.array([0.1, 0.9, 0.1]) + np.array(cube_coords)))
            )

        transformed_center_in_prev_passage = self.env._transform_sample_to_global_frame(
            sample_center, cube_coords, MidCuboidRegions.PASSAGE1
        )

        if transformed_center_in_prev_passage != approx(np.array([0.5, 0.5, 0.05]) + np.array(cube_coords)):
            errors.append(
                'incorrect transform for center of previous passage: is %s, should be %s'
                % (str(transformed_center_in_prev_passage), str(np.array([0.5, 0.5, 0.05]) + np.array(cube_coords)))
            )

        transformed_center_in_center = self.env._transform_sample_to_global_frame(
            sample_center, cube_coords, MidCuboidRegions.CENTER
        )

        if transformed_center_in_center != approx(np.array(cube_coords) + np.ones(3) * 0.5):
            errors.append(
                'incorrect transform for center of center: is %s, should be %s'
                % (str(transformed_center_in_center), str(np.array(cube_coords)) + np.ones(3) * 0.5)
            )

        transformed_center_in_succ_passage = self.env._transform_sample_to_global_frame(
            sample_center, cube_coords, MidCuboidRegions.PASSAGE2
        )

        if transformed_center_in_succ_passage != approx(np.array([0.5, 0.95, 0.5]) + np.array(cube_coords)):
            errors.append(
                'incorrect trasnform for center of succeeding passage: is %s, should be %s'
                % (str(transformed_center_in_succ_passage), str(np.array([0.5, 0.95, 0.5]) + np.array(cube_coords)))
            )

        assert not errors

    def test_samples_are_all_within_walls(self):
        points = []
        for i in range(100000):
            points.append(self.env.sample_from_env())

        errors = []
        for p in points:
            d = self.env.distance_to_wall(p)
            if d <= 0.0:
                errors.append('%s was sampled and distance_to_wall evaluated %f' % (str(p), d))

        assert not errors


class TestGrayCodeEnvCurveRep:
    env_odd = GrayCodeWalls(3, 3, 0.1)
    n_legs_odd = 1 + 2 * 10 + 1
    curve_len_odd = n_legs_odd * 0.5

    env_even = GrayCodeWalls(3, 2, 0.1)
    n_legs_even = 1 + 2 * 6 + 1
    curve_len_even = n_legs_even * 0.5

    n_test_points = 10

    def test_first_cube(self):
        t_0_to_05 = np.linspace(0, 1.0 / self.n_legs_odd, self.n_test_points)
        arclen_t_0_to_05 = t_0_to_05 * self.curve_len_odd
        base = np.ones(3) * 0.5
        points_on_curve = base + np.hstack([
            np.zeros((self.n_test_points, 1)),
            np.zeros((self.n_test_points, 1)),
            arclen_t_0_to_05.reshape(-1, 1)
        ])

        errors = []
        for t, point in zip(t_0_to_05, points_on_curve):
            mapped_point = self.env_odd.arclength_to_curve_point(t)
            if mapped_point != approx(point):
                errors.append('Incorrect point mapping in first block: expected %s, received %s' %
                              (str(point), str(mapped_point)))

        assert not errors

    def test_last_cube_odd_length(self):
        t_neg0_to_neg05 = np.linspace(0, 1.0 / self.n_legs_odd, self.n_test_points)
        arclen_t_neg0_to_neg05 = t_neg0_to_neg05 * self.curve_len_odd
        base = np.array(self.env_odd.no_walls_linear_list[-1]) + np.ones(3) * 0.5
        points_on_curve = base + np.hstack([
            np.zeros((self.n_test_points, 1)),
            np.zeros((self.n_test_points, 1)),
            -arclen_t_neg0_to_neg05.reshape(-1, 1)
        ])

        errors = []
        for neg_t, point in zip(t_neg0_to_neg05, points_on_curve):
            mapped_point = self.env_odd.arclength_to_curve_point(1.0 - neg_t)
            if mapped_point != approx(point):
                errors.append('Incorrect point mapping in last block: expected %s, received %s' %
                              (str(point), str(mapped_point)))

    def test_last_cube_even_length(self):
        t_neg0_to_neg05 = np.linspace(0, 1.0 / self.n_legs_even, self.n_test_points)
        arclen_t_neg0_to_neg05 = t_neg0_to_neg05 * self.curve_len_even
        base = np.array(self.env_even.no_walls_linear_list[-1]) + np.ones(3) * 0.5
        points_on_curve = base + np.hstack([
            np.zeros((self.n_test_points, 1)),
            np.zeros((self.n_test_points, 1)),
            -arclen_t_neg0_to_neg05.reshape(-1, 1)
        ])

        errors = []
        for neg_t, point in zip(t_neg0_to_neg05, points_on_curve):
            mapped_point = self.env_even.arclength_to_curve_point(1.0 - neg_t)
            if mapped_point != approx(point):
                errors.append('Incorrect point mapping in last block: expected %s, received %s' %
                              (str(point), str(mapped_point)))

    def test_mid_cube_entry_leg(self):
        errors = []
        t_neg0_to_neg05 = np.linspace(0, 1.0 / self.n_legs_odd, self.n_test_points)
        arclen_t_neg0_to_neg05 = t_neg0_to_neg05 * self.curve_len_odd
        # we'll be testing block 2 and block 4 (0-relative indexing)
        # block 2: predecessor...(0, 0, 1) -> (0, 1, 1)...base
        base = np.array([0, 1, 1]) + np.ones(3) * 0.5
        points_on_curve = base + np.hstack([
            np.zeros((self.n_test_points, 1)),
            -arclen_t_neg0_to_neg05.reshape(-1, 1),
            np.zeros((self.n_test_points, 1))
        ])

        t_center_block2 = (1.0 / self.n_legs_odd) * 4
        for t_neg, point in zip(t_neg0_to_neg05, points_on_curve):
            mapped_point = self.env_odd.arclength_to_curve_point(t_center_block2 - t_neg)
            if mapped_point != approx(point):
                errors.append('Incorrect point mapping in entry leg of block 2: expected %s, received %s' %
                              (str(point), str(mapped_point)))

        # block 4: predecessor...(0, 1, 0) -> (1, 1, 0)...base
        base = np.array([1, 1, 0]) + np.ones(3) * 0.5
        points_on_curve = base + np.hstack([
            -arclen_t_neg0_to_neg05.reshape(-1, 1),
            np.zeros((self.n_test_points, 1)),
            np.zeros((self.n_test_points, 1))
        ])

        t_center_block4 = (1.0 / self.n_legs_odd) * 8
        for t_neg, point in zip(t_neg0_to_neg05, points_on_curve):
            mapped_point = self.env_odd.arclength_to_curve_point(t_center_block4 - t_neg)
            if mapped_point != approx(point):
                errors.append('Incorrect point mapping in entry leg of block 4: expected %s, received %s' %
                              (str(point), str(mapped_point)))

        assert not errors

    def test_mid_cube_exit_leg(self):
        errors = []
        t_0_to_05 = np.linspace(0, 1.0 / self.n_legs_odd, self.n_test_points)
        arclen_t_0_to_05 = t_0_to_05 * self.curve_len_odd
        # we'll be testing block 2 and block 4 (0-relative indexing)
        # block 2: base...(0, 1, 1) -> (0, 1, 0)...successor
        base = np.array([0, 1, 1]) + np.ones(3) * 0.5
        points_on_curve = base + np.hstack([
            np.zeros((self.n_test_points, 1)),
            np.zeros((self.n_test_points, 1)),
            -arclen_t_0_to_05.reshape(-1, 1),
        ])

        for t_adv, point in zip(t_0_to_05, points_on_curve):
            mapped_point = self.env_odd.arclength_to_curve_point((1.0 / self.n_legs_odd) * 4 + t_adv)
            if mapped_point != approx(point):
                errors.append('Incorrect point mapping in exit leg of block 2: expected %s, received %s' %
                              (str(point), str(mapped_point)))

        # block 4: base...(1, 1, 0) -> (1, 1, 1)...successor
        base = np.array([1, 1, 0]) + np.ones(3) * 0.5
        points_on_curve = base + np.hstack([
            np.zeros((self.n_test_points, 1)),
            np.zeros((self.n_test_points, 1)),
            arclen_t_0_to_05.reshape(-1, 1),
        ])

        for t_adv, point in zip(t_0_to_05, points_on_curve):
            mapped_point = self.env_odd.arclength_to_curve_point((1.0 / self.n_legs_odd) * 8 + t_adv)
            if mapped_point != approx(point):
                errors.append('Incorrect point mapping in exit leg of block 4: expected %s, received %s' %
                              (str(point), str(mapped_point)))

        assert not errors

    def test_cube_transition(self):
        ts = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]) / self.curve_len_even
        cube_centers = np.array(self.env_even.no_walls_linear_list) + np.ones(3) * 0.5
        divider_locations = (cube_centers[:-1] + cube_centers[1:]) / 2

        errors = []
        for ti, (_t, divider) in enumerate(zip(ts, divider_locations)):
            mapped_point = self.env_even.arclength_to_curve_point(_t)
            if mapped_point != approx(divider):
                errors.append('Inccorect point mapping in divider between blocks %i and %i, expected %s, received %s.' %
                              (ti, ti + 1, str(divider), str(mapped_point)))

        assert not errors


class TestGrayCodeEnvMotionValidityChecker:
    env = GrayCodeWalls(2, 2, 0.375)
    start_center = np.array([0.5, 0.5])
    two_block_goal_center = np.array([0.5, 1.5])
    three_block_goal_center = np.array([1.5, 1.5])

    anti_regression_env = GrayCodeWalls(2, 2, 0.1)

    def test_same_block_start_is_valid_goal_is_invalid(self):
        start = self.start_center + np.array([-0.1, -0.1])
        goal = self.start_center + np.array([0.126, 0.0])
        assert not self.env.is_motion_valid(start, goal)

    def test_same_block_start_is_invalid_goal_is_valid(self):
        start = self.start_center + np.array([0.126, 0.0])
        goal = self.start_center + np.array([-0.1, -0.1])
        assert not self.env.is_motion_valid(start, goal)

    def test_same_block_start_and_goal_valid(self):
        start = self.start_center + np.array([-0.1, -0.1])
        goal = self.start_center + np.array([0.1, 0.1])
        assert self.env.is_motion_valid(start, goal)

    def test_same_block_start_and_goal_invalid(self):
        start = self.start_center + np.array([-0.126, -0.1])
        goal = self.start_center + np.array([0.1, 0.126])
        assert not self.env.is_motion_valid(start, goal)

    def test_two_blocks_corner_clip(self):
        start = self.start_center + np.array([0.124, -0.124])
        goal = self.two_block_goal_center + np.array([0.49, -0.124])
        assert not self.env.is_motion_valid(start, goal)

    def test_two_blocks_valid_in_hallway_projection(self):
        start = self.start_center
        goal = self.two_block_goal_center
        assert self.env.is_motion_valid(start, goal)

    def test_two_blocks_valid_corner_dodge(self):
        start = self.start_center + np.array([-0.124, 0.49])
        goal = self.two_block_goal_center + np.array([0.49, 0.124])
        assert self.env.is_motion_valid(start, goal)

    def test_two_blocks_run_along_wall(self):
        # this is a measure edge case that we'd _like" to pass, but won't
        # ever realistically happen.
        start = self.two_block_goal_center + np.array([0.0, -0.25])
        goal = self.three_block_goal_center + np.array([0.0, -0.25])
        assert self.env.is_motion_valid(start, goal)

    def test_three_blocks_corner_clip(self):
        start = self.start_center
        goal = self.three_block_goal_center + np.array([0.0, 0.1])
        assert not self.env.is_motion_valid(start, goal)

    def test_three_blocks_corner_dodge(self):
        start = self.start_center + np.array([-0.124, 0.49])
        goal = self.three_block_goal_center + np.array([-0.49, 0.124])
        assert self.env.is_motion_valid(start, goal)

    def test_four_blocks_wall_breach_at_perp(self):
        start = self.start_center
        goal = self.start_center + np.array([0.5, 0.0])
        assert not self.env.is_motion_valid(start, goal)

    def test_four_blocks_wall_breach_not_perp(self):
        start = self.start_center
        goal = self.start_center + np.array([0.5, -0.1])
        assert not self.env.is_motion_valid(start, goal)

    def test_no_opener_solve_puts_start_point_out_of_end(self):
        start = np.array([1.127, 0.26])
        goal = np.array([0.802, 0.3702])
        assert not self.anti_regression_env.is_motion_valid(start, goal)


class TestEpsilonDeltaCompletePropertyTest:

    def test_consistent_readings(self):
        env = StraightLine(dim=2, delta_clearance=0.5, seed=1)

        prm = SimpleRadiusPRM(0.1, env.is_motion_valid, env.sample_from_env, env.distance_to_path, seed=1)
        prm.grow_to_n_samples(1000)
        completes = []

        for _ in range(100):
            is_complete, _ = env.is_prm_epsilon_delta_complete(prm, 0.25, n_samples_per_check=100, timeout=60)
            completes.append(is_complete)

        assert np.all(np.array(completes)) or np.all(~np.array(completes))

    def test_timeout(self):
        env = StraightLine(dim=2, delta_clearance=0.5, seed=1)
        prm = SimpleRadiusPRM(0.1, env.is_motion_valid, env.sample_from_env, env.distance_to_path, seed=1)
        prm.grow_to_n_samples(1000)

        is_complete, info = env.is_prm_epsilon_delta_complete(prm, 0.22, n_samples_per_check=100, timeout=0.1)
        assert info == 'timed out'
