from sympy.combinatorics.graycode import GrayCode
import networkx as nx
import numpy as np

import copy
from enum import Enum


class MidCuboidRegions(Enum):
    PASSAGE1 = 0
    CENTER = 1
    PASSAGE2 = 2


class EndCuboidRegions(Enum):
    PASSAGE = 3
    CENTER = 4


class GrayCodeWalls:
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

        self.no_walls_graph = nx.Graph()
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

    def distance_to_wall(self, x):
        assert x.shape == (self.dim,)

        # begin by rounding to know which hypercube we are in
        cube_coords = np.floor(x).astype('int64')

        # translate to cube center
        x_c = x - (cube_coords + 0.5)

        # get neighbors so we know there are free passageways
        neighbors = list(self.no_walls_graph.neighbors(tuple(cube_coords)))

        # fill in coordinates of walls, accounting for neighbors with no walls
        walls_low = np.ones(self.dim) * -0.5
        walls_high = np.ones(self.dim) * 0.5

        for i in range(len(neighbors)):
            walls_low, walls_high = self._unblock_wall(
                cube_coords, np.array(neighbors[i]), walls_low, walls_high)

        walls_low_dists = np.abs(x_c - walls_low)
        walls_high_dists = np.abs(x_c - walls_high)

        return min(np.min(walls_low_dists), np.min(walls_high_dists)) - self.thickness

    def sample_from_env(self):
        # first, sample which cuboid we will land
        sampled_cuboid_coords = self.rng.choice(
            self.no_walls_linear_list,
            p=self.cuboid_pmf_vect
        )

        # next, sample if we are in the center or (one of/only) open wall in selected cuboid
        if (sampled_cuboid_coords == self.no_walls_linear_list[0]
                or sampled_cuboid_coords == self.no_walls_linear_list[-1]):
            sampled_region = self.rng.choice(
                [EndCuboidRegions.CENTER, EndCuboidRegions.PASSAGE],
                p=self.end_passage_regions_pmf_vect
            )
        else:
            sampled_region = self.rng.choice(
                [MidCuboidRegions.PASSAGE1, MidCuboidRegions.CENTER, MidCuboidRegions.PASSAGE2],
                p=self.mid_passage_regions_pmf_vect
            )

        # sample point from the defined region (sample from a box then do coord scale)
        # and then transform back
        unit_cube_sample = self.rng.uniform(size=(self.dim,))

        return self._transform_sample_to_global_frame(unit_cube_sample, sampled_cuboid_coords, sampled_region)

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
            adjoined_cuboid_coords = list(self.no_walls_graph.neighbors(cuboid_coords))[ix]

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
