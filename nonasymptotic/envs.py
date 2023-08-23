import math

from sympy.combinatorics.graycode import GrayCode
import networkx as nx
import numpy as np
import copy


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

        self.no_walls_linear_list = [] # we'll hold onto this because the ordering start -> end is helpful
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
        vol_center = math.pow(0.5 - 2 * thickness, dim)
        vol_open_wall = thickness * math.pow(0.5 - 2 * thickness, dim - 1)
        vol_end_passage = vol_center + vol_open_wall
        vol_mid_passage = vol_center + 2 * vol_open_wall

        n_nodes = self.no_walls_graph.number_of_nodes()
        total_volume = (n_nodes - 2) * vol_mid_passage + 2 * vol_end_passage

        self.rng = np.random.default_rng()

        # probability a sample falls in an end passage/mid passage
        self.prob_end_passage = vol_end_passage / total_volume  # this is the probability of one _specific_ end passage
        self.prob_mid_passage = vol_mid_passage / total_volume

        # probability a sample falls in cube center/space opened by absence of wall
        self.prob_center_in_end_passage = vol_center / vol_end_passage
        self.prob_open_wall_in_end_passage = vol_open_wall / vol_end_passage

        self.prob_center_in_mid_passage = vol_center / vol_mid_passage
        self.prob_open_wall_in_end_passage = vol_open_wall / vol_mid_passage

        # we'll need these in later computations
        self.dim = dim
        self.length = length
        self.thickness = thickness

    def distance_to_wall(self, x):
        assert x.shape == (self.dim,)

        # begin by rounding to know which hypercube we are in
        cube_coords = np.round(x).astype('int64')

        # translate to cube center
        leveler = np.zeros(self.dim)
        leveler[0] += int(x[0])
        x_c = (x - leveler) - ((cube_coords - leveler) / 2 + 0.25)

        # get neighbors so we know there are free passageways
        neighbors = list(self.no_walls_graph.neighbors(tuple(cube_coords)))

        # fill in coordinates of walls, accounting for neighbors with no walls
        walls_low = np.ones(self.dim) * -0.25
        walls_high = np.ones(self.dim) * 0.25

        for i in range(len(neighbors)):
            walls_low, walls_high = self._unblock_wall(
                cube_coords, np.array(neighbors[i]), walls_low, walls_high)

        walls_low_dists = np.abs(x_c - walls_low)
        walls_high_dists = np.abs(x_c - walls_high)

        return min(np.min(walls_low_dists), np.min(walls_high_dists)) - self.thickness

    def sample_from_env(self):
        # first, sample which cuboid we will land

        # next, sample if we are in the center or (one of/only) open wall in selected cuboid

        # sample point from the defined region and then return it
        pass

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
