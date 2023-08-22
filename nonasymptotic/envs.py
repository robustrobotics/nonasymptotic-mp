from sympy.combinatorics.graycode import GrayCode
import networkx as nx
import numpy as np
import copy


class GrayCodePathWalls:
    def __init__(self, dim, length, thickness=0.0):
        assert dim >= 2
        assert length > 0

        gray_code = GrayCode(dim - 1)
        gray_coords = []
        for gray_str in gray_code.generate_gray():
            gray_coords.append(tuple(list(map(lambda s: int(s), [*gray_str]))))

        gray_coords_reversed = copy.deepcopy(gray_coords)
        gray_coords_reversed.reverse()

        no_walls_linear_list = []
        for i in range(length):
            if i % 2 == 0:
                no_walls_linear_list += self._prepend_coord(i, gray_coords)
            else:
                no_walls_linear_list += self._prepend_coord(i, gray_coords_reversed)

        no_walls_edge_list = list(zip(no_walls_linear_list, no_walls_linear_list[1:]))

        self.no_walls_graph = nx.Graph()
        self.no_walls_graph.add_nodes_from(no_walls_linear_list)
        self.no_walls_graph.add_edges_from(
            no_walls_edge_list)  # we could store the complement, (edge = wall) but that's larger memory

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

    @staticmethod
    def _unblock_wall(cube_coords, neighbor_coords, walls_low, walls_high):
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
        return list(
            map(
                lambda tup: (coord,) + tup,
                list_of_coords
            )
        )
