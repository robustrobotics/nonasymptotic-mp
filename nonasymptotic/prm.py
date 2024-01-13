from scipy import sparse
from networkit.graphtools import GraphTools
from itertools import product

import numpy as np
import pynndescent as pynn
import networkit as nk
import networkx as nx


class SimplePRM:
    def __init__(self, connection_rad, motion_validity_checker, valid_state_sampler, sdf_to_path,
                 max_k_connection_neighbors=2048, seed=None, verbose=False):
        self.d = valid_state_sampler().size  # dummy sample to compute dimension

        self.conn_r = connection_rad
        self.k_neighbors = 16
        self.max_k_neighbors = max_k_connection_neighbors

        self.check_motion = motion_validity_checker
        self.sample_state = valid_state_sampler
        self.dist_points_to_path = sdf_to_path

        self.samples = None
        self.nn_index = None

        self.g_prm = None
        self.g_cc = None
        self.g_sp_lookup = None
        self.sample_to_lookup_ind = None

        self.rng_seed = seed
        self.verbose = verbose

        self.GT = GraphTools()

    def grow_to_n_samples(self, n_samples):
        # batch_size = 64

        def _build_threshold_index(samples):
            # TODO: tinker with the NN tree parameters
            while True:
                nn_index = pynn.NNDescent(samples,
                                          n_neighbors=self.k_neighbors,
                                          random_state=self.rng_seed,
                                          verbose=self.verbose)  # Euclidean metric is default
                _, index_dists = nn_index.neighbor_graph

                if self.k_neighbors >= self.max_k_neighbors or np.all(index_dists[:, -1] >= self.conn_r):
                    if self.verbose:
                        print('Using %i neighbors for graph.' % self.k_neighbors)
                    return nn_index

                self.k_neighbors *= 2

        def _nn_edge_list_and_dist_list_to_nk_prm_graph(_edge_arr, _dist_arr, start_ind=0):
            # TODO: saving a file may be faster... if slow, see what happens
            rows = np.arange(start_ind, n_samples).repeat(self.k_neighbors)
            cols = _edge_arr[start_ind:, :].flatten(order='C')
            data = _dist_arr[start_ind:, :].flatten(order='C')

            # check valid motions and connectivity
            valid_motions = self.check_motion(self.samples[rows], self.samples[cols]) & (data <= self.conn_r)

            # then filter for valid connections and construct graph
            gx = nx.from_scipy_sparse_array(
                sparse.coo_array(
                    (data[valid_motions], (rows[valid_motions], cols[valid_motions])),
                    shape=(n_samples, n_samples)
                ),
                edge_attribute='distance'
            )
            return nk.nxadapter.nx2nk(gx, weightAttr='distance')

        # sample new states
        if self.samples is None:  # if new, initialize everything
            self.samples = np.zeros((n_samples, self.d))

            for i in range(n_samples):
                self.samples[i, :] = self.sample_state()

            # apply a doubling scheme for connection neighbors to obtain the threshold graph
            self.nn_index = _build_threshold_index(self.samples)
            adj_arr, dists_arr = self.nn_index.neighbor_graph
            self.g_prm = _nn_edge_list_and_dist_list_to_nk_prm_graph(adj_arr, dists_arr)

            # self.g_prm = nk.Graph(n_samples, weighted=True)
            #
            # for i_batch in range(0, n_samples, batch_size):
            #     query_node_batch = self.samples[i_batch:i_batch + batch_size]
            #     indices, distances = self.nn_index.query(query_node_batch)
            #
            #     for i_node in range(indices.shape[0]):
            #         node_i = query_node_batch[i_node]
            #
            #         for j_neighbor, d_ij in zip(indices[i_node], distances[i_node]):
            #             neighbor_j = self.samples[j_neighbor]
            #
            #             if self.check_motion(node_i, neighbor_j):
            #
            #                 if d_ij < self.conn_r:
            #                     self.g_prm.addEdge(i_batch + i_node, j_neighbor,
            #                                        w=d_ij, checkMultiEdge=True)

        else:  # otherwise, we reuse past computation
            past_n_samples = self.samples.shape[0]
            m_new_samples = n_samples - past_n_samples
            new_samples = np.zeros((m_new_samples, self.d))

            for i in range(m_new_samples):
                new_samples[i, :] = self.sample_state()

            self.nn_index.update(xs_fresh=new_samples)
            self.samples = np.concatenate([self.samples, new_samples])

            # check to make sure we are still within threshold. if not, we need to build a new graph with
            # new neighbors
            adj_arr, dists_arr = self.nn_index.neighbor_graph
            if np.all(dists_arr[:, -1] > self.conn_r):
                self.g_prm.addNodes(m_new_samples)
                g_new_conns = _nn_edge_list_and_dist_list_to_nk_prm_graph(adj_arr, dists_arr, start_ind=past_n_samples)
                self.GT.merge(self.g_prm, g_new_conns)

                # # that will allow us to offload most of the graph loading to some cpp code.
                # for i_batch in range(0, m_new_samples, batch_size):
                #     query_node_batch = self.samples[i_batch:i_batch + batch_size]
                #     indices, distances = self.nn_index.query(query_node_batch)
                #
                #     for i_node in range(indices.shape[0]):
                #         node_i = query_node_batch[i_node]
                #
                #         for j_neighbor, d_ij in zip(indices[i_node], distances[i_node]):
                #             neighbor_j = self.samples[j_neighbor]
                #
                #             if self.check_motion(node_i, neighbor_j):
                #                 d_ij = distances[i_node, j_neighbor]
                #
                #                 if d_ij < self.conn_r:
                #                     self.g_prm.addEdge(m_new_samples + i_batch + i_node, j_neighbor,
                #                                        w=d_ij, checkMultiEdge=True)

            else:
                self.nn_index = _build_threshold_index(self.samples)
                adj_arr, dists_arr = self.nn_index.neighbor_graph
                self.g_prm = _nn_edge_list_and_dist_list_to_nk_prm_graph(adj_arr, dists_arr)

        self.g_cc = nk.components.ConnectedComponents(self.g_prm)
        self.g_cc.run()

        dist_samples_to_line = self.dist_points_to_path(self.samples)
        samples_within_conn_r = np.arange(n_samples)[dist_samples_to_line <= self.conn_r]
        spsp = nk.distance.SPSP(self.g_prm, samples_within_conn_r)
        spsp.setTargets(samples_within_conn_r)
        spsp.run()
        self.g_sp_lookup = spsp.getDistances(asarray=True)
        self.sample_to_lookup_ind = np.zeros(n_samples, dtype=np.intp)

        # we set to the n_samples to throw an error if query for a vertex outside the conn_r
        self.sample_to_lookup_ind[:] = n_samples
        self.sample_to_lookup_ind[samples_within_conn_r] = np.arange(samples_within_conn_r.shape[0])

        # self.g_spsp = spsp

    def query_best_solution(self, start, goal):
        # NOTE: if there isn't a solution... will return an infinite distance. This is
        # just a quirk of networkit that we just need to work around.

        # Returned path is excluding the endpoints

        # first, loop start and goal into graph
        indices, distances = self.nn_index.query(np.vstack([start, goal]))

        i_goal = self.g_prm.addNodes(2)
        i_start = i_goal - 1

        for j_neighbor, d_ij in zip(indices[0], distances[0]):
            neighbor_j = self.samples[j_neighbor]
            if d_ij < self.conn_r and self.check_motion(start, neighbor_j):
                self.g_prm.addEdge(i_start, j_neighbor, w=d_ij)

        for j_neighbor, d_ij in zip(indices[1], distances[1]):
            neighbor_j = self.samples[j_neighbor]
            if d_ij < self.conn_r and self.check_motion(goal, neighbor_j):
                self.g_prm.addEdge(i_goal, j_neighbor, w=d_ij)

        biDij = nk.distance.BidirectionalDijkstra(self.g_prm, i_start, i_goal)
        biDij.run()

        sol_dist = biDij.getDistance()
        sol_path = self.samples[biDij.getPath()]

        # delete start/goal from graph for next query
        self.g_prm.removeNode(i_start)
        self.g_prm.removeNode(i_goal)

        return sol_dist, sol_path

    # def query_same_component(self, v1, v2):
    #     if not self.g_prm.hasNode(v1) or not self.g_prm.hasNode(v2):
    #         raise RuntimeError('v1 or v2 not in the PRM.')
    #
    #     return self.g_cc.componentOfNode(v1) == self.g_cc.componentOfNode(v2)

    def query_all_graph_connections(self, start, goal):

        # returns a N_pairs X 2 X dim array consisting of enter/exit points in the prm graph
        # and an N_pairs vector consisting of the distances between the enter and exit points in the prm
        indices, distances = self.nn_index.query(np.vstack([start, goal]))

        start_nns = indices[0, distances[0] < self.conn_r]
        goal_nns = indices[1, distances[1] < self.conn_r]

        coll_free_start_nns = list(filter(lambda n_i: self.check_motion(start, self.samples[n_i]), start_nns))
        coll_free_goal_nns = list(filter(lambda n_i: self.check_motion(goal, self.samples[n_i]), goal_nns))

        # now we have indices, let's do some shortest path computations
        # using explicit distance call:
        # prm_sols_in_and_outs = np.array(list(product(coll_free_start_nns, coll_free_goal_nns)))
        # prm_sols_distances = np.array(list(
        #     map(lambda ij: self.g_sp_lookup.getDistance(ij[0], ij[1]), prm_sols_in_and_outs)
        # ))

        # use advanced indexing
        prm_sols_in_and_outs = np.transpose([
            np.tile(start_nns, len(goal_nns)), np.repeat(goal_nns, len(start_nns))
        ])
        prm_sols_distances = self.g_sp_lookup[
            self.sample_to_lookup_ind[prm_sols_in_and_outs[:, 0]],
            self.sample_to_lookup_ind[prm_sols_in_and_outs[:, 1]]
        ]

        return (
            np.stack([
                self.samples[prm_sols_in_and_outs[:, 0]],
                self.samples[prm_sols_in_and_outs[:, 1]]
            ], axis=0).swapaxes(0, 1),
            prm_sols_distances
        )

    def num_vertices(self):
        return self.g_prm.numberOfNodes() if self.g_prm is not None else 0

    def num_edges(self):
        return self.g_prm.numberOfEdges() if self.g_prm is not None else 0

    def reset(self):
        self.g_prm = None
        self.g_cc = None
        self.samples = None
        self.nn_index = None

    def save(self, filepath):
        if self.g_prm is not None:
            nk.writeGraph(self.g_prm, filepath, nk.Format.NetworkitBinary)
        else:
            raise RuntimeWarning('Tried to save an uninitialized PRM.')


if __name__ == '__main__':
    from envs import GrayCodeWalls

    walls = GrayCodeWalls(2, 2, 0.1)
    prm = SimplePRM(0.2, walls.is_motion_valid, walls.sample_from_env)
    prm.grow_to_n_samples(1000)
    print('hi')
