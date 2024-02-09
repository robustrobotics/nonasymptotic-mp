from scipy import sparse
from networkit.graphtools import GraphTools

import numpy as np
import pynndescent as pynn
import networkit as nk


class SimplePRM:
    def __init__(self, connection_rad, motion_validity_checker, valid_state_sampler, sdf_to_path,
                 max_k_connection_neighbors=512, seed=None, verbose=False):
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

        def _build_threshold_index(samples):
            while True:
                nn_index = pynn.NNDescent(samples,
                                          n_neighbors=self.k_neighbors,
                                          random_state=self.rng_seed,
                                          diversify_prob=0.0,  # prune no edges, since we're not searching.
                                          pruning_degree_multiplier=1.0,  # keep node degree same as n_neighbord
                                          verbose=self.verbose)  # Euclidean metric is default
                _, index_dists = nn_index.neighbor_graph

                if (self.k_neighbors >= self.max_k_neighbors
                        or self.k_neighbors >= n_samples
                        or np.all(index_dists[:, -1] >= self.conn_r)):
                    if self.verbose:
                        print('Using %i neighbors for graph.' % self.k_neighbors)
                    return nn_index

                self.k_neighbors *= 2

        def _nn_edge_list_and_dist_list_to_nk_prm_graph(_edge_arr, _dist_arr, include_starting=0):
            starts = np.arange(include_starting, n_samples).repeat(self.k_neighbors)
            goals = _edge_arr[include_starting:, :].flatten(order='C')
            dists = _dist_arr[include_starting:, :].flatten(order='C')
            within_conn_r = dists <= self.conn_r

            starts_within_conn_r, goals_within_conn_r, dists_within_conn_r = (starts[within_conn_r],
                                                                              goals[within_conn_r],
                                                                              dists[within_conn_r])

            # check valid motions and connectivity
            valid_motions = self.check_motion(self.samples[starts_within_conn_r],
                                              self.samples[goals_within_conn_r])

            new_graph = nk.graph.GraphFromCoo(
                (
                    dists_within_conn_r[valid_motions],
                    (starts_within_conn_r[valid_motions], goals_within_conn_r[valid_motions])
                ),
                n=n_samples,
                weighted=True,
                directed=False
            )

            new_graph.removeMultiEdges()
            return new_graph


        # sample new states
        if self.samples is None:  # if new, initialize everything
            self.samples = np.zeros((n_samples, self.d))

            for i in range(n_samples):
                self.samples[i, :] = self.sample_state()

            # apply a doubling scheme for connection neighbors to obtain the threshold graph
            self.nn_index = _build_threshold_index(self.samples)
            adj_arr, dists_arr = self.nn_index.neighbor_graph
            self.g_prm = _nn_edge_list_and_dist_list_to_nk_prm_graph(adj_arr, dists_arr)

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
            if np.all(dists_arr[:, -1] > self.conn_r) or self.k_neighbors >= self.max_k_neighbors:
                self.g_prm.addNodes(m_new_samples)
                g_new_conns = _nn_edge_list_and_dist_list_to_nk_prm_graph(adj_arr, dists_arr,
                                                                          include_starting=past_n_samples)
                self.GT.merge(self.g_prm, g_new_conns)

            else:
                self.k_neighbors *= 2  # a bit hacky, but a way to make sure we don't recompute the graph at the same K
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
        start_nns_ids, start_nns_dists = self._query_samples(start)
        goal_nns_ids, goal_nns_dists = self._query_samples(goal)

        i_goal = self.g_prm.addNodes(2)
        i_start = i_goal - 1

        # adding edge locally in a loop is faster than coming up with a big sparse
        # adjacency matrix and merging a converted graph in.
        for s_neighbor, d_sn in zip(start_nns_ids, start_nns_dists):
            self.g_prm.addEdge(i_start, s_neighbor, w=d_sn)

        for g_neighbor, d_gn in zip(goal_nns_ids, goal_nns_dists):
            self.g_prm.addEdge(i_goal, g_neighbor, w=d_gn)

        biDij = nk.distance.BidirectionalDijkstra(self.g_prm, i_start, i_goal)
        biDij.run()

        sol_dist = biDij.getDistance()
        sol_path = self.samples[biDij.getPath()]

        # delete start/goal from graph for next query
        self.g_prm.removeNode(i_start)
        self.g_prm.removeNode(i_goal)

        return sol_dist, sol_path

    def query_all_graph_connections(self, start, goal):

        # returns a N_pairs X 2 X dim array consisting of enter/exit points in the prm graph
        # and an N_pairs vector consisting of the distances between the enter and exit points in the prm
        start_nns_ids, _ = self._query_samples(start)
        goal_nns_ids, _ = self._query_samples(goal)

        # use advanced indexing
        prm_sols_in_and_outs = np.transpose([
            np.tile(start_nns_ids, len(goal_nns_ids)),
            np.repeat(goal_nns_ids, len(start_nns_ids))
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
            prm_sols_distances,
            prm_sols_in_and_outs
        )

    def _query_samples(self, query):
        # Brute force search on and validity check. We're avoiding the PRM index now.
        dists_from_query = np.linalg.norm(self.samples - query, axis=1)
        within_conn_r = dists_from_query <= self.conn_r

        points_within_conn_r = self.samples[within_conn_r]
        ids_within_conn_r = np.arange(self.samples.shape[0])[within_conn_r]

        valid_motions = self.check_motion(
            np.tile(query, (points_within_conn_r.shape[0], 1)),
            points_within_conn_r)
        return ids_within_conn_r[valid_motions], dists_from_query[within_conn_r][valid_motions]

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
        """
        :param filepath: file directory (without extension, since multiple files need to be saved)
        """
        if self.g_prm is not None:
            nk.writeGraph(self.g_prm, filepath + '.nkb', nk.Format.NetworkitBinary)
        else:
            raise RuntimeWarning('Tried to save an uninitialized PRM.')

        if self.samples is not None:
            np.save(filepath + '.npy', self.samples)


if __name__ == '__main__':
    from envs import GrayCodeWalls

    walls = GrayCodeWalls(2, 2, 0.1)
    prm = SimplePRM(0.2, walls.is_motion_valid, walls.sample_from_env)
    prm.grow_to_n_samples(1000)
    print('hi')
