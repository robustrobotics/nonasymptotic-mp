import numpy as np
import pynndescent as pynn
import networkit as nk

from itertools import product


# TODO: easy speedup: add to graph in bulk
class SimplePRM:
    def __init__(self, connection_rad, motion_validity_checker, valid_state_sampler, k_connection_neighbors=20):
        self.d = valid_state_sampler().size  # dummy sample to compute dimension

        self.conn_r = connection_rad
        self.k_conn_neighbors = k_connection_neighbors

        self.check_motion = motion_validity_checker
        self.sample_state = valid_state_sampler

        self.samples = None
        self.nn_index = None

        self.g_prm = None
        self.g_cc = None

    def grow_to_n_samples(self, n_samples):
        batch_size = 64

        # sample new states
        if self.samples is None:  # if new, initialize everything
            self.samples = np.zeros((n_samples, self.d))

            for i in range(n_samples):
                self.samples[i, :] = self.sample_state()

            self.nn_index = pynn.NNDescent(self.samples,
                                           verbose=True,
                                           n_neighbors=self.k_conn_neighbors)  # Euclidean metric is default
            # TODO: tinker with the NN tree parameters

            self.g_prm = nk.Graph(n_samples, weighted=True)

            for i_batch in range(0, n_samples, batch_size):
                query_node_batch = self.samples[i_batch:i_batch + batch_size]
                indices, distances = self.nn_index.query(query_node_batch)

                for i_node in range(indices.shape[0]):
                    node_i = query_node_batch[i_node]

                    for j_neighbor, d_ij in zip(indices[i_node], distances[i_node]):
                        neighbor_j = self.samples[j_neighbor]

                        if self.check_motion(node_i, neighbor_j):

                            if d_ij < self.conn_r:
                                self.g_prm.addEdge(i_batch + i_node, j_neighbor,
                                                   w=d_ij, checkMultiEdge=True)

        else:  # otherwise, we reuse past computation
            past_n_samples = self.samples.shape[0]
            m_new_samples = n_samples - past_n_samples
            new_samples = np.zeros((m_new_samples, self.d))

            for i in range(m_new_samples):
                new_samples[i, :] = self.sample_state()

            self.nn_index.update(xs_fresh=new_samples)
            self.samples = np.concatenate([self.samples, new_samples])

            self.g_prm.addNodes(m_new_samples)

            # TODO: if this is a big problem, a good amount of computation hangs here.
            # There is probably a trick of exporting to some graph visualization format
            # that will allow us to offload most of the graph loading to some cpp code.
            for i_batch in range(0, m_new_samples, batch_size):
                query_node_batch = self.samples[i_batch:i_batch + batch_size]
                indices, distances = self.nn_index.query(query_node_batch)

                for i_node in range(indices.shape[0]):
                    node_i = query_node_batch[i_node]

                    for j_neighbor, d_ij in zip(indices[i_node], distances[i_node]):
                        neighbor_j = self.samples[j_neighbor]

                        if self.check_motion(node_i, neighbor_j):
                            d_ij = distances[i_node, j_neighbor]

                            if d_ij < self.conn_r:
                                self.g_prm.addEdge(m_new_samples + i_batch + i_node, j_neighbor,
                                                   w=d_ij, checkMultiEdge=True)

        self.g_cc = nk.components.ConnectedComponents(self.g_prm)
        self.g_cc.run()

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
        spsp = nk.distance.SPSP(self.g_prm, coll_free_start_nns)
        spsp.setTargets(coll_free_goal_nns)
        spsp.run()

        prm_sols_in_and_outs = np.array(list(product(coll_free_start_nns, coll_free_goal_nns)))
        prm_sols_distances = np.array(list(map(lambda ij: spsp.getDistance(ij[0], ij[1]), prm_sols_in_and_outs)))
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
