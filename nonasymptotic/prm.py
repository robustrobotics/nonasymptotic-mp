import numpy as np
import pynndescent as pynn
import networkit as nk


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

    def grow_to_n_samples(self, n_samples):
        batch_size = 64

        # sample new states
        if self.samples is None:  # if new, initialize everything
            self.samples = np.zeros((n_samples, self.d))

            for i in range(n_samples):
                self.samples[i, :] = self.sample_state()

            self.nn_index = pynn.NNDescent(self.samples, verbose=True)  # Euclidean metric is default
            # TODO: tinker with the NN tree parameters

            self.g_prm = nk.Graph(n_samples, weighted=True)

            for i_batch in range(0, n_samples, batch_size):
                query_node_batch = self.samples[i_batch:i_batch + batch_size]
                indices, distances = self.nn_index.query(query_node_batch)

                for i_node in range(indices.shape[0]):
                    node_i = query_node_batch[i_node]

                    for j_neighbor in indices[i_node]:
                        neighbor_j = self.samples[j_neighbor]

                        if self.check_motion(node_i, neighbor_j):
                            d_ij = np.linalg.norm(node_i - neighbor_j)

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

            for i_batch in range(0, m_new_samples, batch_size):
                query_node_batch = self.samples[i_batch:i_batch + batch_size]
                indices, distances = self.nn_index.query(query_node_batch)

                for i_node in range(indices.shape[0]):
                    node_i = query_node_batch[i_node]

                    for j_neighbor in indices[i_node]:
                        neighbor_j = self.samples[j_neighbor]

                        if self.check_motion(node_i, neighbor_j):
                            d_ij = np.linalg.norm(node_i - neighbor_j)

                            if d_ij < self.conn_r:
                                self.g_prm.addEdge(m_new_samples + i_batch + i_node, j_neighbor,
                                                   w=d_ij, checkMultiEdge=True)

    def query_solution(self, start, goal):
        pass

    def query_same_component(self, v1, v2):
        pass

    def num_vertices(self):
        pass

    def num_edges(self):
        pass

    def reset(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


if __name__ == '__main__':
    from envs import GrayCodeWalls

    walls = GrayCodeWalls(2, 2, 0.1)
    prm = SimplePRM(0.2, walls.is_motion_valid, walls.sample_from_env)
    prm.grow_to_n_samples(100000)
