import numpy as np
import pynndescent as pynn
import networkit as ni


class SimplePRM:
    def __init__(self, connection_rad, motion_validity_checker, valid_state_sampler, k_connection_neighbors=20):
        self.d = valid_state_sampler().size  # dummy sample to compute dimension

        self.conn_r = connection_rad
        self.k_conn_neighbors = k_connection_neighbors

        self.check_motion = motion_validity_checker
        self.sample_state = valid_state_sampler

        self.samples = None
        self.nn_index = None

    def grow_to_n_samples(self, n_samples):

        # sample new states
        if self.samples is None:  # if new, initialize everything
            self.samples = np.zeros((n_samples, self.d))

            for i in range(n_samples):
                self.samples[i, :] = self.sample_state()

            self.nn_index = pynn.NNDescent(self.samples, verbose=True)  # Euclidean metric is default
                                                          # TODO: tinker with the NN tree parameters

        else:  # otherwise, we reuse past computation
            past_n_samples = self.samples.shape[0]
            m_new_samples = n_samples - past_n_samples
            new_samples = np.zeros((m_new_samples, self.d))

            for i in range(m_new_samples):
                new_samples[i, :] = self.sample_state()

            self.nn_index.update(xs_fresh=new_samples)
            self.samples = np.concatenate([self.samples, new_samples])

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
