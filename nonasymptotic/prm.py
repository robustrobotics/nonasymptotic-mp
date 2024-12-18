from nonasymptotic.ann import get_ann
from networkit.graphtools import GraphTools
from abc import ABC, abstractmethod

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import networkit as nk
import uuid
import os


# create a PRM superclass that has the same query methods, and then subclass with a different constructor that will
# build a k-nearest neighbor graph and then binary search down the radius.

# create abstract properties that need to be implemented as the standard names of things

class SimplePRM(ABC):
    def __init__(self, motion_validity_checker, valid_state_sampler, seed, verbose, in_mp_exp_mode=False):
        self.check_motion = motion_validity_checker
        self.sample_state = valid_state_sampler

        self.rng_seed = seed
        self.verbose = verbose

        # directory to the temp storage directory for intermediate computations
        # I'd use tempfiles, but networkit requires string paths... and not just file-like objects
        # This ends up being cleaner, but now we need to worry about cleaning up after ourselves.
        self.temp_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../temp/'
        )

        self.in_mp_exp_mode = in_mp_exp_mode

    @abstractmethod
    def grow_to_n_samples(self, n_samples):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def _query_samples(self, query):
        pass

    @abstractmethod
    def _distance_in_graph(self, starts, goals):
        pass

    @property
    @abstractmethod
    def prm_graph(self) -> nk.Graph:
        pass

    @property
    @abstractmethod
    def prm_samples(self) -> np.ndarray:
        pass

    def query_best_solution(self, start, goal):
        # NOTE: if there isn't a solution... will return an infinite distance. This is
        # just a quirk of networkit that we just need to work around.
        
        # Returned path is excluding the endpoints
        # first, loop start and goal into graph
        start_nns_ids, start_nns_dists = self._query_samples(start)
        goal_nns_ids, goal_nns_dists = self._query_samples(goal)

        i_goal = self.prm_graph.addNodes(2)
        i_start = i_goal - 1

        # adding edge locally in a loop is faster than coming up with a big sparse
        # adjacency matrix and merging a converted graph in.
        for s_neighbor, d_sn in zip(start_nns_ids, start_nns_dists):
            self.prm_graph.addEdge(i_start, s_neighbor, w=d_sn)

        for g_neighbor, d_gn in zip(goal_nns_ids, goal_nns_dists):
            self.prm_graph.addEdge(i_goal, g_neighbor, w=d_gn)

        biDij = nk.distance.BidirectionalDijkstra(self.prm_graph, i_start, i_goal)
        biDij.run()

        sol_dist = biDij.getDistance()
        sol_path = self.prm_samples[biDij.getPath()]

        # delete start/goal from graph for next query
        self.prm_graph.removeNode(i_start)
        self.prm_graph.removeNode(i_goal)

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

        prm_sols_distances = self._distance_in_graph(prm_sols_in_and_outs[:, 0], prm_sols_in_and_outs[:, 1])

        return (
            np.stack([
                self.prm_samples[prm_sols_in_and_outs[:, 0]],
                self.prm_samples[prm_sols_in_and_outs[:, 1]]
            ], axis=0).swapaxes(0, 1),
            prm_sols_distances,
            prm_sols_in_and_outs
        )

    def _nn_edge_list_and_dist_list_to_nk_prm_graph(self, _edge_arr, _dist_arr,
                                                    include_starting=0, threshold_rad=np.inf):
        """
        :param _edge_arr: edge array where ith index lists the indices of neighbors
        :param _dist_arr: corresponding weighting array
        :param include_starting: only connect vertices starting at include_starting's index.
                                 helpful to only add new edges.
        :param threshold_rad: optional threshold radius. include edges only if weight is less than threshold_rad.
        :return: corresponding Networkit graph.
        """
        n_samples = _edge_arr.shape[0]
        k_neighbors = _edge_arr.shape[1]

        starts = np.arange(include_starting, n_samples).repeat(k_neighbors)
        goals = _edge_arr[include_starting:, :].flatten(order='C')
        dists = _dist_arr[include_starting:, :].flatten(order='C')

        within_conn_r = dists <= threshold_rad

        starts_within_conn_r, goals_within_conn_r, dists_within_conn_r = (starts[within_conn_r],
                                                                          goals[within_conn_r],
                                                                          dists[within_conn_r])

        # check valid motions and connectivity
        valid_motions = self.check_motion(self.prm_samples[starts_within_conn_r],
                                          self.prm_samples[goals_within_conn_r])

        # it appears that the graph construction is still buggy (segfaults often)
        # but networkx also just constructs graphs with for loops, so this isn't slower
        # TODO: file a git issue
        new_graph = nk.Graph(n_samples, weighted=True)
        # new_graph.addEdges(
        #     (
        #         dists_within_conn_r[valid_motions],
        #         (starts_within_conn_r[valid_motions], goals_within_conn_r[valid_motions])
        #     ),
        #     checkMultiEdge=True
        # )
        for dist, start, goal in zip(dists_within_conn_r[valid_motions],
                                     starts_within_conn_r[valid_motions],
                                     goals_within_conn_r[valid_motions]):
            new_graph.addEdge(start, goal, w=dist, checkMultiEdge=True)

        return new_graph

    def num_vertices(self):
        return self.prm_graph.numberOfNodes() if self.prm_graph is not None else 0

    def num_edges(self):
        return self.prm_graph.numberOfEdges() if self.prm_graph is not None else 0

    def save(self, filepath):
        """
        :param filepath: file directory (without extension, since multiple files need to be saved)
        """
        if self.prm_graph is not None:
            nk.writeGraph(self.prm_graph, filepath + '.nkb', nk.Format.NetworkitBinary)
        else:
            raise RuntimeWarning('Tried to save an uninitialized PRM.')

        if self.prm_samples is not None:
            np.save(filepath + '.npy', self.prm_samples)

    def _compute_spsp(self, samples_to_spsp):
        n_samples = self.prm_samples.shape[0]

        spsp = nk.distance.SPSP(self.prm_graph, samples_to_spsp)
        spsp.setTargets(samples_to_spsp)
        spsp.run()

        # array-ify the lookup
        g_sp_lookup = spsp.getDistances(asarray=True)
        sample_to_lookup_ind = np.zeros(n_samples, dtype=np.intp)
        # we set to the n_samples to throw an error if query for a vertex outside the conn_r
        sample_to_lookup_ind[:] = n_samples
        sample_to_lookup_ind[samples_to_spsp] = np.arange(samples_to_spsp.shape[0])

        return g_sp_lookup, sample_to_lookup_ind


class SimpleFullConnRadiusPRM(SimplePRM):
    def __init__(self, connection_rad, motion_validity_checker, valid_state_sampler, seed=None, verbose=False):
        """
        This PRM just explicitly enumerates all the possible connections (and vectorizes the check),
        and then filters by the connection radius. Only tractable to compute in low sample counts (<1000).
        """
        super().__init__(motion_validity_checker, valid_state_sampler, seed, verbose)
        self.conn_r = connection_rad
        self.max_conn_r = connection_rad

        self._g_prm = None
        self._samples = None
        self.d = valid_state_sampler().size

        self.tmp_graph_cache_path = os.path.join(self.temp_dir, str(uuid.uuid4()) + '.nkbg003')

    def grow_to_n_samples(self, n_samples):

        if self._samples is None:  # if new, initialize everything
            self._samples = np.zeros((n_samples, self.d))

            for i in range(n_samples):
                self._samples[i, :] = self.sample_state()
        else:
            past_n_samples = len(self._samples)
            n_new_samples = n_samples - past_n_samples

            if n_new_samples < 0:
                raise ArithmeticError(
                    'PRM is already %i large, cannot grow to %i samples.' % (past_n_samples, n_samples)
                )

            new_samples = np.zeros((n_new_samples, self.d))

            for i in range(n_new_samples):
                new_samples[i, :] = self.sample_state()

            self._samples = np.concatenate([self._samples, new_samples])

        # product between two objects using tile and repeat
        outer_loop_elts = np.repeat(self._samples, n_samples, axis=0)
        outer_loop_args = np.repeat(np.arange(n_samples), n_samples)

        inner_loop_elts = np.tile(self._samples, (n_samples, 1))
        inner_loop_args = np.tile(np.arange(n_samples), n_samples)

        # compute distances appropriately.
        dists = np.linalg.norm(inner_loop_elts - outer_loop_elts, axis=1)
        within_conn_r = dists <= self.conn_r

        starts_in_range, goals_in_range = outer_loop_args[within_conn_r], inner_loop_args[within_conn_r]
        dists_in_range = dists[within_conn_r]

        valid_motions = self.check_motion(self._samples[starts_in_range], self._samples[goals_in_range])

        master_graph = nk.Graph(n_samples, weighted=True)
        valid_dists_in_range = dists_in_range[valid_motions]
        for dist, start, goal in zip(valid_dists_in_range,
                                     starts_in_range[valid_motions],
                                     goals_in_range[valid_motions]):
            master_graph.addEdge(start, goal, w=dist, checkMultiEdge=True)

        master_graph.indexEdges()
        nk.writeGraph(master_graph, self.tmp_graph_cache_path, nk.Format.NetworkitBinary)

        self._g_prm = master_graph
        # self.max_conn_r = self.conn_r  # update max_conn_r to reflect rad of master graph.

        return valid_dists_in_range

    def set_connection_radius(self, new_conn_r):
        # there may be a way to do this with networkit.sparsification, but I can't find it

        # if above max, recompute graph.
        # if new_conn_r > self.max_conn_r:
        self.conn_r = new_conn_r # TODO: conn_r is now poorly handled
        # just trigger a rebuild.
        self.grow_to_n_samples(self._samples.shape[0])
        # return

        # if the requested conn_r is below max, then we are asking for a subgraph of the
        # master graph. Reload master graph.
        # if new_conn_r > self.conn_r:
        #     self._g_prm = nk.readGraph(self.tmp_graph_cache_path, nk.Format.NetworkitBinary)

        # then iterate over and remove the edges that are too large
        # (if master graph needed to be reloaded or otherwise)
        # for u, v, w in self._g_prm.iterEdgesWeights():
        #     if w >= new_conn_r:
        #         self._g_prm.removeEdge(u, v)

        # self.conn_r = new_conn_r

    def reset(self):
        self._samples = None
        self._g_prm = None

        # delete the cache file
        if os.path.exists(self.tmp_graph_cache_path):
            os.remove(self.tmp_graph_cache_path)

    def _query_samples(self, query):
        dists_from_query = np.linalg.norm(self._samples - query, axis=1)
        within_conn_r = dists_from_query <= self.conn_r

        points_within_conn_r = self._samples[within_conn_r]
        ids_within_conn_r = np.arange(self._samples.shape[0])[within_conn_r]

        valid_motions = self.check_motion(
            np.tile(query, (points_within_conn_r.shape[0], 1)),
            points_within_conn_r)
        return ids_within_conn_r[valid_motions], dists_from_query[within_conn_r][valid_motions]

    def _distance_in_graph(self, starts, goals):
        return NotImplementedError('Cannot compute graph distance efficiently with this version.')

    @property
    def prm_graph(self) -> nk.Graph:
        return self._g_prm

    @property
    def prm_samples(self) -> np.ndarray:
        return self._samples


class SimpleNearestNeighborRadiusPRM(SimplePRM):
    """
    A K-NN PRM adapted to be a radius PRM. Radius thresholds are implemented by set_connection_radius(),
    and but will automatically be
    cleared when the PRM is grown. For us, the experiments turned out to be
    more elegant if we took a full K-NN (as PyNNDescent/Kgraph would compute it) and then find the radius
    where path sufficiency checks fail.
    """

    def __init__(self, k_neighbors, motion_validity_checker, valid_state_sampler, sdf_to_path,
                 truncate_to_eff_rad=True, seed=None, verbose=False):
        super().__init__(motion_validity_checker, valid_state_sampler, seed, verbose)

        self.d = valid_state_sampler().size
        self.max_k_neighbors = k_neighbors
        self.k_neighbors = k_neighbors
        self.conn_r = None

        self.truncate_to_eff_rad = truncate_to_eff_rad
        self.certified_max_conn_r = None  # this is the maximal conn_r that we know will recapture the correct PRM graph
        self.dist_points_to_path = sdf_to_path

        self.g_sp_lookup = None
        self.sample_to_lookup_ind = None

        self._samples = None
        self._g_prm = None

        self.master_edges = None
        self.master_dists = None

        # create a temporary graph cache file
        self.tmp_graph_cache_path = os.path.join(self.temp_dir, str(uuid.uuid4()) + '.nkbg003')

    @property
    def prm_graph(self) -> nk.Graph:
        return self._g_prm

    @property
    def prm_samples(self) -> np.ndarray:
        return self._samples

    def grow_to_n_samples(self, n_samples):

        self.conn_r = None  # clear the connection radius

        if self._samples is None:  # if new, initialize everything
            self._samples = np.zeros((n_samples, self.d))

            for i in range(n_samples):
                self._samples[i, :] = self.sample_state()
        else:
            past_n_samples = len(self._samples)
            n_new_samples = n_samples - past_n_samples

            if n_new_samples <= 0:
                raise ArithmeticError(
                    'PRM is already %i large, cannot grow to %i samples.' % (past_n_samples, n_samples)
                )

            new_samples = np.zeros((n_new_samples, self.d))

            for i in range(n_new_samples):
                new_samples[i, :] = self.sample_state()

            self._samples = np.concatenate([self._samples, new_samples])

        # build the index
        # if we are growing the graph, it means that a previous check with a larger radius worked.
        # so know the PRM is complete for a larger radius, so we can lose NNs losslessly.
        effective_k = min(n_samples - 1, self.k_neighbors)

        # build the master graph
        ann_builder = get_ann(name="kgraph")  # will default to pynndescent if not available
        self.master_edges, self.master_dists = ann_builder.new_graph_from_data(self._samples, effective_k)
        master_graph = self._nn_edge_list_and_dist_list_to_nk_prm_graph(self.master_edges, self.master_dists)
        master_graph.indexEdges()

        # we'll save the master graph -- don't want to hold multiple PRMs in RAM.
        nk.writeGraph(master_graph, self.tmp_graph_cache_path, nk.Format.NetworkitBinary)

        # then write in the master graph.
        self._g_prm = master_graph

        self.certified_max_conn_r = np.min(
            self.master_dists[:, -1])  # the closest kth neighbor makes the certified conn_r
        if self.verbose:
            print("Certified maximal correct connection radius: %f" % self.certified_max_conn_r)
            
        # NOT true Knn from points on line, but not important for us
        if self.in_mp_exp_mode:
            n_samples = self._samples.shape[0]
            dist_samples_to_line = self.dist_points_to_path(self._samples)
            samples_within_conn_r = np.arange(n_samples)[dist_samples_to_line <= self.certified_max_conn_r]
            self.g_sp_lookup, self.sample_to_lookup_ind = self._compute_spsp(samples_within_conn_r)

        # set the new connection radius
        if self.truncate_to_eff_rad:
            self.set_connection_radius(self.certified_max_conn_r)

            # returned the certified max and the neighbor dists (since they will be used in experiment runs)
            # it's a bit of a kludge, but we return here so we do not need to duplicate the sorted dists
            # (could make gigabytes of a difference)
            return self.certified_max_conn_r, np.unique(self.master_dists, axis=None)

    def set_nearest_neighbors(self, new_k_nearest_neighbors):
        if new_k_nearest_neighbors > self.max_k_neighbors:
            raise ArithmeticError('Cannot grow KNN PRM past the original set number of neighbors.')

        # if we need to grow the graph, then reload the master graph.
        if new_k_nearest_neighbors > self.k_neighbors:
            self._g_prm = nk.readGraph(self.tmp_graph_cache_path, nk.Format.NetworkitBinary)

        # next, throw away edges we don't need.
        throw_away_edge_list = self.master_edges[:, new_k_nearest_neighbors:]
        for u, vs in enumerate(throw_away_edge_list):
            for v in vs:
                if self._g_prm.hasEdge(u, v):
                    self._g_prm.removeEdge(u, v)

        self.k_neighbors = new_k_nearest_neighbors

        if self.in_mp_exp_mode:
            # recompute shortest paths over new graph and store their distances
            n_samples = self._samples.shape[0]
            dist_samples_to_line = self.dist_points_to_path(self._samples)
            samples_within_conn_r = np.arange(n_samples)[dist_samples_to_line <= self.conn_r]
            self.g_sp_lookup, self.sample_to_lookup_ind = self._compute_spsp(samples_within_conn_r)

    def set_connection_radius(self, new_conn_r):
        # there may be a way to do this with networkit.sparsification, but I can't find it
        if new_conn_r > self.certified_max_conn_r:
            print("Warning: asking for a connection radius larger than maximum certified. "
                  "Behavior may be inconsistent.")

        # if we are growing the radius, we need to reload the master graph
        if self.conn_r is not None and new_conn_r > self.conn_r:
            self._g_prm = nk.readGraph(self.tmp_graph_cache_path, nk.Format.NetworkitBinary)

        # then iterate over and remove the edges that are too large
        for u, v, w in self._g_prm.iterEdgesWeights():
            if w >= new_conn_r:
                self._g_prm.removeEdge(u, v)

        self.conn_r = new_conn_r

        if self.in_mp_exp_mode:
            # recompute shortest paths over new graph and store their distances
            n_samples = self._samples.shape[0]
            dist_samples_to_line = self.dist_points_to_path(self._samples)
            samples_within_conn_r = np.arange(n_samples)[dist_samples_to_line <= self.conn_r]
            self.g_sp_lookup, self.sample_to_lookup_ind = self._compute_spsp(samples_within_conn_r)

    def reset(self):
        self.conn_r = None

        self.certified_max_conn_r = None  # this is the maximal conn_r that we know will recapture the correct PRM graph

        self.g_sp_lookup = None
        self.sample_to_lookup_ind = None

        self._samples = None
        self._g_prm = None

        # delete the cache file
        if os.path.exists(self.tmp_graph_cache_path):
            os.remove(self.tmp_graph_cache_path)

    def _query_samples(self, query):
        dists_from_query = np.linalg.norm(self._samples - query, axis=1)
        within_conn_r = dists_from_query <= self.conn_r if self.conn_r is not None \
            else dists_from_query <= self.certified_max_conn_r

        points_within_conn_r = self._samples[within_conn_r]
        ids_within_conn_r = np.arange(self._samples.shape[0])[within_conn_r]

        valid_motions = self.check_motion(
            np.tile(query, (points_within_conn_r.shape[0], 1)),
            points_within_conn_r)
        return ids_within_conn_r[valid_motions], dists_from_query[within_conn_r][valid_motions]

    def _distance_in_graph(self, starts, goals):
        if not self.in_mp_exp_mode:
            raise NotImplementedError('Cannot compute distance when in_mp_exp_mode is False.')

        return self.g_sp_lookup[
            self.sample_to_lookup_ind[starts],
            self.sample_to_lookup_ind[goals]
        ]


class SimpleRadiusPRM(SimplePRM):
    def __init__(self, connection_rad, motion_validity_checker, valid_state_sampler, sdf_to_path,
                 max_k_connection_neighbors=512, seed=None, verbose=False):
        """
        This PRM constructs a KNN, and doubles K until every node is guaranteed to be connected to all nodes with
        the specified connection radius.
        """
        super().__init__(motion_validity_checker, valid_state_sampler, seed, verbose)
        self.d = valid_state_sampler().size  # dummy sample to compute dimension

        self.conn_r = connection_rad
        self.k_neighbors = 16
        self.max_k_neighbors = max_k_connection_neighbors

        self.dist_points_to_path = sdf_to_path

        self.ann = None
        self.g_sp_lookup = None
        self.sample_to_lookup_ind = None

        self.GT = GraphTools()

        # these are attributes that are accessed by properties for code-sharing inheritance/niceness,
        # so they are underscored.
        self._samples = None
        self._g_prm = None

    def grow_to_n_samples(self, n_samples):

        def _build_threshold_index(samples):
            ann_builder = get_ann("kgraph")
            while True:
                edge_lists, dists = ann_builder.new_graph_from_data(samples, self.k_neighbors)

                if (self.k_neighbors >= self.max_k_neighbors
                        or self.k_neighbors >= n_samples
                        or np.all(dists[:, -1] >= self.conn_r)):
                    if self.verbose:
                        print('Using %i neighbors for graph.' % self.k_neighbors)
                    return edge_lists, dists, ann_builder

                self.k_neighbors *= 2

        # sample new states
        if self._samples is None:  # if new, initialize everything
            self._samples = np.zeros((n_samples, self.d))

            for i in range(n_samples):
                self._samples[i, :] = self.sample_state()

            # apply a doubling scheme for connection neighbors to obtain the threshold graph
            adj_arr, dists_arr, self.ann = _build_threshold_index(self._samples)
            self._g_prm = self._nn_edge_list_and_dist_list_to_nk_prm_graph(adj_arr, dists_arr)

        else:  # otherwise, we (try to) reuse past computation
            past_n_samples = self._samples.shape[0]
            m_new_samples = n_samples - past_n_samples
            new_samples = np.zeros((m_new_samples, self.d))

            for i in range(m_new_samples):
                new_samples[i, :] = self.sample_state()

            self._samples = np.concatenate([self._samples, new_samples])
            adj_arr, dists_arr = self.ann.update_graph_with_data(new_samples)

            # check to make sure we are still within threshold. if not, we need to build a new graph with
            # new neighbors
            if np.all(dists_arr[:, -1] > self.conn_r) or self.k_neighbors >= self.max_k_neighbors:
                self.prm_graph.addNodes(m_new_samples)
                g_new_conns = self._nn_edge_list_and_dist_list_to_nk_prm_graph(adj_arr, dists_arr,
                                                                               include_starting=past_n_samples,
                                                                               threshold_rad=self.conn_r)
                self.GT.merge(self._g_prm, g_new_conns)

            else:
                self.k_neighbors *= 2  # a bit hacky, but a way to make sure we don't recompute the graph at the same K
                adj_arr, dists_arr = self.ann.new_graph_from_data(self.prm_samples, self.k_neighbors)
                self._g_prm = self._nn_edge_list_and_dist_list_to_nk_prm_graph(adj_arr, dists_arr,
                                                                               threshold_rad=self.conn_r)

        if self.in_mp_exp_mode:
            n_samples = self._samples.shape[0]
            dist_samples_to_line = self.dist_points_to_path(self._samples)
            samples_within_conn_r = np.arange(n_samples)[dist_samples_to_line <= self.conn_r]
            self.g_sp_lookup, self.sample_to_lookup_ind = self._compute_spsp(samples_within_conn_r)

    def _query_samples(self, query):
        # Brute force search on and validity check. We're avoiding the PRM index now.
        dists_from_query = np.linalg.norm(self.prm_samples - query, axis=1)
        within_conn_r = dists_from_query <= self.conn_r

        points_within_conn_r = self.prm_samples[within_conn_r]
        ids_within_conn_r = np.arange(self.prm_samples.shape[0])[within_conn_r]

        valid_motions = self.check_motion(
            np.tile(query, (points_within_conn_r.shape[0], 1)),
            points_within_conn_r)
        return ids_within_conn_r[valid_motions], dists_from_query[within_conn_r][valid_motions]

    def _distance_in_graph(self, starts, goals):

        if not self.in_mp_exp_mode:
            raise NotImplementedError('Cannot compute distance when in_mp_exp_mode is False.')

        return self.g_sp_lookup[
            self.sample_to_lookup_ind[starts],
            self.sample_to_lookup_ind[goals]
        ]

    def reset(self):
        self._g_prm = None
        self._samples = None
        self.ann = None

    @property
    def prm_graph(self) -> nk.Graph:
        return self._g_prm

    @property
    def prm_samples(self) -> np.ndarray:
        return self._samples


def animate_knn_prm(_prm, _sol, node_batches=5, edge_batches=5, interval=50, animation_embed_limit=None):
    if animation_embed_limit is not None:
        import matplotlib
        matplotlib.rcParams['animation.embed_limit'] = animation_embed_limit

    fig, ax = plt.subplots()
    x_min, x_max = np.min(_prm.prm_samples[:, 0]), np.max(_prm.prm_samples[:, 0])
    y_min, y_max = np.min(_prm.prm_samples[:, 1]), np.max(_prm.prm_samples[:, 1])

    ax.set_xlim([int(x_min) - 1, int(x_max) + 1])
    ax.set_ylim([int(y_min) - 1, int(y_max) + 1])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # first, generate the frame information.
    n_verts = _prm.num_vertices()
    k = _prm.k_neighbors

    sampling_frames = []
    for i in range(int(n_verts / node_batches) + 1):
        sampling_frames.append(("sampling", (i + 1) * node_batches))

    # next, the edge connection frames
    edge_frames = []
    all_edges = list(_prm.prm_graph.iterEdges())

    for i in range(int(len(all_edges) / edge_batches) + 1):
        edge_frames.append(("connecting", all_edges[i * edge_batches: (i + 1) * edge_batches]))

    # then, the path render/turn graph red frame
    path_frames = []
    if len(_sol) > 0:
        for i in range(len(_sol) - 1):
            path_frames.append(("pathing", i))

    else:
        path_frames += [("non-pathing", None)] * 10

    frames = sampling_frames + edge_frames + path_frames + [('hold', None)] * 20

    edge_artists = []
    vert_artist = ax.scatter([], [], s=20, c='b')

    def update(_f):
        _mode, _data = _f

        if _mode == 'sampling':
            vert_artist.set_offsets(_prm.prm_samples[:_data])
            return (vert_artist,)

        elif _mode == "connecting":
            for _e in _data:
                ux, uy = _prm.prm_samples[_e[0]]
                vx, vy = _prm.prm_samples[_e[1]]

                checking_edge_artist = ax.plot([ux, vx], [uy, vy], linestyle='-', c='b', alpha=0.1)[0]
                edge_artists.append(checking_edge_artist)

            return edge_artists[:-len(_data)]

        elif _mode == "pathing":

            u, v = _sol[_data], _sol[_data + 1]
            edge_artists.append(ax.plot([u[0], v[0]], [u[1], v[1]], c='lime', alpha=0.75)[0])

            return (edge_artists[-1],)

        elif _mode == "non-pathing":
            # turn everything red.
            vert_artist.set_color('r')
            for e_art in edge_artists:
                e_art.set_color('r')

            return (vert_artist, *edge_artists)

        else:
            return (vert_artist,)

    return animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval)






if __name__ == '__main__':
    from envs import GrayCodeWalls

    walls = GrayCodeWalls(2, 2, 0.1)
    prm = SimpleRadiusPRM(0.2, walls.is_motion_valid, walls.sample_from_env)
    prm.grow_to_n_samples(1000)
