import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path
import os
import uuid

import pynndescent as pynn  # we can always use pynndescent, since it's easy to pip install

# attempt to import kgraph, but move on if it's not there (tougher to install)
try:
    import kgraph
except ImportError:
    KGRAPH_AVAILABLE = False
else:
    KGRAPH_AVAILABLE = True

TEMP_DIR = os.path.join(Path(__file__).parent.parent, 'temp')


# thinks PRMs need from the ANN
#  Consume number of neighbors, a random seed, and samples array outputs edge list/distance list
#  "Compute K-NN graph"
# <-- required by both the K1R2 PRM and R1K2 PRMs

# update existing Knn graph with some new samples
# <-- only required by R1K2 PRM.

class ApproximateNearestNeighbor(ABC):

    @abstractmethod
    def new_graph_from_data(self, data, k_neighbors) -> (np.ndarray, np.ndarray):
        pass

    @abstractmethod
    def update_graph_with_data(self, fresh_data) -> (np.ndarray, np.ndarray):
        pass


class KGraphANN(ApproximateNearestNeighbor):
    def __init__(self):
        self.k_neighbors = None
        self.seed = None
        self.embedded_data = None
        self.save_txt_path = None

    def new_graph_from_data(self, data, k_neighbors) -> (np.ndarray, np.ndarray):
        self.k_neighbors = k_neighbors

        # kgraph only accepts data in dimensions of multiples of four, and is optimized for float32
        given_dim = data.shape[1]
        embedded_dim = np.ceil(given_dim / 4).astype(int) * 4
        self.embedded_data = np.zeros((data.shape[0], embedded_dim))
        self.embedded_data[:, :given_dim] = data.astype('float32')

        nn_index = kgraph.KGraph(self.embedded_data, 'euclidean')
        nn_index.build(reverse=0, K=self.k_neighbors, L=self.k_neighbors + 50, S=30)

        # save, since we only want the graph (not to use as an index)
        self.save_txt_path = str(os.path.join(TEMP_DIR, str(uuid.uuid4()) + '.txt'))
        nn_index.save_text(self.save_txt_path)

        # parse the text file and return
        lines = open(self.save_txt_path, 'r').readlines()

        # line 0 has the number of nodes. so skip that.
        # line i corresponds to row i-1 in the input data.
        # first string in line is # of neighbors
        # then we have NN id and distances interleaved.

        edge_list = []
        dist_list = []
        for line in lines[1:]:
            nns_ids_and_dists = line.split()
            edge_list.append(nns_ids_and_dists[1::2])
            dist_list.append(nns_ids_and_dists[2::2])

        # for numerical stability, kgraph computes distances in half \ell_2 squared.
        # so we undo that to convert.
        return np.array(edge_list, dtype='int'), np.sqrt(np.array(dist_list, dtype='float32') * 2)

    def update_graph_with_data(self, fresh_data) -> (np.ndarray, np.ndarray):
        if self.k_neighbors is None:
            raise RuntimeError('Need to call new_graph_from_data before calling update_graph_with_data.')

        old_r, old_c = self.embedded_data.shape
        fresh_r, fresh_c = fresh_data.shape
        new_data = np.zeros((old_r + fresh_r, old_c))

        new_data[:old_r, :] = self.embedded_data
        new_data[old_r:, :fresh_c] = fresh_data

        return self.new_graph_from_data(new_data, self.k_neighbors)


class PyNNDescentANN(ApproximateNearestNeighbor):
    def __init__(self):
        self.nn_index = None  # PyNN does provide an update interface,

    def new_graph_from_data(self, data, k_neighbors, seed=1998, verbose=True) -> (np.ndarray, np.ndarray):
        self.nn_index = pynn.NNDescent(
            data,
            n_neighbors=k_neighbors,
            random_state=seed,
            diversify_prob=0.0,  # no pruning
            pruning_degree_multiplier=1.0,  # node degrees stay the same
            verbose=verbose
        )
        nn_edgelist, nn_dists = self.nn_index.neighbor_graph

        return nn_edgelist, nn_dists

    def update_graph_with_data(self, fresh_data) -> (np.ndarray, np.ndarray):
        if self.nn_index is None:
            raise RuntimeError('Need to call new_graph_from_data before calling update_graph_with_data.')

        self.nn_index.update(xs_fresh=fresh_data)
        nn_edgelist, nn_dists = self.nn_index.neighbor_graph

        return nn_edgelist, nn_dists


def get_ann(name="pynndescent") -> ApproximateNearestNeighbor:
    """
    :param name: Options are "pynndescent" and "kgraph". If "kgraph" is not installed,
    then "pynndescent" will be returned instead.
    :return: An ApproximateNearestNeighbor instance.
    """

    if name == "pynndescent":
        return PyNNDescentANN()
    elif name == "kgraph":
        if KGRAPH_AVAILABLE:
            return KGraphANN()
        else:
            print("kgraph not installed, using pynndescent")
            return PyNNDescentANN()
    else:
        raise NotImplementedError("Do not have ANN library called: {}".format(name))


if __name__ == '__main__':
    rng = np.random.default_rng()
    _data = rng.uniform(size=(1000, 5)).astype('float32')
    # _ann_p = get_ann("pynndescent")
    # el, dl = _ann_p.new_graph_from_data(_data, 32)
    # print(el)
    # print(dl)
    #
    _ann_k = get_ann("kgraph")
    el, dl = _ann_k.new_graph_from_data(_data, 32)
    print(el)
    print(dl[2])

    hi = np.linalg.norm(_data[el[2]] - _data[2], axis=1)
    print(hi - dl[2])