import numpy as np
import pdb

class GraphDataset(object):
    def __init__(self, graphs, labels, batch_size, params):
        self._graphs = graphs
        self._batch_size = batch_size
        self._size = len(self._graphs)
        self._avg_nodes = None
        self._index = 0
        self._all_vert_labels = self._get_all_vert_labels()
        self._all_edge_labels = self._get_all_edge_labels()
        self._data = self.gen_channels(**params)
        self._graph_labels = self.make_one_hot(labels)

    def _get_all_edge_labels(self):
        all_labels = set()
        for g in self._graphs:
            all_labels = all_labels.union(g.edge_labels())

        return list(all_labels)

    def _get_all_vert_labels(self):
        all_labels = set()
        for g in self._graphs:
            all_labels = all_labels.union(g.node_labels())

        return list(all_labels)

    def data(self):
        return self._data

    def labels(self):
        return self._graph_labels

    def avg_nodes(self):
        if self._avg_nodes == None:
            self._avg_nodes = np.mean([g.size for g in self._graphs])
        return self._avg_nodes

    def next_batch(self):
        # shuffle and reset index if necessary
        if self._index + self._batch_size > self._size:
            self._index = 0
            permutation = np.random.permutation[self._size]
            self._data = self._data[permutation]
            self._graph_labels = self._graph_labels[permutation]

        start = self._index
        end = start + batch_size
        return self._data[start:end], self._graph_labels[start:end]

    def gen_channels(self, width=4, nbr_size=4, stride=1, channel_type='vertices'):
        if channel_type == 'edges':
            num_channels = len(self._all_edge_labels)
            output_shape = (self._size, width,  nbr_size * nbr_size, num_channels)
        else:
            num_channels = len(self._all_vert_labels)
            output_shape = (self._size, width,  nbr_size, num_channels)
        output = np.zeros(output_shape)

        for index, graph in enumerate(self._graphs):
            sampled_verts = graph.sample(width, stride)
            layer = graph.kneighbors_lst(sampled_verts, nbr_size, labeled=True)
            layer_channels = graph.gen_channels(layer, self._all_vert_labels)
            output[index] = layer_channels

        return output

    def make_one_hot(self, labels, max_label=None):
        '''
        Takes in a nx1 vector of discrete labels(or a list).
        Returns nxc vector, where c is the total number of discrete labels.
        '''
        if max_label == None:
            max_label = max(labels)
        one_hot = np.zeros((len(labels), max_label))
        for i in range(len(labels)):
            one_hot[i, labels[i]-1] = 1 # assume that labels are 1 to max_labels so we -1

        return one_hot
