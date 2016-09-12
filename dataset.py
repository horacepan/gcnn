import numpy as np
import pdb

class GraphDataset(object):
    def __init__(self, graphs, labels, batch_size, params):
        self._graphs = graphs
        self._graph_labels = labels
        self._batch_size = batch_size
        self._size = len(self._graphs)
        self._index = 0
        self._all_vert_labels = self._get_all_vert_labels()
        self._all_edge_labels = self._get_all_edge_labels()
        self._data = self.gen_channels(**params)
        pdb.set_trace()

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

    def next_batch(self):
        # shuffle and reset index if necessary
        if self._index + self._batch_size > self._size:
            self._index = 0
            permutation = np.random.permutation[self._size]
            self._data = self._data[permutation]

        return self.data[self._index: self._index + self._batch_size]

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
