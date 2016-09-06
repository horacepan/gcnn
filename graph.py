import numpy as np
import pdb

class Graph(object):

    def __init__(self, adj_mat, node_labels, edge_labels, adj_lst, labels):
        self._adj_mat = adj_mat
        self._node_labels = node_labels
        self._edge_labels = edge_labels
        self._adj_lst = adj_lst
        self._labels = labels
        self._vlabels = sorted(node_labels.values()) # almost always numerical so default sort
        self._elabels = sorted(edge_labels.values())

    @property
    def size(self):
        return len(self._labels)

    def labels(self, vert):
        if type(vert) == list:
            return map(lambda x: self._node_labels.get(x, None), vert)
        else:
            return self._node_labels.get(vert, None)

    def neighbors(self, vert):
        if vert > self.size or vert <= 0:
            raise Exception("%d is not a valid vertice" % vert)
        return self._adj_lst[vert]

    def kneighbors(self, vert, k, sortfunc=None, labeled=True):
        '''
        TODO: Return a tensor of size (k, num vert labels)
        '''
        sortfunc = (lambda x: x) if sortfunc == None else sortfunc
        nbrs = set([vert])
        knbrs = []

        while len(knbrs) < k:
            knbrs.extend(nbrs)
            nbrs = reduce(lambda acc, x: acc.union(self.neighbors(x)), nbrs, set())

        knbrs.sort(key=sortfunc)
        knbrs = knbrs[:k]
        if labeled:
            return self.labels(knbrs)
        else:
            return knbrs

    def knbr_edges(self, vert, k, sortfunc=None, labeled=True):
        '''
        TODO: Return a tensor of size (k, num vert labels)
        '''
        sortfunc = (lambda x: x) if sortfunc == None else sortfunc
        output = np.zeros((k, k))

        knbrs = self.kneighbors(vert, k, sortfunc, labeled=False)
        for i in range(k):
            for j in range(k):
                v1 = knbrs[i]
                v2 = knbrs[j]
                if labeled:
                    output[i, j] = self._edge_labels.get((v1, v2), 0)
                else:
                    output[i, j] = 1 if (v1, v2) in self._edge_labels else 0

        return output

    def knbr_channels(self, labeled_knbrs, k, channel_type):
        labels = self._vlabels if channel_type == 'vertices' else self._elabels
        channels = np.zeros((k, len(labels)))

        for ind, vlabel in enumerate(self._vlabels):
            channels[ind] = (labeled_knbrs == vlabel)

        return channels
