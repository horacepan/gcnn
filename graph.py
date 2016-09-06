class Graph(object):

    def __init__(self, adj_mat, node_labels, edge_labels, adj_lst, labels):
        self._adj_mat = adj_mat
        self._node_labels = node_labels
        self._edge_labels = edge_labels
        self._adj_lst = adj_lst
        self._labels = labels
        self._vlabels = sorted(node_labels.values()) # almost always numerical so default sort
        self._elabels = sorted(edge_labels.values())

    def size(self):
        return len(self._labels)

    def labels(self, vert):
        if type(vert) == list:
            return map(lambda x: self._node_labels.get(x, None), vert)
        else:
            return self._node_labels.get(vert, None)

    def neighbors(self, vert):
        if vert > size or vert <= 0:
            raise Exception("%d is not a valid vertice" % vert)
        return self._adj_lst[vert]

    def kneighbors(self, vert, k, sortfunc, labeled=True):
        '''
        TODO: Return a tensor of size (k, num vert labels)
        '''
        nbrs = set(self._adj_lst[vert])
        knbrs = []

        while len(nbrs) < k:
            knbrs.extend(nbrs)
            nbrs = reduce(lambda acc, x: acc.union(self.neighbors(x)), nbrs, [])

        knbrs.sort(key=sortfunc)
        knbrs = knbrs[:k]

        if labeled:
            return self.labels(knbrs)
        else:
            return knbrs

    def knbr_edges(self, vert, k, sortfunc, labeled=True):
        '''
        TODO: Return a tensor of size (k, num vert labels)
        '''
        output = np.zeros((k, k))

        knbrs = self.kneighbors(vert, k, sortfunc, labeled=False)
        for i in knbrs:
            for j in knbrs:
                if labeled:
                    output[i, j] = self._edge_labels.get((i, j), 0)
                else:
                    output[i, j] = 1 if (i, j) in self._edge_labels else 0

        return output

    def knbr_channels(self, labeled_knbrs, k, channel_type):
        labels = self._vlabels if channel_type == 'vertices' else self._elabels
        channels = np.zeros((k, len(labels))

        for ind, vlabel in enumerate(self._vlabels):
            channels[ind] = (labeled_knbrs == vlabel)

        return channels
