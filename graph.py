class Graph(object):

    def __init__(self, adj_mat, node_labels, edge_labels, adj_lst, labels):
        self._adj_mat = adj_mat
        self._node_labels = node_labels
        self._edge_labels = edge_labels
        self._adj_lst = adj_lst
        self._labels = labels

    def size(self):
        return len(self._labels)

    def label(self, vert):
        return self._node_labels.get(vert, None)
