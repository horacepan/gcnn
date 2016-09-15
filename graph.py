import numpy as np
import util
import pdb

class Graph(object):

    def __init__(self, adj_mat, node_labels_dict, edge_labels_dict, adj_lst_dict=None):
        self._adj_mat = adj_mat         # a nxn numpy array
        self._node_labels_dict = node_labels_dict # map of node to their label
        self._edge_labels_dict = edge_labels_dict # map of edge(i,j) to label
        self._adj_lst_dict = adj_lst_dict         # map from vertex id to list of vertex ids
        self._vertices = range(1, len(self._adj_mat) + 1)       # vertex ids will always be 1 to n
        self._distances = util.floyd_warshall(self._adj_mat)    # matrix of pairwise distances
        if adj_lst_dict == None:
            self._adj_lst_dict = util.gen_adj_lst_dict(adj_mat)

    @property
    def size(self):
        return self._adj_mat.shape[0]

    def node_labels(self):
        return set(self._node_labels_dict.values())

    def edge_labels(self):
        return set(self._edge_labels_dict.values())

    def labels(self, vert_id):
        '''
        Input: a list of vertex ids(int) or a single vertex id(int)
        Returns the labels of the given vertex/vertices. Use none if the vert_id isnt in the graph.
        '''
        if type(vert_id) == list:
            return map(lambda x: self._node_labels_dict.get(x, None), vert_id)
        else:
            return [self._node_labels_dict.get(vert_id, None)]

    def neighbors(self, vert_id):
        '''
        Returns the neighbors of the given vertex id as a list.
        vert can be a single vertex(an integer) or an iterable of vertices.
        Order of the given vertices is not guaranteed.
        '''
        if vert_id > self.size or vert_id < 1:
            raise Exception("%d is not a valid vertex id" % vert)
        if type(vert_id)!= list:
            return self._adj_lst_dict.get(vert_id, [])
        else: # vert_id is a list of vert ids
            nbrs = set()
            for v in vert_id:
                nbrs = nbrs.union(self.neighbors(v))
            return list(nbrs)

    def ball(self, vert_id, radius=None, max_size=None):
        '''
        Returns a list of vertices that lie within the given radius of the vertex
        TODO: epsilon ball with weighted adjacency matrix
        '''
        ball_vertices = set([vert_id])
        for _ in range(radius):
            ball_vertices = ball_vertices.union()

        ball_vertices = []
        temp_vertices = set([vert])

        for _ in range(radius):
            ball_vertices.extend(temp_vertices)
            temp_vert = reduce(lambda acc, x: acc.union(self.neighbors(x), temp_verts), set())

        return ball_vertices

    def ball_subgraph(self, vert_id, radius):
        '''
        Return a graph object of the ball of given radius around the given vertex.
        TODO: Do we really need/want this function?
        '''
        pass

    def _get_k(self, vert_ids, k):
        if len(vert_ids) >= k:
            return vert_ids[:k]
        else:
            padding = [0] * len(vert_ids)
            return vert_ids + padding

    def kneighbors(self, vert_id, k, sortfunc=None, labeled=True):
        '''
        Returns the k vertices closest to the given vertex, sorted by the given sortfunc.
        If labeled is true, it returns their labels instead of their vertex ids.
        '''
        if vert_id == None or vert_id == 0:
            return [0] * k

        # Sort b distance to the origin vert_id, and use label as a tie breaker
        if sortfunc == None:
            sortfunc = lambda x: (self._distances[x, vert_id], self._node_labels_dict[x])

        nbrs = set([vert_id])
        knbrs = []

        while len(knbrs) < k:
            knbrs.extend(nbrs)
            nbrs = reduce(lambda acc, x: acc.union(self.neighbors(x)), nbrs, set())
            # no more new vertices to add to knbrs. Stop to avoid inf loop
            if knbrs == nbrs:
                break

        knbrs.sort(key=sortfunc)
        knbrs = self._get_k(knbrs, k) # pads or truncates the list to length k
        if labeled:
            return self.labels(knbrs)
        else:
            return knbrs

    def knbrs_lst(self, vert_ids, k, sortfunc=None, labeled=True):
        output_shape = (len(vert_ids), k)
        output = np.zeros(output_shape)
        for index, v in enumerate(vert_ids):
            output[index] = self.kneighbors(v, k, sortfunc, labeled)
        return output

    def knbr_edges(self, vert_id, k, sortfunc=None, labeled=True):
        '''
        Finds the k neighbors closest to the given vertex, ordered by the given sortfunc.
        Returns the adjacency matrix of the induced subgraph of the k vertices.
        If labeled is true, the (i, j) entry of the output matrix is the label of (i,j) instead
        of 0/1.
        '''
        output = np.zeros((k, k))

        if vert_id == None:
            return output

        # knbrs is a list of k vertex ids
        knbrs = self.kneighbors(vert_id, k, sortfunc, labeled=False)
        for i in range(k):
            for j in range(k):
                v1 = knbrs[i]
                v2 = knbrs[j]
                if labeled:
                    output[i, j] = self._edge_labels_dict.get((v1, v2), 0)
                else:
                    output[i, j] = 1 if (v1, v2) in self._edge_labels_dict else 0

        return output

    # TODO: This is a gross copy paste. Figure out the pythonic way to do this
    def knbr_edges_lst(self, vert_ids, k, sortfunc, labeled=False):
        output_shape = (len(vert_ids), k*k)
        output = np.zeros(output_shape)
        for index, v in enumerate(vert_ids):
            output[index] = self.knbr_edges(v, k, sortfunc, labeled)
        return output

    def sample(self, width, stride=None, sortfunc=None):
        '''
        Returns {width} vertices, spaced apart by stride according to the given sortfunc.
        This function always returns a list of length w. If there are not enough vertices to
        accomodate w and the stride length, the list is padded with None values.
        '''
        # Sort by the vertex label. Use vertex degree as tiebreaker.
        if sortfunc == None:
            sortfunc = (lambda x: (self._node_labels_dict.get(x, None), len(self._adj_lst_dict.get(x, None))))
        sorted_verts = sorted(self._vertices, key=sortfunc)
        chosen_verts = [sorted_verts[stride*i] if stride*i < len(sorted_verts) else None for i in range(width)]

        return chosen_verts

    def gen_channels(self, tensor, labels):
        '''
        TODO: Since we need to know the full sset of vertex and edge labels, this should probably
        be a graph dataset function?
        Input is of size (w, k) or (w, k^2). Returns (w, k, num_channels) or (w, k^2, num_channels)
        '''
        new_shape = list(tensor.shape) + [len(labels)]
        channels = np.zeros(new_shape)

        for ind, l in enumerate(labels):
            channels[:, :, ind] = (tensor == l)
        return channels


    def debug_print(self):
        print 'adj matrix:'
        print self._adj_mat
        print 'edge labels:'
        print self._edge_labels_dict
        print 'vert labels:'
        print self._node_labels_dict
