import scipy.io as sio
from graph import Graph
import pdb
import sys

ADJ_MAT = 0
NODE_LABELS = 1
EDGE_LABELS = 2
ADJ_LST = 3

def _clean_node_labels(node_labels):
    lbls = node_labels[0][0][0].flatten()
    cleaned = {i+1: val for (i, val) in enumerate(lbls)}
    return cleaned

def _clean_edge_labels(edge_lbls):
    lbls = edge_lbls[0][0][0]
    edge_lbls = {}

    for i, j, k in lbls:
        edge_lbls[(i, j)] = k

    return edge_lbls

def _clean_adj_lst(adj_lst):
    size = len(adj_lst)
    adj_lsts = {}

    for i in range(size):
        adj= list(adj_lst[i][0][0])
        adj_lsts[i+1] = adj

    return adj_lsts

def load_mat(fname, name, lname=None):
    mat = sio.loadmat(fname)
    lname = 'l' + name.lower()
    labels = mat[lname]
    data = mat[name]

    labels = labels.reshape(len(labels))
    graphs = []
    for i in range(len(labels)):
        adj_mat = data[0][i][ADJ_MAT]
        node_labels = _clean_node_labels(data[0][i][NODE_LABELS])
        edge_labels = _clean_edge_labels(data[0][i][EDGE_LABELS])
        adj_lst = _clean_adj_lst(data[0][i][ADJ_LST])
        graphs.append(Graph(adj_mat, node_labels, edge_labels, adj_lst))

    return graphs, labels

def load_dataset(fname, name, k):
    graphs = load_mat(fname, name)
    dataset = GraphDataset(graphs)
    return graphs

def floyd_warshall(adj_mat):
    '''
    Return a matrix whose (i, j) entry is the distance from vertex i to vertex j
    '''
    pass

def gen_adj_list(mat):
    '''
    adj_mat is a nxn numpy array
    TODO: do we want a list of lists as an output or a dict from vertice to list of neighbors
    '''
    adj_lst = []
    adj_lst_dict = {}

    for row in range(len(mat)):
        adj_lst.append(list(np.nonzero(mat[row, :])))
        adj_lst_dict[row+1] = list(np.nonzero(mat[row, :])) # vertices will be 1:n, not 0-indexed

    return adj_lst

if __name__ == '__main__':
    dataset = sys.argv[1]
    fname = 'data/%s.mat' % dataset
    lname = 'l' + dataset.lower()
    mat = load_mat(fname, dataset, lname)
