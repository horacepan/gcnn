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
    lname = 'l' + name.lower() if lname == None else lname
    labels = mat[lname]
    data = mat[name]

    labels = labels.reshape(len(labels))
    graphs = []
    for i in range(len(labels)):
        adj_mat = data[0][i][ADJ_MAT]
        node_labels = _clean_node_labels(data[0][i][NODE_LABELS])
        edge_labels = _clean_edge_labels(data[0][i][EDGE_LABELS])
        adj_lst = _clean_adj_lst(data[0][i][ADJ_LST])
        graphs.append(Graph(adj_mat, node_labels, edge_labels, adj_lst, labels))

    return graphs

def sorted_verts(graph):
    verts = range(1, graph.size()+1)
    verts.sort(key=lambda x: graph.label(x))
    return verts

def prep_graphs(graphs, width, stride, field_size):
    '''
        return a tensor of size (num_graphs
    '''

if __name__ == '__main__':
    dataset = sys.argv[1]
    fname = 'data/%s.mat' % dataset
    lname = 'l' + dataset.lower() if dataset != 'ENZYMES_SYM' else 'lenzymes'
    dataset = 'ENZYMES' if dataset == 'ENZYMES_SYM' else dataset
    mat = load_mat(fname, dataset, lname)
    pdb.set_trace()
