import scipy.io as sio
import numpy as np
from graph import Graph
import pdb
import sys

NODE = 'node'
EDGE = 'edge'
def floyd_warshall(adj_mat):
    '''
    Return a matrix whose (i, j) entry is the distance from vertex i to vertex j
    TODO: to be used for the sort function
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

def make_one_hot(labels, max_label=None):
    '''
    Takes in a nx1 vector of discrete labels(or a list).
    Returns nxc vector, where c is the total number of discrete labels.
    '''
    # TODO: some datasets do +/- 1 as labels... so this will have some index ob errors
    if max_label == None:
        max_label = max(labels)
    one_hot = np.zeros((len(labels), max_label))
    for i in range(len(labels)):
        one_hot[i, labels[i]-1] = 1 # assume that labels are 1 to max_labels so we -1

    return one_hot

def get_all_labels(graphs, label_type='node'):
    labels = set()
    fname = 'node_labels' if label_type == 'node' else 'edge_labels'
    for g in graphs:
        g_labels = getattr(g, fname)()
        labels = labels.union(g_labels)
    return labels

def gen_channels(graphs=None, width=5, nbr_size=5, stride=1, channel_type=NODE):
    '''
    Returns a (num_graphs, width, nbr_size, num_labels) tensor if the channel type is NODE
    Returns a (num_graphs, width, nbr_size^2, num_labels) tensor if the channel type is EDGE 
    '''
    output = np.zeros(output_shape)
    all_labels = get_all_node_labels(graphs, channel_type)
    func_params = {
        'vert_ids': None,           # gets filled in in each loop iteration
        'k': nbr_size,
        'sortfunc': (lambda x: x),  # TODO: use a real sort func
    }

    if channel_type == NODE:
        output_shape = (len(graphs), width,  nbr_size, len(all_labels))
        fname = 'knbrs_lst'
    else:
        output_shape = (len(graphs), width,  nbr_size * nbr_size, len(all_labels))
        fname = 'knbr_edges'

    for index, graph in enumerate(graphs):
        sampled_verts = graph.sample(width, stride)
        func_params['vert_ids'] = sampled_verts
        layer = getattr(graph, fname)(**func_params)
        output[index] = single_channel(layer, all_labels)

    return output

def single_channel(layer, all_labels):
    '''
    Given an nxm matrix of labels, returns a n x m x l matrix where
    the n x m slice at index i = the boolean mask of the input matrix for label l_i
    '''
    new_shape = list(layer.shape) + [len(all_labels)]
    channels = np.zeros(new_shape)
    for ind, l in enumerate(labels):
        channels[:, :, ind] = (layer == l)
    return channels

def avg_nodes(graphs):
    return np.mean([g.size for g in graphs])

if __name__ == '__main__':
    dataset = sys.argv[1]
    fname = 'data/%s.mat' % dataset
    lname = 'l' + dataset.lower()
    mat = load_mat(fname, dataset, lname)
