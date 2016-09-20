from __future__ import division
from graph import Graph
import util
import numpy as np
import scipy.io as sio
import dataset

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

def make_one_hot(labels):
    '''
    Takes in a nx1 vector of discrete labels(or a list).
    Returns nxc vector, where c is the total number of discrete labels.
    '''
    unique_labels = list(set(labels))
    cleaned_labels = range(1, len(unique_labels)+1) # always convert labels to 1 to n
    one_hot = np.zeros((len(labels), len(cleaned_labels)))
    for i in range(len(cleaned_labels)):
        one_hot[i, labels[i]-1] = 1 # assume that labels are 1 to max_labels so we -1

    return one_hot

def train_val_test_split(data, labels, proportions):
    '''
    Proportions is a list of proportions for the train, val and test split
    Returns a 3 tuple of dictionaries for the train, val and test in that order.
    Each dict has data and labels as keys.
    '''
    if len(proportions) != 3:
        raise Exception("Must supply a proportion for the train, val and test split")
    if sum(proportions) != 1:
        total = sum(proportions)
        proportions = [(p / total) for p in proportions]

    size = len(data)
    train_size = int(proportions[0] * size)
    val_size   = int(proportions[1] * size)
    test_size  = int(proportions[2] * size)
    # give the leftovers to the test set
    if (train_size + val_size + test_size < size):
        test_size = size - (train_size + val_size + test_size)

    # TODO: do something to ensure class balance?
    permutation = np.random.permutation(size)
    data = data[permutation]
    labels = labels[permutation]

    train_data, train_labels = data[:train_size], labels[:train_size]
    val_data, val_labels = data[train_size:train_size + val_size], labels[train_size:train_size+val_size]
    test_data, test_labels = data[train_size+val_size:], labels[train_size+val_size:]

    train = {'data': train_data, 'labels': train_labels}
    val = {'data': val_data, 'labels': val_labels}
    test = {'data': test_data, 'labels': test_labels}
    return train, val, test

def load_mat(fname, name, lname=None, one_hot=True):
    '''
    Input: name of the file, name of the matlab variable for the graphs, name of the matlab
    variable for the labels.
    Returns a tuple of a list of graphs and a list of labels.
    '''
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

    # Turn the labels into one hot encoded vectors.
    if one_hot:
        labels = make_one_hot(labels)

    return graphs, labels

def load_dataset(fname, name, params):
    '''
    params must hold: nbr_size, stride, channel_type
    '''
    graphs, labels = load_mat(fname, name)
    avg_nodes = util.avg_nodes(graphs)
    params['graphs'] = graphs
    params['width'] = int(avg_nodes)
    normalized_data = util.gen_channels(**params)

    return normalized_data, labels

