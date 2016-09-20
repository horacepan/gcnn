import scipy.io as sio
import numpy as np
import pdb
import sys

NODE = 'node'
EDGE = 'edge'
def floyd_warshall(adj_mat):
    '''
    Return a matrix whose (i, j) entry is the distance from vertex i to vertex j
    TODO: vectorize
    '''
    num_verts = len(adj_mat)
    dist = np.zeros((num_verts, num_verts))
    xs, ys = np.nonzero(adj_mat)

    for i in range(len(xs)):
        dist[xs[i], ys[i]] = 1
        dist[ys[i], xs[i]] = 1

    for k in range(num_verts):
        for i in range(num_verts):
            for j in range(num_verts):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    dist[j, i] = dist[i, j]

    assert (dist == dist.T).all()
    return dist

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
    all_labels = get_all_labels(graphs, channel_type)
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

    output = np.zeros(output_shape)
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
    for ind, l in enumerate(all_labels):
        channels[:, :, ind] = (layer == l)
    return channels

def avg_nodes(graphs):
    return np.mean([g.size for g in graphs])

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))) / predictions.shape[0]

if __name__ == '__main__':
    dataset = sys.argv[1]
    fname = 'data/%s.mat' % dataset
    lname = 'l' + dataset.lower()
    mat = load_mat(fname, dataset, lname)
