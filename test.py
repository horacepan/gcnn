import util
import numpy as np
import graph
from dataset import GraphDataset
import pdb

def test_mutag():
    graphs, labels = util.load_mat('data/MUTAG.mat', 'MUTAG')
    params = {
       'width': 30,
       'nbr_size': 3,
       'stride': 1,
       'channel_type': 'vertices',
    }
    dataset = GraphDataset(graphs, labels, 32, params)

if __name__ == '__main__':
    test_mutag()
