from gcnn import GCNN
import util
import ioutil
import numpy as np
import graph
from dataset import *
import pdb

def test_dataset(name):
    fname = 'data/%s.mat' %name
    batch_size = 32
    params = {
       'nbr_size': 5,
       'stride': 1,
       'channel_type': util.NODE,
    }
    normalized_data, labels = ioutil.load_dataset('data/MUTAG.mat', 'MUTAG', params)
    train, val, test = ioutil.train_val_test_split(normalized_data, labels, [3,1,1])
    train_dataset = Dataset(train['data'], train['labels'], batch_size)
    print("Done loading dataset %s" %name)

    gcnn_params = {
        'train_dataset': train_dataset,
        'val': val,
        'test': test,
        'batch_size': 32,
        'learning_rate': 0.005,
        'drop_out': 0.5,
        'width': 17, # fix
        'nbr_size': 5,
        'num_vert_labels': 7,
    }
    gcnn = GCNN(**gcnn_params)
    gcnn.train()

if __name__ == '__main__':
    #test_dataset('MUTAG')
    test_dataset('PTC')
