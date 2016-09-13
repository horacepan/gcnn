import util
import ioutil
import numpy as np
import graph
from dataset import *
import pdb

def test_mutag():
    fname = 'data/MUTAG.mat'
    name = 'MUTAG'
    batch_size = 32
    params = {
       'nbr_size': 5,
       'stride': 1,
       'channel_type': util.NODE,
    }
    normalized_data, labels = ioutil.load_dataset('data/MUTAG.mat', 'MUTAG', params)
    train, val, test = ioutil.train_val_test_split(normalized_data, labels, [3,1,1])
    train_dataset = Dataset(train['data'], train['labels'], batch_size)
    print train, val, test
    pdb.set_trace()

if __name__ == '__main__':
    test_mutag()
