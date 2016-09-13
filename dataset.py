import numpy as np
import util
import pdb

class Dataset(object):
    '''
    Simple class that deals with spitting out batch sizes for training.
    The index gets reset(to 0) and the data/labels are shuffled once the index
    gets too high.
    '''
    def __init__(self, data, labels, batch_size):
        self._data = data
        self._labels = labels
        self._batch_size = batch_size
        self._index = 0     # denote where the current batch starts

    def next_batch(self):
        # shuffle and reset index if necessary
        if self._index + self._batch_size > self._size:
            self._index = 0
            permutation = np.random.permutation[self._size]
            self._data = self._data[permutation]
            self._graph_labels = self._graph_labels[permutation]

        start = self._index
        end = start + batch_size
        self._index = end
        return self._data[start:end], self._graph_labels[start:end]
