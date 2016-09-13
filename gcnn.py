import tensorflow as tf
import numpy as np

# TODO: figure out how to do the pretty graphs
class GCNN(object):
    def __init__(self, batch_size=32, learning_rate=0.01, dropout=0.5, width=10, nbr_size=5,
                 out_channels_1=16, out_channels_2=8, vert_labels=None, edge_labels=None, **kwargs):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._dropout = dropout
        self._width = width
        self._nbr_size = nbr_size
        self._vert_labels = vert_labels
        self._edge_labels = edge_labels
        self._num_output = 2
        self._params = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'width': width,
            'nbr_size': nbr_size,
            'vert_labels': vert_labels,
            'edge_labels': edge_lables,
            'out_channels_1': out_channels_1,
            'out_channels_2': out_channels_2,
        }

        self.session = tf.session()
        self.session.run(tf.initialize_all_variables())
        self._opt_handle, self._loss_handle, self._train_handle = self._build_graph()

    def _build_graph(self):
        input_data = tf.nn.placeholder(tf.float32, shape=())
        tf.reshape(input_data, [self._batch_size, 1, self._width * self._nbr_size,
len(self._vert_labels)]) # only doing verts for now
        input_labels = tf.nn.placeholder(int, shape=[self._batch_size])
        step = tf.nn.placeholder(int)

        out_channels_1 = 16
        out_channels_2 = 8
        fc_layer_shape = ()
        with tf.name_scope('conv1'):
            filter_shape = (1, self._nbr_size, len(self._vert_labels), self._out_channels_1)
            conv_filter = tf.nn.Variable(tf.float32, shape=filter_shape)
            conv1 = tf.conv2d(input_data, conv_filter, [1, 1, self._nbr_size, 1],
                              padding='VALID')
            conv1 = tf.nn.relu(conv1)
            conv1 = tf.nn.dropout(conv1, self._dropout)
        with tf.name_scope('conv2'):
            filter_shape = (1, 1,  self._out_channels1, self._out_channels_2)
            conv_filter = tf.nn.Variable(tf.float32, shape=filter_shape)
            conv2 = tf.conv2d(input_data, conv_filter, [1, 1, 1, 1],
                              padding='VALID')
            conv2 = tf.nn.relu(conv2)
            conv2 = tf.nn.dropout(conv2, self._dropout)
        with tf.name_scope('output'):
            shape = conv2.get_shape().as_list()
            reshaped = tf.reshape(conv2, [shape[0], -1])
            weight = tf.nn.Variable(tf.truncated_normal(
                [8*self._width], self._num_output], stddev=0.1)
            )
            bias = tf.Variable(tf.nn.zeros(2))
            logits = tf.matmul(conv2, weight) + bias
        with tf.name-scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits, input_labels)
        with tf.name_scope('optimizer'):
            opt = tf.train.RMSPropOptimizer(self._learning_rate)
            opt.minimize(loss)
        return opt, loss, train_prediction

    def train(self):
        dataset = util.load_dataset()
        iteration = 0
        while True:
            batch_data, batch_labels = dataset.next_batch()
            feed_dict = { # what else do I need?
                input_data: batch_data,
                input_labels: batch_data,
                step: iteration,
            }
            parts_to_compute = [self._opt, self._loss, self._train_prediction]
            opt, loss, train_pred = self.session.run(parts_to_compute, feed_dict)
            if iteration % 1000 == 0:
                print("loss: %.2f" %loss)
                # TODO: print the validation error
