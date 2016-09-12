import tensorflow as tf
import numpy as np

# TODO: figure out how to do the pretty graphs
class GCNN(object):
    def __init__(self):
        self._defaults = {
            'batch_size': 32,
            'learning_rate': 0.05,
            'dropout': 0.5,
            'width': 10,
            'nbr_size': 10,
            'num_vert_labels': 20,
            'num_edge_labels': 20,
        }
        self.session = tf.session()
        self._opt_handle, self._loss_handle, self._train_handle = self._build_graph
        self.session.run(tf.initialize_all_variables())
    def _build_graph(self):
        input_data = tf.nn.placeholder()
        input_label = tf.placeholder()
        step = tf.nn.placeholder(int)

        with tf.name_scope('conv1'):
            conv_filter = tf.nn.Variable(tf.float32, shape=())
            conv1 = tf.conv2d(conv_filter, input_data)
            conv1 = tf.nn.relu(conv)
            conv1 = tf.nn.dropout(conv1, self._defaults['dropout'])
        with tf.name_scope('conv1'):
            conv_filter = tf.nn.Variable(tf.float32, shape=())
            conv2 = tf.conv2d(conv_filter, conv)
            conv2 = tf.nn.relu(conv)
            conv2 = tf.nn.dropout(conv2, self._defaults['dropout'])
        with tf.name_scope('output'):
            weight = tf.nn.Variable(tf.float32, shape=())
            bias = tf.nn.Zeros(shape=())
            fc_layer = tf.matmul(weight, conv2) + bias
            softmax = tf.nn.softmax(fc_layer)
        with tf.name-scope('loss'):
            loss = tf.nn.loss.binary_cross_entropy(softmax, input_label)
        with tf.name_scope('optimizer'):
            # TODO: gradient clipping?
            opt = tf.train.AdamOptimizer(self._learning_rate)
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
