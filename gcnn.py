import tensorflow as tf
import util
import numpy as np

# TODO: figure out how to do the pretty graphs
def variable(shape=None, stddev=0.1, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def accuracy(pred_labels, true_labels):
    return np.mean(pred_labels == true_labels)

class GCNN(object):
    def __init__(self, train_dataset=None, val=None, test=None, batch_size=32, learning_rate=0.1,
                 dropout=0.5, width=10, nbr_size=5, out_channels_1=16, out_channels_2=8,
                 num_vert_labels=None, num_edge_labels=None, **kwargs):
        self._train_dataset = train_dataset # dataset object
        self._val = val # dict mapping data/labels to the np arrays
        self._test = test # ditto ^

        # model params
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._dropout = dropout
        self._width = width
        self._nbr_size = nbr_size
        self._num_vert_labels = num_vert_labels
        self._num_edge_labels = num_edge_labels
        self._num_output = 2
        self._out_channels_1 = out_channels_1
        self._out_channels_2 = out_channels_2

        self.session = tf.Session()
        self._build_graph()
        self.session.run(tf.initialize_all_variables())

    def model(self, input_data):
        out_channels_1 = 8
        out_channels_2 = 4
        fc_layer_shape = [4 * self._width, 2]

        with tf.name_scope('conv1'):
            filter_shape = (1, self._nbr_size, self._num_vert_labels, self._out_channels_1)
            conv_filter = variable(shape=filter_shape, stddev=0.1, name='filter_c1')
            conv1 = tf.nn.conv2d(input_data, conv_filter, [1, 1, self._nbr_size, 1],
                                 padding='VALID', name='conv_c1')
            conv1 = tf.nn.relu(conv1, name='relu_c1')
            conv1 = tf.nn.dropout(conv1, self._dropout, name='dropout_c1')

        with tf.name_scope('conv2'):
            filter_shape = (1, 1,  self._out_channels_1, self._out_channels_2)
            conv_filter = variable(shape=filter_shape, stddev=0.1, name='filter_c2')
            conv2 = tf.nn.conv2d(conv1, conv_filter, [1, 1, 1, 1],
                                 padding='VALID', name='conv_c2')
            conv2 = tf.nn.relu(conv2, name='relu_c2')
            conv2 = tf.nn.dropout(conv2, self._dropout, name='dropped_c2')

        with tf.name_scope('fc_layer'):
            shape = conv2.get_shape().as_list()
            reshaped = tf.reshape(conv2, [shape[0], -1])
            weight = variable([8*self._width, self._num_output], stddev=0.1, name='fc_weight')
            bias = tf.Variable(tf.zeros(2), name='fc_bias')
            logits = tf.nn.bias_add(tf.matmul(reshaped, weight), bias, name='logits')

        return logits

    def _build_graph(self):
        self._input_data = tf.placeholder(tf.float32, shape=(self._batch_size, self._width,
                                          self._nbr_size, self._num_vert_labels), name='input')
        tf.reshape(self._input_data, [self._batch_size, 1, self._width * self._nbr_size,
                   self._num_vert_labels]) # only doing verts for now
        self._input_labels = tf.placeholder(tf.float32, shape=[self._batch_size, 2], name='labels')
        self._step = tf.placeholder(np.int32, name='iter_step')
        self._val_dataset = tf.constant(self._val['data'], dtype=tf.float32)
        self._test_dataset = tf.constant(self._test['data'], dtype=tf.float32)

        train_logits = self.model(self._input_data)
        self._train_prediction = tf.nn.softmax(train_logits, name='softmax_train')
        self._val_prediction = tf.nn.softmax(self.model(self._val_dataset), name='softmax_val')
        self._test_prediction = tf.nn.softmax(self.model(self._test_dataset), name='softmax_test')

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, self._input_labels), name='loss')
            #self._loss = accuracy(tf.nn.softmax(train_logits), self._input_labels)
        with tf.name_scope('optimizer'):
            self._opt = tf.train.RMSPropOptimizer(self._learning_rate).minimize(self._loss)
        #with tf.name_scope('prediction'):
        #   self._train_prediction = tf.argmax(tf.nn.softmax(train_logits), 1)


    def train(self, max_epochs=100, max_iters=20000):
        with self.session as sess:
            iteration = 0
            while iteration < max_iters:
                batch_data, batch_labels = self._train_dataset.next_batch()
                feed_dict = { # what else do I need?
                    self._input_data: batch_data,
                    self._input_labels: batch_labels,
                    self._step: iteration,
                }
                parts_to_compute = [self._opt, self._loss, self._train_prediction]
                opt, loss, train_pred = sess.run(parts_to_compute, feed_dict)
                if iteration % 100 == 0:
                    print("Iteration %d" % iteration)
                    valid_acc = util.accuracy(self._val_prediction.eval(), self._val['labels'])
                    train_acc = util.accuracy(train_pred, batch_labels)
                    print("Train acc: %.2f" % train_acc)
                    print("Valid acc: %.2f" % valid_acc)
                    print("=" * 80)
                iteration += 1
            print("Done with iterations")
            test_acc = util.accuracy(self._test_prediction.eval(), self._test['labels'])
            print("Test acc: %.2f" % test_acc)

