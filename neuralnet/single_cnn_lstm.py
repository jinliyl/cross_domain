import tensorflow as tf
import numpy as np
from neuralnet.ops import highway, last_relevant

class single_cnn_lstm(object):

    def __init__(self, target_vec_dic, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda = 0.0, label_smoothing = 0.0, highway_flag = False, dropout_keep_prob = 0.5):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "input_y")
        self.seq_len = tf.placeholder(tf.int32, [None], name = "seq_len")
        self.batch_size = tf.placeholder(tf.int32, name = "batch_size")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        self.label_smoothing = label_smoothing
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"), tf.variable_scope("CNN") as scope:
            self.embedded_W = tf.constant(
                target_vec_dic,
                #tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedded_W, self.input_x)
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob)
            if highway_flag:
                split_chars = tf.split(1, sequence_length, self.embedded_chars)
                split_chars_highway = []
                for idx in range(sequence_length):
                    if idx != 0:
                        scope.reuse_variables()
                    tmp = highway(tf.reshape(split_chars[idx], [-1, embedding_size]), embedding_size, layer_size = 1)
                    split_chars_highway.append(tf.expand_dims(tmp, 0))
                split_chars_highway = tf.concat(0, split_chars_highway)
                split_chars_highway = tf.transpose(split_chars_highway, [1, 0, 2])
                self.embedded_chars_expanded = tf.expand_dims(split_chars_highway, -1)
            else:
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
               
        
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        self.embedded_chars = tf.transpose(self.embedded_chars, [1, 0, 2])
        with tf.variable_scope("f"):
            lstm_cell_f = tf.contrib.rnn.LSTMBlockFusedCell(embedding_size, forget_bias=1)
            output_f, output_fs = lstm_cell_f(self.embedded_chars, dtype=tf.float32, sequence_length = self.seq_len)
            output_fw = last_relevant(tf.transpose(output_f, [1, 0, 2]), self.seq_len)
        with tf.variable_scope("b"):
            lstm_cell_b = tf.contrib.rnn.LSTMBlockFusedCell(embedding_size, forget_bias=1)
            output_b, output_bs = lstm_cell_b(tf.reverse(self.embedded_chars, [True, False, False]), dtype=tf.float32, sequence_length = self.seq_len)
            output_bw = last_relevant(tf.transpose(output_b, [1, 0, 2]), self.seq_len)
        self.output_fb = tf.concat(1, [self.h_pool_flat, output_fw, output_bw])
        #self.output_fb = tf.concat(1, [output_fw, output_bw])
        #self.output_fb = self.h_pool_flat

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output_fb, self.dropout_keep_prob)
            #self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total + embedding_size * 2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            #W2 = tf.Variable(np.zeros([embedding_size * 2, num_classes], dtype=np.float32))
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #b2 = tf.constant(0.0, shape=[num_classes])
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            #self.scores = tf.nn.log_softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name="scores"))
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b)# + tf.nn.xw_plus_b(self.h_drop2, W2, b2)*0.0
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.kl = self.loss 
            #losses = tf.contrib.losses.softmax_cross_entropy(self.scores, self.input_y, label_smoothing = self.label_smoothing)
            #selflosses = tf.contrib.losses.softmax_cross_entropy(tf.cast(self.input_y, tf.float32),
            #            tf.cast(self.input_y, tf.float32), label_smoothing = self.label_smoothing)
            #self.kl = tf.reduce_mean(losses - selflosses)
            #self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
