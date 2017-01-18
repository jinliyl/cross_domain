import tensorflow as tf
import numpy as np
from neuralnet.ops import highway, last_relevant

class single_bi_lstm(object):

    def __init__(self, target_vec_dic, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda = 0.0, label_smoothing = 0.0, highway_flag = False):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "input_y")
        self.seq_len = tf.placeholder(tf.int32, [None], name = "seq_len")
        self.batch_size = tf.placeholder(tf.int32, name = "batch_size")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        self.label_smoothing = label_smoothing
        self.rnn_cell = "LSTMBlock"
        wx_b = False
        latentsize = 1024


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        #embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"), tf.variable_scope("CNN") as scope:
            self.embedded_W = tf.Variable(
                target_vec_dic,
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedded_W, self.input_x)
            #split_chars = tf.split(1, sequence_length, self.embedded_chars)
            #res = []
            #for idx in range(sequence_length):
            #    #print(tf.shape(split_chars[idx]))
            #    tmp = tf.nn.dropout(tf.reshape(split_chars[idx], [-1, embedding_size]), self.dropout_keep_prob)
            #    res.append(tf.expand_dims(tmp, 0))
            #res = tf.concat(0, res)
            #self.embedded_chars = tf.transpose(res, [1, 0, 2])
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
                self.embedded_chars = split_chars_highway
            else:
                pass
            if wx_b :
                self.embedded_chars_reshape = tf.reshape(self.embedded_chars, [-1, embedding_size])
                W = tf.Variable(tf.truncated_normal([embedding_size, embedding_size], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[embedding_size]), name="b")

                #w = tf.Variable(tf.random_uniform([embedding_size, embedding_size], dtype=tf.float32))
                #b = tf.Variable(tf.random_uniform([embedding_size], dtype=tf.float32))
                self.embedded_chars = tf.nn.relu(tf.nn.xw_plus_b(self.embedded_chars_reshape, W, b))
                self.embedded_chars = tf.reshape(self.embedded_chars, [-1, sequence_length, embedding_size])
            self.embedded_chars = tf.transpose(self.embedded_chars, [1, 0, 2])
       
        # Lstm layer
        with tf.name_scope("lstm_layer"):
            if self.rnn_cell == "LSTMBlock":
                print("using block lstm cell")
                with tf.variable_scope("f"):
                    lstm_cell_f = tf.contrib.rnn.LSTMBlockFusedCell(embedding_size, forget_bias=1)
                    output_f, output_fs = lstm_cell_f(self.embedded_chars, dtype=tf.float32, sequence_length = self.seq_len)
                    output_fw = last_relevant(tf.transpose(output_f, [1, 0, 2]), self.seq_len)
                    #output_fs = last_relevant(tf.transpose(output_fs, [1, 0, 2]), self.seq_len)
                with tf.variable_scope("b"):
                    lstm_cell_b = tf.contrib.rnn.LSTMBlockFusedCell(embedding_size, forget_bias=1)
                    output_b, output_bs = lstm_cell_b(tf.reverse(self.embedded_chars, [True, False, False]), dtype=tf.float32, sequence_length = self.seq_len)
                    output_bw = last_relevant(tf.transpose(output_b, [1, 0, 2]), self.seq_len)
                    #output_bs = last_relevant(tf.transpose(output_bs, [1, 0, 2]), self.seq_len)
                self.output_fb = tf.concat(1, [output_fw, output_bw])
                #self.output_fb = output_fw
            elif self.rnn_cell == "LSTM":
                lstm_cell_f = tf.nn.rnn_cell.LSTMCell(embedding_size, forget_bias=1, use_peepholes=True)
                lstm_cell_b = tf.nn.rnn_cell.LSTMCell(embedding_size, forget_bias=1, use_peepholes=True)
                #lstm_cell_f = tf.nn.rnn_cell.GRUCell(embedding_size)
                #lstm_cell_b = tf.nn.rnn_cell.GRUCell(embedding_size)
                #lstm_cell_f = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=1)
                #lstm_cell_b = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=1)
                lstm_cell_f = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_f, output_keep_prob=self.dropout_keep_prob)
                lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_b, output_keep_prob=self.dropout_keep_prob)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_f, lstm_cell_b, self.embedded_chars, self.seq_len, dtype=tf.float32, time_major=True)
                #self.embedded_chars = tf.split(0, sequence_length, tf.reshape(self.embedded_chars, [-1, embedding_size]))
                #outputs, _, _ = tf.nn.bidirectional_rnn(lstm_cell_f, lstm_cell_b, self.embedded_chars, sequence_length=self.seq_len, dtype=tf.float32)
                output_fw, output_bw = outputs
                output_fw = last_relevant(tf.transpose(output_fw, [1, 0, 2]), self.seq_len)
                output_bw = last_relevant(tf.transpose(output_bw, [1, 0, 2]), self.seq_len)
                self.output_fb = tf.concat(1, [output_fw, output_bw])
                #outputs = tf.transpose(outputs, [1, 0, 2])
                #self.output_fb = last_relevant(outputs, self.seq_len)


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output_fb, self.dropout_keep_prob)
            #self.h_drop = self.output_fb

            
        #with tf.name_scope("latent"):
        #    W = tf.Variable(tf.random_uniform([embedding_size, latentsize], dtype=tf.float32))
        #    b = tf.Variable(tf.random_uniform([latentsize], dtype=tf.float32))
        #    self.h_drop = tf.nn.tanh(tf.nn.xw_plus_b(self.h_drop, W, b))
 
        #with tf.name_scope("dropout"):
        #    self.h_drop = tf.nn.dropout(self.h_drop, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size * 2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            #W = tf.Variable(tf.random_uniform([embedding_size * 2, num_classes], dtype=tf.float32))
            #b = tf.Variable(tf.random_uniform([num_classes], dtype=tf.float32))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.kl = self.loss

            #losses = tf.contrib.losses.softmax_cross_entropy(self.scores, self.input_y)
            #selflosses = tf.contrib.losses.softmax_cross_entropy(tf.cast(self.input_y, tf.float32),
            #            tf.cast(self.input_y, tf.float32))
            #self.kl = tf.reduce_mean(losses - selflosses)
            #self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
