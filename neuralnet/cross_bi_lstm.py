import tensorflow as tf
import numpy as np
from neuralnet.ops import highway, last_relevant

class cross_bi_lstm(object):

    def __init__(self, 
        target_vec_dic, 
        sequence_length, 
        num_classes, 
        vocab_size, 
        embedding_size, 
        filter_sizes, 
        num_filters, 
        dropout_keep_prob, 
        trans_w,
        trans_b,
        sour_dic,
        tar_dic,
        wei_dic,
        l2_reg_lambda = 0.0, 
        label_smoothing = 0.0, 
        highway_flag = False):

        # Placeholders for input, output and dropout
        self.seq_len = tf.placeholder(tf.int32, [None], name = "seq_len")
        self.input_f_en = tf.placeholder(tf.float32, name = "input_f_en")
        self.input_f_cn = tf.placeholder(tf.float32, name = "input_f_cn")
        self.input_trans_en = tf.placeholder(tf.int32, [None], name = "input_trans_en")
        self.input_trans_cn = tf.placeholder(tf.int32, [None], name = "input_trans_cn")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        self.cn_weight = tf.placeholder(tf.float32, name = "cn_weight")
        self.en_weight = tf.placeholder(tf.float32, name = "en_weight")
        self.trans_weight = tf.placeholder(tf.float32, name = "trans_weight")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        self.label_smoothing = label_smoothing
        self.rnn_cell = "LSTMBlock"
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

 
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        #embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"), tf.variable_scope("CNN") as scope:
            self.embedded_W = tf.Variable(
            #    tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0),
            #self.embedded_W = tf.constant(
                target_vec_dic,
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedded_W, self.input_x)
            #self.weight_W = tf.Variable(
            self.weight_W = tf.constant(
                wei_dic,
                name="W2")

            #self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob)
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
        
        # Transfer layer
        with tf.device('/cpu:0'), tf.name_scope("transfer"):
            ident_w = tf.constant(np.identity(embedding_size, dtype=np.float32), name = "ident_w")
            ident_b = tf.constant(np.zeros(embedding_size, dtype=np.float32), name = "ident_b")

            self.trans_w = tf.Variable(trans_w, name = "trans_w", dtype=np.float32)
            self.trans_b = tf.Variable(trans_b, name = "trans_b", dtype=np.float32)

            self.final_w = tf.add(tf.mul(ident_w, self.input_f_cn), tf.mul(self.trans_w, self.input_f_en), name = "final_w")
            self.final_b = tf.add(tf.mul(ident_b, self.input_f_cn), tf.mul(self.trans_b, self.input_f_en), name = "final_b")

            self.embedded_chars = tf.reshape(self.embedded_chars, [-1, embedding_size])
            #self.transfer_chars = tf.nn.relu(tf.add(tf.matmul(self.embedded_chars, self.final_w), self.final_b))
            self.transfer_chars = tf.add(tf.matmul(self.embedded_chars, self.final_w), self.final_b)
            self.transfer_chars = tf.reshape(self.transfer_chars, [-1, sequence_length, embedding_size])
            self.embedded_chars = self.transfer_chars

            self.embedded_chars = tf.transpose(self.embedded_chars, [1, 0, 2])
        
        # Lstm layer
        with tf.name_scope("lstm_layer"):
            if self.rnn_cell == "LSTMBlock":
                print("using block lstm cell")
                with tf.variable_scope("f"):
                    lstm_cell_f = tf.contrib.rnn.LSTMBlockFusedCell(embedding_size, forget_bias=1)
                    output_f, _ = lstm_cell_f(self.embedded_chars, dtype=tf.float32, sequence_length = self.seq_len)
                    output_fw = last_relevant(tf.transpose(output_f, [1, 0, 2]), self.seq_len)
                with tf.variable_scope("b"):
                    lstm_cell_b = tf.contrib.rnn.LSTMBlockFusedCell(embedding_size, forget_bias=1)
                    output_b, _ = lstm_cell_b(tf.reverse(self.embedded_chars, [True, False, False]), dtype=tf.float32, sequence_length = self.seq_len)
                    output_bw = last_relevant(tf.transpose(output_b, [1, 0, 2]), self.seq_len)
                self.output_fb = tf.concat(1, [output_fw, output_bw])
            elif self.rnn_cell == "LSTM":
                lstm_cell_f = tf.nn.rnn_cell.LSTMCell(embedding_size, forget_bias=1)
                lstm_cell_b = tf.nn.rnn_cell.LSTMCell(embedding_size, forget_bias=1)
                lstm_cell_f = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_f, output_keep_prob = dropout_keep_prob)
                lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_b, output_keep_prob = dropout_keep_prob)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_f, lstm_cell_b, self.embedded_chars, self.seq_len, dtype=tf.float32)
                #outputs, _ = tf.nn.dynamic_rnn(lstm_cell_f, self.embedded_chars, self.seq_len, dtype=tf.float32)
                output_fw, output_bw = outputs
                output_fw = tf.transpose(output_fw, [1, 0, 2])[-1]
                output_bw = tf.transpose(output_bw, [1, 0, 2])[-1]
                self.output_fb = tf.concat(1, [output_fw, output_bw])

   

        # Transfer loss
        with tf.device('/cpu:0'), tf.name_scope("transfer_loss"):
            trans_en_weight = tf.nn.embedding_lookup(self.weight_W, self.input_trans_en)
            trans_cn_weight = tf.nn.embedding_lookup(self.weight_W, self.input_trans_cn)
            trans_en_embedded = tf.nn.embedding_lookup(self.embedded_W, self.input_trans_en)
            trans_cn_embedded = tf.nn.embedding_lookup(self.embedded_W, self.input_trans_cn)

            loss_none_weight = tf.nn.xw_plus_b(trans_en_embedded, self.trans_w, self.trans_b) - trans_cn_embedded
            self.l2_loss_none_weight = tf.reduce_mean(tf.mul(loss_none_weight, loss_none_weight), 1)
            self.final_weight = tf.div(trans_cn_weight, trans_en_weight)
            self.transfer_loss = tf.reduce_mean(tf.mul(self.l2_loss_none_weight, self.final_weight))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output_fb, self.dropout_keep_prob)
            #self.h_drop = self.output_fb
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size*2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.log_softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name="scores"))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.contrib.losses.softmax_cross_entropy(self.scores, self.input_y)
            selflosses = tf.contrib.losses.softmax_cross_entropy(tf.cast(self.input_y, tf.float32),
                        tf.cast(self.input_y, tf.float32), label_smoothing = self.label_smoothing)
            self.kl = tf.reduce_mean(losses - selflosses)
            self.all_weight = self.input_f_en * self.en_weight + self.input_f_cn * self.cn_weight
            self.loss = tf.reduce_mean(losses) * self.all_weight + l2_reg_lambda * l2_loss + self.transfer_loss * self.trans_weight


        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
