import tensorflow as tf
import numpy as np

class cross_cnn():

   def __init__(self, 
        target_vec_dic, 
        sequence_length, 
        num_classes, 
        vocab_size, 
        embedding_size, 
        filter_sizes, 
        num_filters, 
        trans_w,
        trans_b,
        sour_dic,
        tar_dic,
        wei_dic,
        l2_reg_lambda=0.0, 
        label_smoothing = 0.00):

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
        self.label_smoothing = label_smoothing
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)


        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #self.embedded_W = tf.Variable(
            self.embedded_W = tf.constant(
                target_vec_dic, 
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedded_W, self.input_x)
            
            #self.weight_W = tf.Variable(
            self.weight_W = tf.constant(
                wei_dic, 
                name="W2")      

        # Transfer layer
        with tf.device('/cpu:0'), tf.name_scope("transfer"):
            ident_w = tf.constant(np.identity(embedding_size, dtype=np.float32), name = "ident_w")
            ident_b = tf.constant(np.zeros(embedding_size, dtype=np.float32), name = "ident_b")
            
            self.trans_w = tf.Variable(trans_w, name = "trans_w", dtype=np.float32)
            self.trans_b = tf.Variable(trans_b, name = "trans_b", dtype=np.float32)
            
            self.final_w = tf.add(tf.mul(ident_w, self.input_f_cn), tf.mul(self.trans_w, self.input_f_en), name = "final_w")
            self.final_b = tf.add(tf.mul(ident_b, self.input_f_cn), tf.mul(self.trans_b, self.input_f_en), name = "final_b")
            
            self.embedded_chars = tf.reshape(self.embedded_chars, [-1, embedding_size])
            self.transfer_chars = tf.nn.xw_plus_b(self.embedded_chars, self.final_w, self.final_b)
            self.transfer_chars = tf.reshape(self.transfer_chars, [-1, sequence_length, embedding_size])
            self.embedded_chars_expanded = tf.expand_dims(self.transfer_chars, -1)

            
               
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

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
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
