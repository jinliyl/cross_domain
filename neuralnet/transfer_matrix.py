import tensorflow as tf
import numpy as np


class transfer_matrix(object):

    def __init__(self, target_vec_dic, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, label_smoothing = 0.00):
        
        # placeholder for input
        trans_en = tf.placeholder(tf.int32, [None], name = "trans_en")
        trans_cn = tf.placeholder(tf.int32, [None], name = "trans_cn")

        trans_en_weight = tf.placeholder(tf.float32, [None], name = "trans_en_weight")
        trans_cn_weight = tf.placeholder(tf.float32, [None], name = "trans_cn_weight")

        # embedding seq
        embedded_W = tf.constant(target_vec_dic, name="W")
        trans_en_embedded = tf.nn.embedding_lookup(embedded_W, trans_en)
        trans_cn_embedded = tf.nn.embedding_lookup(embedded_W, trans_cn)

        trans_w = tf.Variable(tf.random_uniform([embedding_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), name = "trans_w", dtype=np.float32)
        trans_b = tf.Variable(tf.random_uniform([embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), name = "trans_b", dtype=np.float32)

        # loss
        loss_none_weight = tf.nn.xw_plus_b(trans_en_embedded, trans_w, trans_b) - trans_cn_embedded
        l2_loss_none_weight = tf.reduce_mean(tf.mul(loss_none_weight, loss_none_weight), 1)
        vec_weight = tf.div(trans_cn_weight, trans_en_weight)
        loss = tf.reduce_mean(tf.mul(l2_loss_none_weight, vec_weight))


        
        
