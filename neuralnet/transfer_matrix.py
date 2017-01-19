import tensorflow as tf
import numpy as np


class transfer_matrix(object):

    def __init__(self, vec_dic, wei_dic, embedding_size):
        
        # placeholder for input
        self.trans_en = tf.placeholder(tf.int32, [None], name = "trans_en")
        self.trans_cn = tf.placeholder(tf.int32, [None], name = "trans_cn")

        # embedding seq
        embedded_W = tf.constant(vec_dic, name="W")
        weight_W = tf.constant(wei_dic, name="W2")
        trans_en_embedded = tf.nn.embedding_lookup(embedded_W, self.trans_en)
        trans_cn_embedded = tf.nn.embedding_lookup(embedded_W, self.trans_cn)
        trans_en_weight = tf.nn.embedding_lookup(weight_W, self.trans_en)
        trans_cn_weight = tf.nn.embedding_lookup(weight_W, self.trans_cn)

        self.trans_w = tf.Variable(tf.random_uniform([embedding_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), name = "trans_w", dtype=np.float32)
        self.trans_b = tf.Variable(tf.random_uniform([embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), name = "trans_b", dtype=np.float32)

        # loss
        loss_none_weight = tf.nn.xw_plus_b(trans_en_embedded, self.trans_w, self.trans_b) - trans_cn_embedded
        l2_loss_none_weight = tf.reduce_mean(tf.mul(loss_none_weight, loss_none_weight), 1)
        final_weight = tf.div(trans_cn_weight, trans_en_weight)
        self.loss = tf.reduce_mean(tf.mul(l2_loss_none_weight, final_weight))


        
        
