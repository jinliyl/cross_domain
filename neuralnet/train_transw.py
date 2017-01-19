import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from neuralnet.transfer_matrix import transfer_matrix

class train_transw(object):

    def __init__(self, 
            target_dic_path, 
            source_dic_path, 
            transfer_path, 
            word_vec_target, 
            word_vec_source, 
            tf_df_target,
            tf_df_source,
            weight_kind = "1",
            embedding_size = 128,
            trainable = True, 
            max_iter = 20000):

        self.target_dic_path = target_dic_path
        self.source_dic_path = source_dic_path
        self.transfer_path = transfer_path
        self.word_vec_target = word_vec_target
        self.word_vec_source = word_vec_source
        self.tf_df_target = tf_df_target
        self.tf_df_source = tf_df_source

        self.weight_kind = weight_kind
        self.embedding_size = embedding_size
        self.trainable = trainable
        self.max_iter = max_iter

        self.sour_dic = []
        self.tar_dic = []
        self.trans_w = []
        self.trans_b = []

    def __call__(self):
        if not self.trainable:
            self.load__wb()
            return self.trans_w, self.trans_b

        print("start training w_b...")
        target_dic_len = self.load_dic_len(self.target_dic_path)
        print("target dic " + self.target_dic_path + " len: " + str(target_dic_len))
        source_dic_len = self.load_dic_len(self.source_dic_path)
        print("source dic " + self.source_dic_path + " len: " + str(source_dic_len))
        
        self.load_transfer(self.transfer_path, source_dic_len)
        
        self.vec_dic = {}
        self.vec_dic[0] = [0 for x in range(self.embedding_size)]
        self.load_word_vec(self.vec_dic, self.word_vec_source, 1)
        self.load_word_vec(self.vec_dic, self.word_vec_target, 1 + source_dic_len)
        self.vec_dic = [x for _, x in list(sorted(self.vec_dic.items(), key = lambda x:x[0]))]
        print(len(self.vec_dic))

        self.wei_dic = {}
        self.wei_dic[0] = 0.0
        self.load_word_tf_df(self.wei_dic, self.tf_df_source, 1)
        self.load_word_tf_df(self.wei_dic, self.tf_df_target, 1 + source_dic_len)
        self.wei_dic = [x for _, x in list(sorted(self.wei_dic.items(), key = lambda x:x[0]))]
        print(len(self.wei_dic))
        

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                ## new class
                trans_model = transfer_matrix(
                    vec_dic = self.vec_dic, 
                    wei_dic = self.wei_dic, 
                    embedding_size = self.embedding_size)
                
                ## optimize
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
                grads_and_vars = optimizer.compute_gradients(trans_model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                sess.run(tf.global_variables_initializer())

                ## feed dict
                for i in range(self.max_iter):
                    #current_step = tf.train.global_step(sess, global_step)
      
                    feed_dict = {
                        trans_model.trans_en: self.sour_dic,
                        trans_model.trans_cn: self.tar_dic
                    }
                    _, step, loss, self.trans_w, self.trans_b = sess.run(
                        [train_op, global_step, trans_model.loss, trans_model.trans_w, trans_model.trans_b],
                        feed_dict)
                    time_str = datetime.now().isoformat()
                    print("train {}: step {}, loss {:g}".format(time_str, step, loss))
                self.save_wb()
                return self.trans_w, self.trans_b

    def load__wb(self):
        with open("./temp_para/trans_w") as f:
            for line in f:
                ll = line.strip().split(" ")
                self.trans_w.append([float(x) for x in ll])
        with open("./temp_para/trans_b") as f:
            for line in f:
                ll = line.strip().split(" ")
                self.trans_b = [float(x) for x in ll]

    def save_wb(self):
        with open("./temp_para/trans_w", "w") as fw:
            for line in self.trans_w:
                print(" ".join([str(x) for x in line]), file=fw)
           
        with open("./temp_para/trans_b", "w") as fw:
            print(" ".join([str(x) for x in self.trans_b]), file=fw)

    def load_dic_len(self, path):
        dic_len = 0;
        with open(path) as f:
            for line in f:
                dic_len += 1
        return dic_len


    def load_transfer(self, path, add_len):
        with open(path) as f:
            for line in f:
                line = line.strip()
                ind_sour, ind_tar, _, _ = line.split("\t")
                self.sour_dic.append(int(ind_sour) + 1)
                self.tar_dic.append(int(ind_tar) + 1 + add_len)


    def load_word_vec(self, dic, vec_path, add_len):
        with open(vec_path) as f:
            for line in f:
                ll = line.strip().split(" ")
                if len(ll) < 129 or ll[0] == "</s>":
                    continue
                word = int(ll[0])
                embedding = [float(x) for x in ll[1:]]
                dic[word + add_len] = embedding


    def load_word_tf_df(self, dic, tf_df_path, add_len):
        with open(tf_df_path) as f:
            for line in f:
                ll = line.strip().split("\t")
                if len(ll) != 3:
                    continue
                ind, tf, df = ll
                ind = int(ind)
                tf = float(tf)
                df = float(df)     
                if self.weight_kind == "tf":
                    wei = tf
                elif self.weight_kind == "df":
                    wei = df
                elif self.weight_kind == "tfidf":
                    wei = tf/idf
                else:
                    wei = 1.0
                dic[ind + add_len] = wei






