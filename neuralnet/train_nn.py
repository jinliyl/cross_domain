import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from neuralnet.cross_cnn import cross_cnn
from neuralnet.single_cnn import single_cnn
from neuralnet.single_bi_lstm import single_bi_lstm
from neuralnet.cross_bi_lstm import cross_bi_lstm
from neuralnet.single_cnn_lstm import single_cnn_lstm
from neuralnet.train_transw import train_transw

class train_nn():

    def __init__(self, 
                emotion_list = [], 
                target_dic_path = "",
                source_dic_path = "", 
                target_path = "", 
                source_path = "", 
                transfer_path = "", 
                part = 2, 
                model = "cnn", 
                sequence_length = 150, 
                cross_lingual = True,
                embedding_dim = 128, 
                filter_sizes = [3, 4, 5], 
                num_filters = 128, 
                dropout_keep_prob = 0.5,
                l2_reg_lambda = 0.00, 
                batch_size = 64, 
                num_epochs = 500, 
                evaluate_every = 10, 
                checkpoint_every = 100,
                random_train = False,
                op_step = 1e-3, 
                word_vec_target = "",
                word_vec_source = "",
                tf_df_target = "", 
                tf_df_source = "",
                max_iter = 600):
        
        self.emotion_list = emotion_list
        self.target_dic_path = target_dic_path
        self.source_dic_path = source_dic_path
        self.target_path = target_path
        self.source_path = source_path
        self.transfer_path = transfer_path
        self.part = part
        self.model = model
        self.sequence_length = sequence_length
        self.cross_lingual = cross_lingual

        self.embedding_dim = embedding_dim
        self.filter_sizes =  filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
              
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every

        self.pre_step = 200
        self._seed = 10
        self.random_train = random_train
        self.op_step = op_step
        self.word_vec_target = word_vec_target
        self.word_vec_source = word_vec_source
        self.tf_df_target = tf_df_target
        self.tf_df_source = tf_df_source
        self.max_iter = max_iter


    def run(self):
        print("emotions: " + str(self.emotion_list))
        if self.cross_lingual:
            tt = train_transw(
                target_dic_path = self.target_dic_path,
                source_dic_path = self.source_dic_path,
                transfer_path = self.transfer_path,
                word_vec_target = self.word_vec_target,
                word_vec_source = self.word_vec_source,
                tf_df_target = self.tf_df_target,
                tf_df_source = self.tf_df_source,
                weight_kind = "1",
                embedding_size = 128,
                trainable = False,
                max_iter = 2000)
            tt()
            self.trans_w = tt.trans_w
            self.trans_b = tt.trans_b
            self.sour_dic = tt.sour_dic
            self.tar_dic = tt.tar_dic
            self.wei_dic = tt.wei_dic
 
            self.load_cross_data()
            return self.cross_training()
        else:
            self.load_single_data()
            return self.single_training()

    def load_data(self, path, add_len = 0):
        data_label = []
        data_feature = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                label, line = line.split("\t")
                if label not in self.emotion_list:
                    continue
                labelfeature = [0 for ii in range(len(self.emotion_list))]
                pos = self.emotion_list.index(label)
                labelfeature[pos] = 1
                data_label.append(labelfeature)
                ll = line.split(" ")
                data_feature.append([(int(x) + add_len + 1) for x in ll])
        return data_label, data_feature


    def load_dic_len(self, path):
        dic_len = 0;
        with open(path) as f:
            for line in f:
                dic_len += 1
        return dic_len

    def load_transfer(self, path, add_len):
        res = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                index_en, index_cn, _, _ = line.split("\t")
                res[int(index_en) + 1] = int(index_cn) + 1 + add_len
        return res

    def shuffle_data(self, label, feature, part = 1):
        assert(len(label) == len(feature))
        if self.random_train:
            random.seed(self._seed)
        shuffle_indices = np.random.permutation(np.arange(len(label)))
        feature_shuffled = [feature[i] for i in shuffle_indices]
        label_shuffled = [label[i] for i in shuffle_indices]
        if part == 1:
            return feature_shuffled, label_shuffled
        elif part > 1:
            train = int(len(label) * 1.0 / part)
            return feature_shuffled[:train], feature_shuffled[train:], label_shuffled[:train], label_shuffled[train:]
        else:
            train = int(len(label) * 1.0 * part)
            return feature_shuffled[:train], feature_shuffled[train:], label_shuffled[:train], label_shuffled[train:]
          
    def batch_iter(self, data, batch_size, shuffle = False):
        data_size = len(data)
        num_batches = int(len(data)/batch_size) + 1
        if shuffle:
            if self.random_train:
               np.random.seed(self._seed) 
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = [data[x] for x in shuffle_indices]
        else:
            shuffled_data = data

        data_len = [len(x[1]) for x in shuffled_data]
   
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index < end_index:
                yield shuffled_data[start_index:end_index]

    def gene_pad_seq(self, batches):
        seq_len = self.sequence_length
        batch_seq_len = []
        batch_pad = []
        for b in batches:
            if len(b) < seq_len:
                feature = b + [0 for i in range(seq_len - len(b))]
                batch_seq_len.append(len(b))
            else:
                feature = b[:seq_len]
                batch_seq_len.append(seq_len)
            batch_pad.append(feature)
        return batch_pad, batch_seq_len

    def load_word_vec(self, dic, vec_path, add_len):
        with open(vec_path) as f:
            for line in f:
                ll = line.strip().split(" ")
                if len(ll) < 129 or ll[0] == "</s>":
                    continue
                word = int(ll[0])
                embedding = [float(x) for x in ll[1:]]
                dic[word + add_len] = embedding

    def load_cross_data(self):
        print("Loading data...")
        target_dic_len = self.load_dic_len(self.target_dic_path)
        print("target dic " + self.target_dic_path + " len: " + str(target_dic_len))
        source_dic_len = self.load_dic_len(self.source_dic_path)
        print("source dic " + self.source_dic_path + " len: " + str(source_dic_len))
        self.vocab_size = target_dic_len + source_dic_len + 1
        
        self.target_label, self.target_feature = self.load_data(self.target_path, source_dic_len)
        print("target_path: " + self.target_path + " || len: " + str(len(self.target_label)))
        source_label, source_feature = self.load_data(self.source_path)
        print("source_path: " + self.source_path + " || len: " + str(len(source_label)))

        self.target_vec_dic = {}
        self.target_vec_dic[0] = [0 for x in range(self.embedding_dim)]
        self.load_word_vec(self.target_vec_dic, self.word_vec_source, 1)
        self.load_word_vec(self.target_vec_dic, self.word_vec_target, 1 + source_dic_len)
        self.target_vec_dic = [x for _, x in list(sorted(self.target_vec_dic.items(), key = lambda x:x[0]))]

        self.transform_dic = self.load_transfer(self.transfer_path, source_dic_len)
        
        target_train_feature, self.target_test_feature, target_train_label, self.target_test_label = \
            self.shuffle_data(self.target_label, self.target_feature, self.part)
        self.source_train_feature, source_train_label = self.shuffle_data(source_label, source_feature)
        
        print("Target Train/Test split: {:d}/{:d}".format(len(target_train_feature), len(self.target_test_feature)))
        print("Source: {:d}".format(len(self.source_train_feature)))

        self.all_batches = []
        for i in range(self.num_epochs):
            if self.part >= 1:
                part_epochs = 4#self.part
            else:
                part_epochs = 1
            batches = []
            for j in range(part_epochs):
                arr_1 = [1 for k in range(len(target_train_label))]
                arr_0 = [0 for k in range(len(target_train_label))]
                target_batches = self.batch_iter(list(zip(target_train_label, target_train_feature, arr_1, arr_0)), self.batch_size)
                batches += target_batches
            source_batches = self.batch_iter(list(zip(source_train_label, self.source_train_feature, arr_0, arr_1)), self.batch_size)
            batches += source_batches
            if self.random_train:
                random.seed(self._seed)
            random.shuffle(batches)
            self.all_batches += batches

    def cross_training(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)#, intra_op_parallelism_threads = 24)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                if self.model == "cnn":
                    cross_model = cross_cnn(
                        target_vec_dic = self.target_vec_dic,
                        sequence_length = self.sequence_length,
                        num_classes = len(self.emotion_list),
                        vocab_size = self.vocab_size,
                        embedding_size = self.embedding_dim,
                        filter_sizes = self.filter_sizes,
                        num_filters = self.num_filters,
                        l2_reg_lambda = self.l2_reg_lambda,
                        trans_w = self.trans_w,
                        trans_b = self.trans_b,
                        sour_dic = self.sour_dic,
                        tar_dic = self.tar_dic,
                        wei_dic = self.wei_dic)
                elif self.model == "bi_lstm":
                    cross_model = cross_bi_lstm(
                        target_vec_dic = self.target_vec_dic,
                        sequence_length = self.sequence_length,
                        num_classes = len(self.emotion_list),
                        vocab_size = self.vocab_size,
                        embedding_size = self.embedding_dim,
                        filter_sizes = self.filter_sizes,
                        num_filters = self.num_filters,
                        dropout_keep_prob = self.dropout_keep_prob,
                        l2_reg_lambda = self.l2_reg_lambda) 
                else:
                    pass

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate = cross_model.learning_rate)
                grads_and_vars = optimizer.compute_gradients(cross_model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                

                sess.run(tf.global_variables_initializer())

                max_accu = 0
                #training and test
                
                target_test_feature_pad, target_test_feature_seq_len = self.gene_pad_seq(self.target_test_feature) 

                for batch in self.all_batches:
                    y_batch, x_batch, f_cn, f_en = zip(*batch)
                    x_batch_pad, x_batch_seq_len = self.gene_pad_seq(x_batch)
                    # train
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step < self.pre_step:
                        rate = self.op_step
                    else:
                        rate = self.op_step / 2.0
                    feed_dict = {
                        cross_model.seq_len: x_batch_seq_len,
                        cross_model.input_f_en: f_en[0],
                        cross_model.input_f_cn: f_cn[0],
                        cross_model.input_trans_en: self.sour_dic,
                        cross_model.input_trans_cn: self.tar_dic,
                        cross_model.input_x: x_batch_pad,
                        cross_model.input_y: y_batch,
                        cross_model.dropout_keep_prob: self.dropout_keep_prob,
                        cross_model.cn_weight: 1,
                        cross_model.en_weight: 1,
                        cross_model.trans_weight: 1,
                        cross_model.learning_rate: rate
                    }
                    _, step, loss, kl, accuracy = sess.run(
                        [train_op, global_step, cross_model.loss, cross_model.kl, cross_model.accuracy],
                        feed_dict)
                    time_str = datetime.now().isoformat()
                    #print("train {}: step {}, loss {:g}, kl {:g}, acc {:g}".format(time_str, step, loss, kl, accuracy))

                    # test
                    #current_step = tf.train.global_step(sess, global_step)
                    if current_step > self.max_iter:
                        break
                    if current_step % self.evaluate_every == 0:
                        feed_dict = {
                            cross_model.seq_len: target_test_feature_seq_len,
                            cross_model.input_f_en: 0,
                            cross_model.input_f_cn: 1,
                            cross_model.input_trans_en: [0],
                            cross_model.input_trans_cn: [0],
                            cross_model.input_x: target_test_feature_pad,
                            cross_model.input_y: self.target_test_label,
                            cross_model.dropout_keep_prob: 1,
                            cross_model.cn_weight: 1,
                            cross_model.en_weight: 0,
                            cross_model.trans_weight: 0,
                            cross_model.learning_rate: rate}
                        step, loss, kl, accuracy = sess.run(
                            [global_step, cross_model.loss, cross_model.kl, cross_model.accuracy],
                            feed_dict)
                        if accuracy > max_accu:
                            max_accu = accuracy
                        time_str = datetime.now().isoformat()
                        print("eval  {}: step {}, loss {:g}, kl {:g}, acc {:g}, max_accu {:g}".format(time_str, step, loss, kl, accuracy, max_accu))
                        #print("eval  {}: loss {:g}".format(time_str, loss))
                return max_accu

    def load_single_data(self):
        print("Loading data...")
        target_dic_len = self.load_dic_len(self.target_dic_path)
        print("target dic " + self.target_dic_path + " len: " + str(target_dic_len))
        self.vocab_size = target_dic_len + 1
        self.target_vec_dic = {}
        self.target_vec_dic[0] = [0 for x in range(self.embedding_dim)]
        self.load_word_vec(self.target_vec_dic, self.word_vec_target, 1)
        self.target_vec_dic = [x for _, x in list(sorted(self.target_vec_dic.items(), key = lambda x:x[0]))]
        
        #print(len(self.target_vec_dic))
        #print(len(self.target_vec_dic[0]))

        target_label, target_feature = self.load_data(self.target_path)

        target_train_feature, self.target_test_feature, target_train_label, self.target_test_label = \
            self.shuffle_data(target_label, target_feature, self.part)

        print("Target Train/Test split: {:d}/{:d}".format(len(target_train_feature), len(self.target_test_feature)))

        self.all_batches = []
        for i in range(self.num_epochs):
            batches = []
            if self.part >= 1:
                part_epochs = self.part
            else:
                part_epochs = 1
            for j in range(part_epochs):
                target_batches = self.batch_iter(list(zip(target_train_label, target_train_feature)), self.batch_size)
                batches += target_batches
            if self.random_train:
                random.seed(self._seed)
            random.shuffle(batches)
            self.all_batches += batches
    

    def single_training(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
            sess = tf.Session(config = session_conf)
            with sess.as_default():
                if self.model == "cnn":
                    single_model = single_cnn(
                        target_vec_dic = self.target_vec_dic,                
                        sequence_length = self.sequence_length,
                        num_classes = len(self.emotion_list),
                        vocab_size = self.vocab_size,
                        embedding_size = self.embedding_dim,
                        filter_sizes = self.filter_sizes,
                        num_filters = self.num_filters,
                        l2_reg_lambda = self.l2_reg_lambda)
                elif self.model == "bi_lstm":
                    single_model = single_bi_lstm(
                        target_vec_dic = self.target_vec_dic,
                        sequence_length = self.sequence_length,
                        num_classes = len(self.emotion_list),
                        vocab_size = self.vocab_size,
                        embedding_size = self.embedding_dim,
                        filter_sizes = self.filter_sizes,
                        num_filters = self.num_filters,
                        l2_reg_lambda = self.l2_reg_lambda,
                        dropout_keep_prob = 0.5)
                elif self.model == "cnn_lstm":
                    single_model = single_cnn_lstm(
                        target_vec_dic = self.target_vec_dic,
                        sequence_length = self.sequence_length,
                        num_classes = len(self.emotion_list),
                        vocab_size = self.vocab_size,
                        embedding_size = self.embedding_dim,
                        filter_sizes = self.filter_sizes,
                        num_filters = self.num_filters,
                        l2_reg_lambda = self.l2_reg_lambda,
                        dropout_keep_prob = 0.5)
                else:
                    pass
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(self.op_step)
                grads_and_vars = optimizer.compute_gradients(single_model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    
                sess.run(tf.global_variables_initializer())

                max_accu = 0
                #training and test
                for batch in self.all_batches:
                    y_batch, x_batch = zip(*batch)
                    
                    x_batch_pad, x_batch_seq_len = self.gene_pad_seq(x_batch) 
                    feed_dict = {
                        single_model.input_x: x_batch_pad,
                        single_model.input_y: y_batch,
                        single_model.seq_len: x_batch_seq_len,
                        single_model.batch_size: len(y_batch),
                        single_model.dropout_keep_prob: 0.5,
                    }
                    _, step, loss, kl, accuracy = sess.run(
                        [train_op, global_step, single_model.loss, single_model.kl, single_model.accuracy],
                        feed_dict)
                    time_str = datetime.now().isoformat()
                    #print("train {}: step {}, loss {:g}, kl {:g}, acc {:g}".format(time_str, step, loss, kl, accuracy))

                    # test
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.evaluate_every == 0:
                        target_test_feature_pad, target_test_feature_seq_len = self.gene_pad_seq(self.target_test_feature)
                        feed_dict = {
                            single_model.input_x: target_test_feature_pad,
                            single_model.input_y: self.target_test_label,
                            single_model.seq_len: target_test_feature_seq_len,
                            single_model.batch_size: len(y_batch),
                            single_model.dropout_keep_prob: 1,
                        }
                        step, loss, kl, accuracy = sess.run(
                            [global_step, single_model.loss, single_model.kl, single_model.accuracy],
                            feed_dict)
                        if accuracy > max_accu:
                            max_accu = accuracy
                        time_str = datetime.now().isoformat()
                        print("eval  {}: step {}, loss {:g}, kl {:g}, acc {:g}, max_accu {:g}".format(time_str, step, loss, kl, accuracy, max_accu))
        return max_accu
