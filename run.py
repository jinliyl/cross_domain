# -*- coding: utf-8 -*-
from preprocess.load_data import load_data
from preprocess.translator import translator
from preprocess.statistic import label_statistic
from preprocess.make_index import make_index
from preprocess.make_index import transform
from neuralnet.train_nn import train_nn

def load_and_save():
    qq_data = load_data("QQ", "pre_data", "cn", "data/QQ/NewsQQ.txt", "data/QQ/LabelQQ.txt")
    qq_data.load_dataset()
    qq_data.save_dataset()

    sina_data = load_data("Sina", "pre_data", "cn", "data/Sina/NewsSina.txt", "data/Sina/LabelSina.txt")
    sina_data.load_dataset()
    sina_data.save_dataset()

    reddit_st_data = load_data("Reddit_st", "pre_data", "en", "data/Reddit_st", "")
    reddit_st_data.load_dataset()
    reddit_st_data.save_dataset()

    reddit_pw_data = load_data("Reddit_pw", "pre_data", "en", "data/Reddit_pw", "")
    reddit_pw_data.load_dataset()
    reddit_pw_data.save_dataset()


def translate():
    en_trans = translator("20151118000005686", "o6bBgmTZZg78x3LXqsB8", "http://api.fanyi.baidu.com/api/trans/vip/translate?")
    
    en_trans.copy("pre_data/Sina_dic", "pre_data/Sina_dic_trans")
    en_trans.copy("pre_data/QQ_dic", "pre_data/QQ_dic_trans")
    en_trans.copy("pre_data/Reddit_st_dic", "pre_data/Reddit_st_dic_trans")
    en_trans.copy("pre_data/Reddit_pw_dic", "pre_data/Reddit_pw_dic_trans")


def statistics():
    label_statistic("pre_data/QQ")
    label_statistic("pre_data/Sina")
    label_statistic("pre_data/Reddit_st")
    label_statistic("pre_data/Reddit_pw")

def index_content():
    qq_index = make_index("pre_data/QQ", "data/QQ/qqlabel", "pre_data/QQ_dic")
    qq_index.process_index()

    sina_index = make_index("pre_data/Sina", "data/Sina/sinalabel", "pre_data/Sina_dic")
    sina_index.process_index()

    reddit_st_index = make_index("pre_data/Reddit_st", "data/Reddit_st/redditlabel", "pre_data/Reddit_st_dic")
    reddit_st_index.process_index()

    reddit_pw_index = make_index("pre_data/Reddit_pw", "data/Reddit_pw/redditlabel", "pre_data/Reddit_pw_dic")
    reddit_pw_index.process_index()


    # source target source_dic source->target
    transform("pre_data/Sina_dic", "pre_data/QQ_dic", "pre_data/Sina_dic_trans", "pre_data/Sina--QQ_trans")
    transform("pre_data/QQ_dic", "pre_data/Sina_dic", "pre_data/QQ_dic_trans", "pre_data/QQ--Sina_trans")
    transform("pre_data/Reddit_st_dic", "pre_data/Reddit_pw_dic", "pre_data/Reddit_st_dic_trans", "pre_data/Reddit_st--pw_trans")
    transform("pre_data/Reddit_pw_dic", "pre_data/Reddit_st_dic", "pre_data/Reddit_pw_dic_trans", "pre_data/Reddit_pw--st_trans")



def train_test(t):
    if t == 0:
        cross_train = train_nn(
            #emotion_list = ["happy", "touched", "sympathetic", "angry", "amused", "sad", "surprised", "anxious"],
            emotion_list = ["happy", "sympathetic", "angry", "amused", "sad", "surprised"],
            target_dic_path = "pre_data/QQ_dic",
            source_dic_path = "pre_data/Sina_dic",
            target_path = "pre_data/QQ_index",
            source_path = "pre_data/Sina_index",
            transfer_path = "pre_data/Sina_QQ_trans",
            part = 16,
            model = "bi_lstm",
            batch_size = 32,
            num_filters = 128,
            sequence_length = 150,
            embedding_dim = 128,
            op_step = 5e-4,
            word_vec_target = "pre_data/QQ_vectors", 
            word_vec_source = "pre_data/Sina_vectors")
        return cross_train.run()
    else:
        single_train = train_nn(
            emotion_list = ["happy", "sympathetic", "angry", "amused", "sad", "surprised"],
            #emotion_list = ["happy", "touched", "sympathetic", "angry", "amused", "sad", "surprised", "anxious"],
            target_dic_path = "pre_data/QQ_dic",
            target_path = "pre_data/QQ_index",
            part = 64,
            model = "cnn",
            cross_lingual = False,
            filter_sizes = [1, 2, 3],
            embedding_dim = 128,
            batch_size = 32,
            num_filters = 128,
            sequence_length = 150,
            op_step = 1e-3, 
            word_vec_target = "pre_data/QQ_vectors")
        return single_train.run()

if __name__ == "__main__":
    load_and_save()
    translate()
    statistics()
    index_content()
    #train_test(1)