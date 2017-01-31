# -*- coding: utf-8 -*-
from preprocess.load_data import load_data
from preprocess.translator import translator
from preprocess.statistic import label_statistic
from preprocess.make_index import make_index
from preprocess.make_index import transform
from neuralnet.train_nn import train_nn
from datetime import datetime

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



def train_test(cross, part, model, name, op_step):
    if cross:
        cross_train = train_nn(
            #emotion_list = ["happy", "touched", "sympathetic", "angry", "amused", "sad", "surprised", "anxious"],
            #emotion_list = ["happy", "sympathetic", "angry", "amused", "sad", "surprised"],
            emotion_list = ["angry", "disgust", "happy", "sad", "surprised", "sympathetic"],
            target_dic_path = "pre_data/Reddit_st_dic",
            source_dic_path = "pre_data/Reddit_pw_dic",
            target_path = "pre_data/Reddit_st_index",
            source_path = "pre_data/Reddit_pw_index",
            transfer_path = "pre_data/Reddit_pw--st_trans",
            part = part,
            model = model,
            batch_size = 32,
            num_filters = 128,
            sequence_length = 150,
            embedding_dim = 128,
            op_step = op_step,
            word_vec_target = "pre_data/Reddit_st_vectors_re", 
            word_vec_source = "pre_data/Reddit_pw_vectors_re",
            tf_df_target = "pre_data/Reddit_st_tf_df_re",
            tf_df_source = "pre_data/Reddit_pw_tf_df_re",
            max_iter = 900)
        return cross_train.run()
    else:
        single_train = train_nn(
            #emotion_list = ["happy", "sympathetic", "angry", "amused", "sad", "surprised"],
            #emotion_list = ["happy", "touched", "sympathetic", "angry", "amused", "sad", "surprised", "anxious"],
            emotion_list = ["angry", "disgust", "happy", "sad", "surprised", "sympathetic"],
            target_dic_path = "pre_data/Reddit_st_dic",
            target_path = "pre_data/Reddit_st_index",
            part = part,
            model = model,
            cross_lingual = False,
            filter_sizes = [1, 2, 3],
            embedding_dim = 128,
            batch_size = 32,
            num_filters = 128,
            sequence_length = 150,
            op_step = op_step, 
            word_vec_target = "pre_data/Reddit_st_vectors_re",
            max_iter = 700,
            dropout_keep_prob = 0.5)
        return single_train.run()

if __name__ == "__main__":
    '''
    load_and_save()
    translate()
    statistics()
    index_content()
    '''
    for x in [4, 8, 16, 32, 64]:
        res = []
        para = [True, x, "bi_lstm", "Reddit_st", 4e-4]
        time_str = datetime.now().isoformat()
        f = open("./result/" + time_str, "w")
        for i in range(10):
            print(str(i) + " times training...")
            accu = train_test(para[0], para[1], para[2], para[3], para[4])
            res.append(accu)
            print(accu, file = f)
            print("")
            f.flush()
        print(res, file=f)
        print(para, file=f)
        f.close()

    for x in [4, 8, 16, 32, 64]:
        res = []
        para = [True, x, "cnn", "Reddit_st", 1e-3]
        time_str = datetime.now().isoformat()
        f = open("./result/" + time_str, "w")
        for i in range(10):
            print(str(i) + " times training...")
            accu = train_test(para[0], para[1], para[2], para[3], para[4])
            res.append(accu)
            print(accu, file = f)
            print("")
            f.flush()
        print(res, file=f)
        print(para, file=f)
        f.close()

    for x in [4, 8, 16, 32, 64]:
        res = []
        para = [False, x, "bi_lstm", "Reddit_st", 4e-4]
        time_str = datetime.now().isoformat()
        f = open("./result/" + time_str, "w")
        for i in range(10):
            print(str(i) + " times training...")
            accu = train_test(para[0], para[1], para[2], para[3], para[4])
            res.append(accu)
            print(accu, file = f)
            print("")
            f.flush()
        print(res, file=f)
        print(para, file=f)
        f.close()

    for x in [4, 8, 16, 32, 64]:
        res = []
        para = [False, x, "cnn", "Reddit_st", 1e-3]
        time_str = datetime.now().isoformat()
        f = open("./result/" + time_str, "w")
        for i in range(10):
            print(str(i) + " times training...")
            accu = train_test(para[0], para[1], para[2], para[3], para[4])
            res.append(accu)
            print(accu, file = f)
            print("")
            f.flush()
        print(res, file=f)
        print(para, file=f)
        f.close()


