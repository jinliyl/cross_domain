# -*- coding: utf-8 -*-

class make_index:
    
    def __init__(self, data_path, label_path, dic_path):
        self.data_path = data_path
        self.label_path = label_path
        self.dic_path = dic_path
        
        self.word_dic = {}
        self.label_emotion_dic = {}
        self.word_dic_tf = {}
        self.word_dic_df = {}

    def load_word_dic(self):
        with open(self.dic_path) as f:
            for line in f:
                line = line.strip()
                word, count = line.split("\t")
                self.word_dic[word] = count
    
    def load_label_dic(self):
        with open(self.label_path) as f:
            for line in f:
                line = line.strip()
                label, emotion = line.split(" ")
                self.label_emotion_dic[label] = emotion

    def load_save_comment(self):
        self.doc_count = 0
        with open(self.data_path) as f, open(self.data_path + "_index", "w") as fw, open(self.data_path + "_index_noemo", "w") as fw2:
            for line in f:
                self.doc_count += 1
                line = line.strip()
                label, content = line.split("\t")
                if label not in self.label_emotion_dic.keys():
                    continue
                emotion = self.label_emotion_dic[label]
                word_list = []
                for word in content.split(" "):
                    word_list.append(str(self.word_dic[word]))
                for word in word_list:
                    if word in self.word_dic_tf.keys():
                        self.word_dic_tf[word] += 1
                    else:
                        self.word_dic_tf[word] = 1

                for word in set(word_list):
                    if word in self.word_dic_df.keys():
                        self.word_dic_df[word] += 1
                    else:
                        self.word_dic_df[word] = 1

                print("\t".join([emotion, " ".join(word_list)]), file = fw)
                print("\t".join([" ".join(word_list)]), file = fw2)

    def write_dic_tf_df(self):
        with open(self.data_path + "_tf_df", "w") as fw:
            keys_list = sorted(self.word_dic_tf.keys(), key=lambda x:int(x), reverse=True)
            max_tf = max([int(x) for x in self.word_dic_tf.values()])
            for k in keys_list:
                tf = str(self.word_dic_tf[k] * 1.0 / max_tf)
                df = str(self.word_dic_df[k] * 1.0 / self.doc_count)
                print("\t".join([k, tf, df]), file = fw)

    def process_index(self):
        self.load_word_dic()
        self.load_label_dic()
        self.load_save_comment()
        self.write_dic_tf_df()

def transform(en_path, cn_path, trans_dic_path, output_path):
        en_dic = {}
        cn_dic = {}

        with open(en_path) as f:
            for line in f:
                line = line.strip()
                word, index = line.split("\t")
                en_dic[word] = index
        with open(cn_path) as f:
            for line in f:
                line = line.strip()
                word, index = line.split("\t")
                cn_dic[word] = index

        with open(trans_dic_path) as f, open(output_path, "w") as fw:
            for line in f:
                line = line.strip()
                ll = line.split("\t")
                if len(ll) != 2:
                    continue
                word_en, word_cn = ll
                index_en = en_dic[word_en]
                if word_cn not in cn_dic.keys():
                    continue
                index_cn = cn_dic[word_cn]
                print("\t".join([index_en, index_cn, word_en, word_cn]), file = fw)
