import sys


def re_complete(path, path_re, embedding_size=128, value=0.0001):
    pre_line = 0
    new_line = [value for r in range(embedding_size)]
    line_dic = {}
    with open(path) as f,\
         open(path_re, "w") as fw:
        for line in f:
            line = line.strip()
            if embedding_size == 128:
                idx, _ = line.split(" ", 1)
            else:
                idx, _ = line.split("\t", 1)
            if idx == "</s>":
                print(line, file=fw)
                continue
            idx = int(idx)
            line_dic[idx] = line

        line_list = sorted(line_dic.items(), key=lambda x:x[0])
        for idx, line in line_list:
            for x in range(pre_line+1, idx):
                if embedding_size == 128:
                    print(" ".join([str(s) for s in ([x]+new_line)]), file=fw)
                else:
                    print("\t".join([str(s) for s in ([x]+new_line)]), file=fw)
            print(line, file=fw)
            pre_line = idx


if __name__ == "__main__":
    re_complete("pre_data/Reddit_st_vectors", "pre_data/Reddit_st_vectors_re")
    re_complete("pre_data/Reddit_pw_vectors", "pre_data/Reddit_pw_vectors_re")
    re_complete("pre_data/Reddit_st_tf_df", "pre_data/Reddit_st_tf_df_re", 2, 1)
    re_complete("pre_data/Reddit_pw_tf_df", "pre_data/Reddit_pw_tf_df_re", 2, 1)
