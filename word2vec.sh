echo "QQ"
time ./word2vec_tool/word2vec -train ./pre_data/QQ_index_noemo -output ./pre_data/QQ_vectors -cbow 0 -size 128 -window 5 -negative 0 -hs 1 -sample 1e-4 -threads 2 -binary 0 -iter 500 -min-count 1
echo "Sina"
time ./word2vec_tool/word2vec -train ./pre_data/Sina_index_noemo -output ./pre_data/Sina_vectors -cbow 0 -size 128 -window 5 -negative 0 -hs 1 -sample 1e-4 -threads 2 -binary 0 -iter 500 -min-count 1
echo "Reddit_st"
time ./word2vec_tool/word2vec -train ./pre_data/Reddit_st_index_noemo -output ./pre_data/Reddit_st_vectors -cbow 0 -size 128 -window 5 -negative 0 -hs 1 -sample 1e-4 -threads 2 -binary 0 -iter 500 -min-count 1
echo "Reddit_pw"
time ./word2vec_tool/word2vec -train ./pre_data/Reddit_pw_index_noemo -output ./pre_data/Reddit_pw_vectors -cbow 0 -size 128 -window 5 -negative 0 -hs 1 -sample 1e-4 -threads 2 -binary 0 -iter 500 -min-count 1



