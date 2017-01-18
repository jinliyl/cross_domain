#time ./word2vec -train ../pre_data/QQ_index_noflag -output ../pre_data/QQ_vectors -cbow 1 -size 128 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 2 -binary 0 -iter 200 -min-count 1
#time ./word2vec -train ../pre_data/Sina_index_noflag -output ../pre_data/Sina_vectors -cbow 1 -size 128 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 2 -binary 0 -iter 200 -min-count 1
time ./word2vec -train ../pre_data/QQ_index_noflag -output ../pre_data/QQ_vectors -cbow 0 -size 128 -window 5 -negative 0 -hs 1 -sample 1e-4 -threads 4 -binary 0 -iter 200 -min-count 1
time ./word2vec -train ../pre_data/Sina_index_noflag -output ../pre_data/Sina_vectors -cbow 0 -size 128 -window 5 -negative 0 -hs 1 -sample 1e-4 -threads 4 -binary 0 -iter 200 -min-count 1
