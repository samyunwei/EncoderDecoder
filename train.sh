#!/bin/bash
source activate py35
nohup                   \
python ./trainer.py     \
--train                 \
--name=senti2class      \
--model_name            \
lstm                    \
--save_dir              \
senti2class_models      \
--data_dir              \
/home/demo1/womin/piguanghua/data/ \
--vocab_path            \
/home/demo1/womin/piguanghua/data/Amazon_Reviews_for_Sentiment_Analysis/vocab.txt       \
--max_epoch 40  \
>> lstm.log  2>&1 &




--train
--name=seq2seq
--model_name
lm
--save_dir
seq2seq
--data_dir
/home/demo1/womin/piguanghua/data/cornell_movie_dialogs_corpus
--max_epoch
1


--train
--name=seq2seq
--model_name
lm
--save_dir
seq2seq
--data_dir
./train_data
--max_epoch
1


#dev
--train
--name=seq2seq
--model_name
lm
--save_dir
seq2seq
--data_dir
./train_data
--load_dir
seq2seq/lm_0_3.3503.pkl

