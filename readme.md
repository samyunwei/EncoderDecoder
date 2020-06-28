#GreedyEncoderDecoder 
#show_main.py
--train --name=seq2seq --model_name lm --save_dir seq2seq --data_dir ./train_data --load_dir seq2seq/lm_0_3.3503.pkl

#Train
#trainer.py
--train --name=seq2seq --model_name lm --save_dir seq2seq --data_dir ./train_data --max_epoch 1






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

