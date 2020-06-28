#!/bin/bash
source activate py35
nohup               \
python ./trainer.py     \
--train                 \
--name=seq2seq      \
--model_name            \
lm                    \
--save_dir  \
seq2seq \
--data_dir  \
./train_data  \
--max_epoch \
10  \
--pretrain \
seq2seq/craw1.npz \
>> lstm.log  2>&1 &


