#!/bin/bash
source activate py35
nohup                   \
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
100  \
--load_dir  \
./seq2seq/lm_99_5650.7593.pkl\
>> lstm.log  2>&1 &


