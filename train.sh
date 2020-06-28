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
2000  \
>> lstm.log  2>&1 &


