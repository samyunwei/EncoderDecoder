#!/bin/bash
source activate py35
python ./app_main.py     \
--train                 \
--name=seq2seq      \
--model_name            \
lm                    \
--load_dir  \
./seq2seq/lm_55_2948.8387.pkl



