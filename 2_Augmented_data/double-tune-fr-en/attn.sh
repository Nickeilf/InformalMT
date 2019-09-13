#!/bin/bash
ONMT_DIR=../../tools/OpenNMT-py
NAME=iter-fr-en
DATA_DIR=../data/bpe
TOOL_DIR=../../tools
RAW_DATA_DIR=../../data

CHECKPOINT=155320
# evaluation on MTNT
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src test.txt \
                                -output test.tran.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
                                -attn_debug \
								-fp32