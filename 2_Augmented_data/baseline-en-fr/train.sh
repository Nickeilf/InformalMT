#!/bin/bash

ONMT_DIR=../../tools/OpenNMT-py
NAME=BASE-en-fr
DATA_DIR=../data/bpe
mkdir result

# building vocabulary
mkdir onmt
onmt-build-vocab --save_vocab onmt/train.vocab.en ${DATA_DIR}/train.bpe.en
onmt-build-vocab --save_vocab onmt/train.vocab.fr ${DATA_DIR}/train.bpe.fr
python ${ONMT_DIR}/preprocess.py -train_src ${DATA_DIR}/train.bpe.en \
                                 -train_tgt ${DATA_DIR}/train.bpe.fr \
                                 -valid_src ${DATA_DIR}/valid.bpe.en \
                                 -valid_tgt ${DATA_DIR}/valid.bpe.fr \
								 -src_vocab onmt/train.vocab.en \
                                 -tgt_vocab onmt/train.vocab.fr \
                                 --src_words_min_frequency 1 \
								 --tgt_words_min_frequency 1 \
                                 -save_data onmt/${NAME} \
                                 -src_seq_length 70 \
                                 -tgt_seq_length 70 \
                                 -seed 1234

# training
# add shared vocab
CUDA_VISIBLE_DEVICES=0
python ${ONMT_DIR}/train.py -word_vec_size 512 \
                            -encoder_type transformer \
                            -decoder_type transformer \
                            -layers 6 \
							-transformer_ff 2048 \
							-rnn_size 512 \
							-accum_count 8 \
							-heads 8 \
                            -data onmt/${NAME} \
                            -save_model models/${NAME} \
                            -save_checkpoint_steps 5000 \
                            -batch_size 4096 \
                            -batch_type tokens \
                            -valid_steps 5000 \
                            -train_steps 2000000 \
                            -early_stopping 16 \
                            -keep_checkpoint 30 \
							-max_generator_batches 2 \
							-param_init 0.0 \
							-param_init_glorot \
							-position_encoding \
                            -optim adam \
							-adam_beta1 0.9 \
							-adam_beta2 0.998 \
                            -dropout 0.1 \
                            -label_smoothing 0.1 \
                            -learning_rate 2.0 \
							-decay_method noam \
							-max_grad_norm 0.0 \
							-warmup_steps 8000 \
                            -log_file ${NAME}.log \
							-report_every 50 \
                            -tensorboard \
                            -tensorboard_log_dir models \
                            -seed 1234 \
							-exp ${NAME} \
			    			-world_size 1 \
			    			-gpu_ranks 0
