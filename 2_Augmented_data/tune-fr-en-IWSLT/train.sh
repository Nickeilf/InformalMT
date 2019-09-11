#!/bin/bash
ONMT_DIR=../../tools/OpenNMT-py
NAME=tuneIWSLT-fr-en
DATA_DIR=../data/bpe

mkdir onmt
mkdir result
mkdir models
STEP=155000

cp -r ../transformer-fr-en/onmt/train.vocab.* ./onmt
cp -r ../transformer-fr-en/models/BASE-fr-en_step_${STEP}.pt ./models

# building vocabulary
python ${ONMT_DIR}/preprocess.py -train_src ${DATA_DIR}/finetune.iwslt.fr-en.bpe.fr \
                                 -train_tgt ${DATA_DIR}/finetune.iwslt.fr-en.bpe.en \
                                 -valid_src ${DATA_DIR}/finetune.valid.bpe.fr \
                                 -valid_tgt ${DATA_DIR}/finetune.valid.bpe.en \
                                 -src_vocab onmt/train.vocab.fr \
                                 -tgt_vocab onmt/train.vocab.en \
                                 -save_data onmt/${NAME} \
								 --src_words_min_frequency 1 \
								 --tgt_words_min_frequency 1 \
                                 -src_seq_length 70 \
                                 -tgt_seq_length 70 \
                                 -seed 1234

# training (reduce batch size because of GPU limit)
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
                            -save_checkpoint_steps 50 \
                            -batch_size 4096 \
                            -batch_type tokens \
                            -valid_steps 50 \
                            -train_steps 2000000 \
                            -early_stopping 5 \
                            -keep_checkpoint 8 \
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
							-valid_batch_size 5 \
                            -seed 1234 \
							-exp ${NAME} \
			    			-world_size 1 \
			    			-gpu_ranks 0 \
							-train_from models/BASE-fr-en_step_${STEP}.pt
