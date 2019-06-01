#!/bin/bash
# building pipeline for baseline on fr-en translation
# train: newscommentary + europarl
# valid: newsdiscussdev2015
# test: newstest2014/newsdiscusstest2015/MTNT-test-fr-en
# fine-tune: MTNT
source path.config

mkdir ${VOCAB_DIR}
mkdir ${DATA_DIR}
mkdir result
mkdir models

cp -r ../base-fr-en/data ./
cp ../base-fr-en/models/fr-en* ./models

perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/train/train.fr-en.en > ${VOCAB_DIR}/train.finetune.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/train/train.fr-en.fr > ${VOCAB_DIR}/train.finetune.tok.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/valid/valid.fr-en.en > ${VOCAB_DIR}/valid.finetune.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/valid/valid.fr-en.fr > ${VOCAB_DIR}/valid.finetune.tok.fr

python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.en.bpe.16k < ${VOCAB_DIR}/train.finetune.tok.en > ${DATA_DIR}/train.finetune.bpe.16k.en
python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.en.bpe.16k < ${VOCAB_DIR}/valid.finetune.tok.en > ${DATA_DIR}/valid.finetune.bpe.16k.en
python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.fr.bpe.16k < ${VOCAB_DIR}/train.finetune.tok.fr > ${DATA_DIR}/train.finetune.bpe.16k.fr
python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.fr.bpe.16k < ${VOCAB_DIR}/valid.finetune.tok.fr > ${DATA_DIR}/valid.finetune.bpe.16k.fr

# building vocabulary
python ${ONMT_DIR}/preprocess.py -train_src ${DATA_DIR}/train.finetune.bpe.16k.fr \
                                 -train_tgt ${DATA_DIR}/train.finetune.bpe.16k.en \
                                 -valid_src ${DATA_DIR}/valid.finetune.bpe.16k.fr \
                                 -valid_tgt ${DATA_DIR}/valid.finetune.bpe.16k.en \
                                 -src_vocab ${DATA_DIR}/train.vocab.fr \
                                 -tgt_vocab ${DATA_DIR}/train.vocab.en \
                                 -save_data ${DATA_DIR}/onmt-vocab/${NAME} \
                                 -src_seq_length 70 \
                                 -tgt_seq_length 70 \
                                 -seed 1234

# training
python ${ONMT_DIR}/train.py -word_vec_size 512 \
                            -encoder_type brnn \
                            -decoder_type rnn \
                            -rnn_size 1024 \
                            -layers 2 \
                            -bridge \
                            -global_attention mlp \
                            -data ${DATA_DIR}/onmt-vocab/${NAME} \
                            -save_model models/${NAME} \
                            -train_from models/fr-en_step_70000.pt \
                            -save_checkpoint_steps 1000 \
                            -batch_size 4000 \
                            -valid_batch_size 10 \
                            -batch_type tokens \
                            -valid_steps 1000 \
                            -train_steps 100000 \
                            -early_stopping 5 \
                            -keep_checkpoint 6 \
                            -optim adam \
                            -dropout 0.3 \
                            -label_smoothing 0.1 \
                            -learning_rate 0.0002 \
                            -decay_steps 1000 \
                            -start_decay_steps 10000 \
                            -report_every 100 \
                            -log_file train_log.log \
                            -tensorboard \
                            -tensorboard_log_dir models \
                            -seed 1234 \
                            -world_size 1 \
                            -gpu_ranks 0
