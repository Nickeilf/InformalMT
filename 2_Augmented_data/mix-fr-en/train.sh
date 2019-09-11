ONMT_DIR=../../tools/OpenNMT-py
TOOL_DIR=../../tools
NAME=mix_fm_nfr-fr-en
DATA_DIR=../data/bpe
RAW_DATA_DIR=../../data

mkdir onmt
mkdir result
mkdir models
STEP=155000

# cp -r ../baseline-fr-en/onmt/train.vocab.* ./onmt
# cp -r ../baseline-fr-en/models/BASE-fr-en_step_${STEP}.pt ./models

# perl ${TOOL_DIR}/tokenizer.perl -l fr < ../data/NFR/nfr.fr-en.fr > ../data/NFR/nfr.fr-en.fr.tok
# perl ${TOOL_DIR}/tokenizer.perl -l fr < ../data/NFR/nfr.fr-en.mono.fr > ../data/NFR/nfr.fr-en.mono.fr.tok
# perl ${TOOL_DIR}/tokenizer.perl -l en < ../data/NFR/nfr.fr-en.en > ../data/NFR/nfr.fr-en.en.tok
# perl ${TOOL_DIR}/tokenizer.perl -l en < ../data/NFR/nfr.fr-en.mono.en > ../data/NFR/nfr.fr-en.mono.en.tok

# perl ${TOOL_DIR}/apply_bpe.py -i ../data/NFR/nfr.fr-en.fr.tok -c ${DATA_DIR}/fr.bpe.50k -o ../data/NFR/nfr.fr-en.fr.bpe
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/NFR/nfr.fr-en.mono.fr.tok -c ${DATA_DIR}/fr.bpe.50k -o ../data/NFR/nfr.fr-en.mono.fr.bpe
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/NFR/nfr.fr-en.en.tok -c ${DATA_DIR}/en.bpe.50k -o ../data/NFR/nfr.fr-en.en.bpe
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/NFR/nfr.fr-en.mono.en.tok -c ${DATA_DIR}/en.bpe.50k -o ../data/NFR/nfr.fr-en.mono.en.bpe


cat ${DATA_DIR}/finetune.train.bpe.fr ../data/NFR/nfr.fr-en.fr.bpe ../data/NFR/nfr.fr-en.mono.fr.bpe > mix.fr
cat ${DATA_DIR}/finetune.train.bpe.en ../data/NFR/nfr.fr-en.en.bpe ../data/NFR/nfr.fr-en.mono.en.bpe > mix.en

# # building vocabulary
python ${ONMT_DIR}/preprocess.py -train_src mix.fr \
                                 -train_tgt mix.en \
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
                            -early_stopping 8 \
                            -keep_checkpoint 15 \
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
							-valid_batch_size 5 \
                            -seed 1234 \
							-exp ${NAME} \
			    			-world_size 1 \
			    			-gpu_ranks 0 \
							-train_from models/BASE-fr-en_step_${STEP}.pt
