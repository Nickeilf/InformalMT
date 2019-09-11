ONMT_DIR=../../tools/OpenNMT-py
TOOL_DIR=../../tools
NAME=tagtune-fr-en
DATA_DIR=../data/bpe
RAW_DATA_DIR=../../data

mkdir onmt
mkdir result
mkdir models

# ../data/tag/nfr.fr-en.fr.bpe ../data/tag/nfr.fr-en.mono.fr.bpe
# ../data/NFR/nfr.fr-en.en.bpe ../data/NFR/nfr.fr-en.mono.en.bpe

cat ../data/tag/finetune.en-fr.bpe.fr ../data/tag/finetune.iwslt.fr-en.bpe.fr ../data/tag/finetune.fr-en.bpe.fr ../data/tag/noisy.fr.bpe.src ../data/tag/noisy.fr.bpe.tgt ../data/tag/nfr.fr-en.fr.bpe ../data/tag/nfr.fr-en.mono.fr.bpe> ../data/tag/mixtune.fr-en.fr
cat ../data/bpe/finetune.en-fr.bpe.en ../data/bpe/finetune.iwslt.fr-en.bpe.en ../data/bpe/finetune.fr-en.bpe.en ../data/bpe/noisy.en.bpe.tgt ../data/bpe/noisy.en.bpe.src ../data/NFR/nfr.fr-en.en.bpe ../data/NFR/nfr.fr-en.mono.en.bpe> ../data/tag/mixtune.fr-en.en


# building vocabulary
python ${ONMT_DIR}/preprocess.py -train_src ../data/tag/mixtune.fr-en.fr \
                                 -train_tgt ../data/tag/mixtune.fr-en.en \
                                 -valid_src ../data/tag/finetune.valid.fr-en.fr \
                                 -valid_tgt ../data/bpe/finetune.valid.fr-en.en \
                                 -src_vocab_size 50100 \
                                 -tgt_vocab_size 50100 \
                                 -save_data onmt/${NAME} \
                                 -src_vocab onmt/train.vocab.fr \
                                 -tgt_vocab onmt/train.vocab.en \
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
                            -save_checkpoint_steps 100 \
                            -batch_size 4096 \
                            -batch_type tokens \
                            -valid_steps 100 \
                            -train_steps 2000000 \
                            -early_stopping 15 \
                            -keep_checkpoint 20 \
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
                            -train_from models/tag-fr-en_step_190000.pt