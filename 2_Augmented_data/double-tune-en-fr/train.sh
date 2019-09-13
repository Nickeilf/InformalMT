ONMT_DIR=../../tools/OpenNMT-py
TOOL_DIR=../../tools
NAME=iter-en-fr
DATA_DIR=../data/bpe
RAW_DATA_DIR=../../data

mkdir onmt
mkdir result
mkdir models
STEP=215500

# cp -r ../mix-en-fr/onmt/train.vocab.* ./onmt
# cp -r ../mix-en-fr/models/mix-en-fr_step_${STEP}.pt ./models

# perl ${TOOL_DIR}/tokenizer.perl -l fr < ../data/NFR/nfr.en-fr.fr > ../data/NFR/nfr.en-fr.fr.tok
# perl ${TOOL_DIR}/tokenizer.perl -l fr < ../data/NFR/nfr.en-fr.mono.fr > ../data/NFR/nfr.en-fr.mono.fr.tok
# perl ${TOOL_DIR}/tokenizer.perl -l en < ../data/NFR/nfr.en-fr.en > ../data/NFR/nfr.en-fr.en.tok
# perl ${TOOL_DIR}/tokenizer.perl -l en < ../data/NFR/nfr.en-fr.mono.en > ../data/NFR/nfr.en-fr.mono.en.tok

# perl ${TOOL_DIR}/apply_bpe.py -i ../data/NFR/nfr.en-fr.fr.tok -c ${DATA_DIR}/fr.bpe.50k -o ../data/NFR/nfr.en-fr.fr.bpe
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/NFR/nfr.en-fr.mono.fr.tok -c ${DATA_DIR}/fr.bpe.50k -o ../data/NFR/nfr.en-fr.mono.fr.bpe
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/NFR/nfr.en-fr.en.tok -c ${DATA_DIR}/en.bpe.50k -o ../data/NFR/nfr.en-fr.en.bpe
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/NFR/nfr.en-fr.mono.en.tok -c ${DATA_DIR}/en.bpe.50k -o ../data/NFR/nfr.en-fr.mono.en.bpe

# perl $TOOL_DIR/tokenizer.perl -l en < $RAW_DATA_DIR/fine-tune/train/asr.en-fr.en > ../data/tok/asr.en-fr.en
# perl $TOOL_DIR/tokenizer.perl -l fr < $RAW_DATA_DIR/fine-tune/train/asr.en-fr.fr > ../data/tok/asr.en-fr.fr
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/tok/asr.en-fr.fr -c ${DATA_DIR}/fr.bpe.50k -o ${DATA_DIR}/asr.en-fr.fr
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/tok/asr.en-fr.en -c ${DATA_DIR}/en.bpe.50k -o ${DATA_DIR}/asr.en-fr.en



cat ${DATA_DIR}/finetune.en-fr.bpe.fr > mix.fr
cat ${DATA_DIR}/finetune.en-fr.bpe.en > mix.en


# building vocabulary
python ${ONMT_DIR}/preprocess.py -train_src mix.en \
                                 -train_tgt mix.fr \
                                 -valid_src ${DATA_DIR}/finetune.valid.bpe.en \
                                 -valid_tgt ${DATA_DIR}/finetune.valid.bpe.fr \
                                 -src_vocab onmt/train.vocab.en \
                                 -tgt_vocab onmt/train.vocab.fr \
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
                            -save_checkpoint_steps 25 \
                            -batch_size 4096 \
                            -batch_type tokens \
                            -valid_steps 25 \
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
							-train_from "models/all(merge)-en-fr_step_${STEP}.pt"
