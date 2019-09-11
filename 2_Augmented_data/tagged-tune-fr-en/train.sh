ONMT_DIR=../../tools/OpenNMT-py
TOOL_DIR=../../tools
NAME=tag-fr-en
DATA_DIR=../data/bpe
RAW_DATA_DIR=../../data

mkdir onmt
mkdir result
mkdir models

# perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/valid/valid.en-fr.fr > ../data/tok/finetune.valid.en-fr.fr
# perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/valid/valid.fr-en.fr > ../data/tok/finetune.valid.fr-en.fr
# perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/valid/valid.en-fr.en > ../data/tok/finetune.valid.en-fr.en
# perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/valid/valid.fr-en.en > ../data/tok/finetune.valid.fr-en.en

# perl ${TOOL_DIR}/apply_bpe.py -i ../data/tok/finetune.valid.en-fr.fr -c ${DATA_DIR}/fr.bpe.50k -o ${DATA_DIR}/finetune.valid.en-fr.fr
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/tok/finetune.valid.fr-en.fr -c ${DATA_DIR}/fr.bpe.50k -o ${DATA_DIR}/finetune.valid.fr-en.fr
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/tok/finetune.valid.en-fr.en -c ${DATA_DIR}/en.bpe.50k -o ${DATA_DIR}/finetune.valid.en-fr.en
# perl ${TOOL_DIR}/apply_bpe.py -i ../data/tok/finetune.valid.fr-en.en -c ${DATA_DIR}/en.bpe.50k -o ${DATA_DIR}/finetune.valid.fr-en.en


# python $TOOL_DIR/add_tag.py -input ../data/bpe/train.bpe.fr -tag '<clean_s>' -output ../data/tag/train.bpe.fr
# python $TOOL_DIR/add_tag.py -input ../data/bpe/valid.bpe.fr -tag '<clean_s>' -output ../data/tag/valid.bpe.fr
# python $TOOL_DIR/add_tag.py -input ../data/bpe/test.bpe.news2014.fr -tag '<clean_s>' -output ../data/tag/test.bpe.news2014.fr
# python $TOOL_DIR/add_tag.py -input ../data/bpe/test.bpe.newsdiscuss2015.fr -tag '<clean_s>' -output ../data/tag/test.bpe.newsdiscuss2015.fr

# python $TOOL_DIR/add_tag.py -input ../data/bpe/finetune.en-fr.bpe.fr -tag '<MTNT_rev>' -output ../data/tag/finetune.en-fr.bpe.fr
# python $TOOL_DIR/add_tag.py -input ../data/bpe/finetune.valid.en-fr.fr -tag '<MTNT_rev>' -output ../data/tag/finetune.valid.en-fr.fr
# python $TOOL_DIR/add_tag.py -input ../data/bpe/finetune.fr-en.bpe.fr -tag '<MTNT_s>' -output ../data/tag/finetune.fr-en.bpe.fr
# python $TOOL_DIR/add_tag.py -input ../data/bpe/finetune.valid.fr-en.fr -tag '<MTNT_s>' -output ../data/tag/finetune.valid.fr-en.fr
# python $TOOL_DIR/add_tag.py -input ../data/bpe/test.fr-en.bpe.fr -tag '<MTNT_s>' -output ../data/tag/test.fr-en.bpe.fr
# python $TOOL_DIR/add_tag.py -input ../data/bpe/test.fr-en.bpe.MTNT2019.fr -tag '<MTNT_s>' -output ../data/tag/test.fr-en.bpe.MTNT2019.fr

# python $TOOL_DIR/add_tag.py -input ../data/bpe/finetune.iwslt.fr-en.bpe.fr -tag '<IWSLT_s>' -output ../data/tag/finetune.iwslt.fr-en.bpe.fr

# python $TOOL_DIR/add_tag.py -input ../data/bpe/noisy.fr.bpe.src -output ../data/tag/noisy.fr.bpe.src -tag '<FT_s>'
# python $TOOL_DIR/add_tag.py -input ../data/bpe/noisy.fr.bpe.tgt -output ../data/tag/noisy.fr.bpe.tgt -tag '<BT_s>'

# python $TOOL_DIR/add_tag.py -input ../data/NFR/nfr.fr-en.fr.bpe -output ../data/tag/nfr.fr-en.fr.bpe -tag '<NFR_s>'
# python $TOOL_DIR/add_tag.py -input ../data/NFR/nfr.fr-en.mono.fr.bpe -output ../data/tag/nfr.fr-en.mono.fr.bpe -tag '<NFR_mono>'

# python $TOOL_DIR/add_tag.py -input ../data/NFR/nfr.en-fr.fr.bpe -output ../data/tag/nfr.en-fr.fr.bpe -tag '<NFR_s_rev>'
# python $TOOL_DIR/add_tag.py -input ../data/NFR/nfr.en-fr.mono.fr.bpe -output ../data/tag/nfr.en-fr.mono.fr.bpe -tag '<NFR_mono_rev>'


# cat ../data/tag/train.bpe.fr ../data/tag/finetune.en-fr.bpe.fr ../data/tag/finetune.fr-en.bpe.fr ../data/tag/finetune.iwslt.fr-en.bpe.fr ../data/tag/noisy.fr.bpe.src ../data/tag/noisy.fr.bpe.tgt ../data/tag/nfr.fr-en.fr.bpe ../data/tag/nfr.fr-en.mono.fr.bpe ../data/tag/nfr.en-fr.fr.bpe ../data/tag/nfr.en-fr.mono.fr.bpe > ../data/tag/mix.fr-en.fr
# cat ../data/bpe/train.bpe.en ../data/bpe/finetune.en-fr.bpe.en ../data/bpe/finetune.fr-en.bpe.en ../data/bpe/finetune.iwslt.fr-en.bpe.en ../data/bpe/noisy.en.bpe.tgt ../data/bpe/noisy.en.bpe.src ../data/NFR/nfr.fr-en.en.bpe ../data/NFR/nfr.fr-en.mono.en.bpe ../data/NFR/nfr.en-fr.en.bpe ../data/NFR/nfr.en-fr.mono.en.bpe > ../data/tag/mix.fr-en.en


# building vocabulary
# onmt-build-vocab --save_vocab onmt/train.vocab.en --size 50000 ../data/tag/mix.fr-en.en
# onmt-build-vocab --save_vocab onmt/train.vocab.fr --size 50000 ../data/tag/mix.fr-en.fr

# python ${ONMT_DIR}/preprocess.py -train_src ../data/tag/train.bpe.fr \
#                                  -train_tgt ../data/bpe/train.bpe.en \
#                                  -valid_src ../data/tag/valid.bpe.fr \
#                                  -valid_tgt ../data/bpe/valid.bpe.en \
#                                  -src_vocab_size 50100 \
#                                  -tgt_vocab_size 50100 \
#                                  -save_data onmt/${NAME} \
#                                  -src_vocab onmt/train.vocab.fr \
#                                  -tgt_vocab onmt/train.vocab.en \
#                                  -src_seq_length 70 \
#                                  -tgt_seq_length 70 \
#                                  -seed 1234

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
                            -save_checkpoint_steps 5000 \
                            -batch_size 4096 \
                            -batch_type tokens \
                            -valid_steps 5000 \
                            -train_steps 2000000 \
                            -early_stopping 10 \
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
			    			-gpu_ranks 0
