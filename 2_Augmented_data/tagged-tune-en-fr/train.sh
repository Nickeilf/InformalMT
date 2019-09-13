ONMT_DIR=../tools/OpenNMT-py
TOOL_DIR=../tools
NAME=tag-en-fr
RAW_DATA_DIR=../data
DATA_DIR=/data/zli/InformalMT/Back-translation/data

mkdir onmt
mkdir result
mkdir models
#

# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/train.bpe.en -tag '<clean_s>' -output $DATA_DIR/tag/train.bpe.en
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/valid.bpe.en -tag '<clean_s>' -output $DATA_DIR/tag/valid.bpe.en
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/test.bpe.news2014.en -tag '<clean_s>' -output $DATA_DIR/tag/test.bpe.news2014.en
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/test.bpe.newsdiscuss2015.en -tag '<clean_s>' -output $DATA_DIR/tag/test.bpe.newsdiscuss2015.en


# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/finetune.fr-en.bpe.en -tag '<MTNT_rev>' -output $DATA_DIR/tag/finetune.fr-en.bpe.en
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/finetune.en-fr.bpe.en -tag '<MTNT_s>' -output $DATA_DIR/tag/finetune.en-fr.bpe.en
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/finetune.valid.en-fr.en -tag '<MTNT_s>' -output $DATA_DIR/tag/finetune.valid.en-fr.en
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/test.en-fr.bpe.en -tag '<MTNT_s>' -output $DATA_DIR/tag/test.en-fr.bpe.en
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/test.en-fr.bpe.MTNT2019.en -tag '<MTNT_s>' -output $DATA_DIR/tag/test.en-fr.bpe.MTNT2019.en


# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/finetune.iwslt.en-fr.bpe.en -tag '<IWSLT_s>' -output $DATA_DIR/tag/finetune.iwslt.en-fr.bpe.en

# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/noisy.en.bpe.src -output $DATA_DIR/tag/noisy.en.bpe.src -tag '<FT_s>'
# python $TOOL_DIR/add_tag.py -input ../noisy.en.bpe.tgt -output $DATA_DIR/tag/noisy.en.bpe.tgt -tag '<BT_noisy>'
# python $TOOL_DIR/add_tag.py -input ../BT.en-fr.en -output $DATA_DIR/tag/BT.en-fr.en -tag '<BT_clean>'

# python $TOOL_DIR/add_tag.py -input $DATA_DIR/NFR/nfr.fr-en.en.bpe -output $DATA_DIR/tag/nfr.fr-en.en.bpe -tag '<NFR_s_rev>'
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/NFR/nfr.fr-en.mono.en.bpe -output $DATA_DIR/tag/nfr.fr-en.mono.en.bpe -tag '<NFR_mono_rev>'

# python $TOOL_DIR/add_tag.py -input $DATA_DIR/NFR/nfr.en-fr.en.bpe -output $DATA_DIR/tag/nfr.en-fr.en.bpe -tag '<NFR_s>'
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/NFR/nfr.en-fr.mono.en.bpe -output $DATA_DIR/tag/nfr.en-fr.mono.en.bpe -tag '<NFR_mono>'

# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/asr.en-fr.en -tag '<asr_s>' -output $DATA_DIR/tag/asr.en-fr.en
# python $TOOL_DIR/add_tag.py -input $DATA_DIR/bpe/mustc.en -tag '<mustc_s>' -output $DATA_DIR/tag/mustc.en

# cat $DATA_DIR/tag/train.bpe.en $DATA_DIR/tag/BT.en-fr.en > train.clean.en
# cat $DATA_DIR/bpe/train.bpe.fr ../BT.en-fr.fr > train.clean.fr

# python ../tools/shuffle.py -src train.clean.en -tgt train.clean.fr

# cat train.clean.en $DATA_DIR/tag/finetune.fr-en.bpe.en $DATA_DIR/tag/finetune.en-fr.bpe.en $DATA_DIR/tag/finetune.iwslt.en-fr.bpe.en $DATA_DIR/tag/noisy.en.bpe.src \
#     $DATA_DIR/tag/noisy.en.bpe.tgt $DATA_DIR/tag/nfr.fr-en.en.bpe $DATA_DIR/tag/nfr.fr-en.mono.en.bpe $DATA_DIR/tag/nfr.en-fr.en.bpe $DATA_DIR/tag/nfr.en-fr.mono.en.bpe \
#     $DATA_DIR/tag/asr.en-fr.en $DATA_DIR/tag/mustc.en > mix.en

# cat train.clean.fr $DATA_DIR/bpe/finetune.fr-en.bpe.fr $DATA_DIR/bpe/finetune.en-fr.bpe.fr $DATA_DIR/bpe/finetune.iwslt.en-fr.bpe.fr $DATA_DIR/bpe/noisy.fr.bpe.tgt \
#     $DATA_DIR/bpe/noisy.fr.bpe.src $DATA_DIR/NFR/nfr.fr-en.fr.bpe $DATA_DIR/NFR/nfr.fr-en.mono.fr.bpe $DATA_DIR/NFR/nfr.en-fr.fr.bpe $DATA_DIR/NFR/nfr.en-fr.mono.fr.bpe \
#     $DATA_DIR/bpe/asr.en-fr.fr $DATA_DIR/bpe/mustc.fr > mix.fr



# building vocabulary
# onmt-build-vocab --save_vocab onmt/train.vocab.en --size 50000 mix.en
# onmt-build-vocab --save_vocab onmt/train.vocab.fr --size 50000 mix.fr

# python ${ONMT_DIR}/preprocess.py -train_src train.clean.en \
#                                  -train_tgt train.clean.fr \
#                                  -valid_src $DATA_DIR/tag/valid.bpe.en \
#                                  -valid_tgt $DATA_DIR/bpe/valid.bpe.fr \
#                                  -src_vocab_size 50100 \
#                                  -tgt_vocab_size 50100 \
#                                  -save_data onmt/${NAME} \
#                                  -src_vocab onmt/train.vocab.en \
#                                  -tgt_vocab onmt/train.vocab.fr \
#                                  -src_seq_length 85 \
#                                  -tgt_seq_length 85 \
#                                  -seed 1234

# training (reduce batch size because of GPU limit)
CUDA_VISIBLE_DEVICES=0,1
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
                            -batch_size 6000 \
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
			    			-world_size 2 \
			    			-gpu_ranks 0 1 \
				-train_from models/tag-en-fr_step_30000.pt
