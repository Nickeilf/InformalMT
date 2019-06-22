#!/bin/bash
# building pipeline for baseline on fr-en translation
# train: newscommentary + europarl
# valid: newsdiscussdev2015
# test: newstest2014/newsdiscusstest2015/MTNT-test-fr-en
source path.config

mkdir ${VOCAB_DIR}
mkdir ${DATA_DIR}
mkdir result

# skip preprocessing if already done
if [ "$(ls -A ${VOCAB_DIR})" ]; then
  echo "data already in directory, skip tokenization"
else
  # first tokenize source and target files
  echo "--------start tokenization----------"
  perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/train/train.en > ${VOCAB_DIR}/train.tok.en
  perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/train/train.fr > ${VOCAB_DIR}/train.tok.fr
  perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/valid/dev.en > ${VOCAB_DIR}/valid.tok.en
  perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/valid/dev.fr > ${VOCAB_DIR}/valid.tok.fr
  perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/test/newstest2014.en > ${VOCAB_DIR}/test.tok.news2014.en
  perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/test/newstest2014.fr > ${VOCAB_DIR}/test.tok.news2014.fr
  perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/test/newsdiscusstest2015.en > ${VOCAB_DIR}/test.tok.newsdiscuss2015.en
  perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/test/newsdiscusstest2015.fr > ${VOCAB_DIR}/test.tok.newsdiscuss2015.fr
  perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/test/test.fr-en.en > ${VOCAB_DIR}/test.tok.en
  perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/test/test.fr-en.fr > ${VOCAB_DIR}/test.tok.fr
  perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.fr-en.en > ${VOCAB_DIR}/test.tok.MTNT2019.en
  perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.fr-en.fr > ${VOCAB_DIR}/test.tok.MTNT2019.fr
  echo "--------finish tokenization----------"
fi


# skip if BPE is done
if [ "$(ls -A ${DATA_DIR})" ]; then
  # apply Byte Pair Encoding
  echo "data already in directory, skip BPE"
else
  echo "--------start learning byte pair encoding----------"
  python ${TOOL_DIR}/learn_bpe.py -i ${VOCAB_DIR}/train.tok.en -s 16000 -o ${DATA_DIR}/fr-en.en.bpe.16k
  python ${TOOL_DIR}/learn_bpe.py -i ${VOCAB_DIR}/train.tok.fr -s 16000 -o ${DATA_DIR}/fr-en.fr.bpe.16k
  echo "--------finish learning byte pair encoding----------"
  echo "--------start applying byte pair encoding----------"
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/train.tok.en -c ${DATA_DIR}/fr-en.en.bpe.16k -o ${DATA_DIR}/train.bpe.16k.en
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/valid.tok.en -c ${DATA_DIR}/fr-en.en.bpe.16k -o ${DATA_DIR}/valid.bpe.16k.en
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/test.tok.en -c ${DATA_DIR}/fr-en.en.bpe.16k -o ${DATA_DIR}/test.bpe.16k.en
  python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.en.bpe.16k < ${VOCAB_DIR}/test.tok.news2014.en > ${DATA_DIR}/test.bpe.16k.news2014.en
  python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.en.bpe.16k < ${VOCAB_DIR}/test.tok.newsdiscuss2015.en > ${DATA_DIR}/test.bpe.16k.newsdiscuss2015.en
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/test.tok.MTNT2019.en -c ${DATA_DIR}/fr-en.en.bpe.16k -o ${DATA_DIR}/test.bpe.16k.MTNT2019.en


  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/train.tok.fr -c ${DATA_DIR}/fr-en.fr.bpe.16k -o ${DATA_DIR}/train.bpe.16k.fr
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/valid.tok.fr -c ${DATA_DIR}/fr-en.fr.bpe.16k -o ${DATA_DIR}/valid.bpe.16k.fr
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/test.tok.fr -c ${DATA_DIR}/fr-en.fr.bpe.16k -o ${DATA_DIR}/test.bpe.16k.fr
  python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.fr.bpe.16k < ${VOCAB_DIR}/test.tok.news2014.fr > ${DATA_DIR}/test.bpe.16k.news2014.fr
  python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.fr.bpe.16k < ${VOCAB_DIR}/test.tok.newsdiscuss2015.fr > ${DATA_DIR}/test.bpe.16k.newsdiscuss2015.fr
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/test.tok.MTNT2019.fr -c ${DATA_DIR}/fr-en.fr.bpe.16k -o ${DATA_DIR}/test.bpe.16k.MTNT2019.fr
  echo "--------finish applying byte pair encoding----------"
fi

# building vocabulary
#mkdir ${DATA_DIR}/onmt
#python ${ONMT_DIR}/preprocess.py -train_src ${DATA_DIR}/train.bpe.16k.fr \
#                                 -train_tgt ${DATA_DIR}/train.bpe.16k.en \
#                                 -valid_src ${DATA_DIR}/valid.bpe.16k.fr \
 #                                -valid_tgt ${DATA_DIR}/valid.bpe.16k.en \
  #                               --src_words_min_frequency 1 \
#								 --tgt_words_min_frequency 1 \
 #                                -save_data ${DATA_DIR}/onmt/${NAME} \
  #                               -src_seq_length 70 \
   #                              -tgt_seq_length 70 \
    #                             -seed 1234

# training
CUDA_VISIBLE_DEVICES=0,1
python ${ONMT_DIR}/train.py -word_vec_size 512 \
                            -encoder_type brnn \
                            -decoder_type rnn \
                            -rnn_size 1024 \
                            -layers 2 \
                            -bridge \
                            -global_attention mlp \
                            -data ${DATA_DIR}/onmt/${NAME} \
                            -save_model models/${NAME} \
                            -save_checkpoint_steps 5000 \
                            -batch_size 4096 \
                            -batch_type tokens \
                            -valid_steps 5000 \
                            -train_steps 300000 \
                            -early_stopping 5 \
                            -keep_checkpoint 8 \
                            -optim adam \
                            -dropout 0.3 \
                            -label_smoothing 0.1 \
                            -learning_rate 0.001 \
							-learning_rate_decay 0.7 \
                            -decay_steps 15000 \
                            -start_decay_steps 30000 \
                            -log_file ${NAME}.log \
                            -tensorboard \
                            -tensorboard_log_dir models \
                            -seed 1234 \
							-exp ${NAME} \
			    -world_size 1 \
			    -gpu_ranks 0
