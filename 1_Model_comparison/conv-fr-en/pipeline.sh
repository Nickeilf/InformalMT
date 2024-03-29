#!/bin/bash
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
mkdir ${DATA_DIR}/fairseq
fairseq-preprocess --source-lang fr --target-lang en \
  --trainpref ${DATA_DIR}/train.bpe.16k --validpref ${DATA_DIR}/valid.bpe.16k \
  --thresholdtgt 0 --thresholdsrc 0 --workers 8 --destdir ${DATA_DIR}/fairseq

# training
# add shared vocab
CUDA_VISIBLE_DEVICES=0
fairseq-train ${DATA_DIR}/fairseq \
  --lr 0.5 --clip-norm 0.1 --dropout 0.1 --max-tokens 3000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 --log-format json --skip-invalid-size-inputs-valid-test \
  --arch fconv_wmt_en_fr --save-dir models --max-epoch 50 --max-sentences-valid 5 > convs2s.log
