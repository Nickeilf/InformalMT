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


# skip if segmentation is done
if [ "$(ls -A ${DATA_DIR})" ]; then
  # apply character segmentation
  echo "data already in directory, skip segmentation"
else
  echo "--------start character segmentation----------"
  python char_split.py --input ${VOCAB_DIR}/train.tok.fr --output ${DATA_DIR}/train.char.fr
  python char_split.py --input ${VOCAB_DIR}/valid.tok.fr --output ${DATA_DIR}/valid.char.fr
  python char_split.py --input ${VOCAB_DIR}/test.tok.fr --output ${DATA_DIR}/test.char.fr
  python char_split.py --input ${VOCAB_DIR}/test.tok.MTNT2019.fr --output ${DATA_DIR}/test.char.MTNT2019.fr
  python char_split.py --input ${VOCAB_DIR}/test.tok.news2014.fr --output ${DATA_DIR}/test.char.news2014.fr
  python char_split.py --input ${VOCAB_DIR}/test.tok.newsdiscuss2015.fr --output ${DATA_DIR}/test.char.newsdiscuss2015.fr
  echo "--------finish character segmentation----------"
	
  python ${TOOL_DIR}/learn_bpe.py -i ${VOCAB_DIR}/train.tok.en -s 16000 -o ${DATA_DIR}/fr-en.en.bpe.16k

  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/train.tok.en -c ${DATA_DIR}/fr-en.en.bpe.16k -o ${DATA_DIR}/train.bpe.16k.en
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/valid.tok.en -c ${DATA_DIR}/fr-en.en.bpe.16k -o ${DATA_DIR}/valid.bpe.16k.en
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/test.tok.en -c ${DATA_DIR}/fr-en.en.bpe.16k -o ${DATA_DIR}/test.bpe.16k.en
  python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.en.bpe.16k < ${VOCAB_DIR}/test.tok.news2014.en > ${DATA_DIR}/test.bpe.16k.news2014.en
  python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.en.bpe.16k < ${VOCAB_DIR}/test.tok.newsdiscuss2015.en > ${DATA_DIR}/test.bpe.16k.newsdiscuss2015.en
  python ${TOOL_DIR}/apply_bpe.py -i ${VOCAB_DIR}/test.tok.MTNT2019.en -c ${DATA_DIR}/fr-en.en.bpe.16k -o ${DATA_DIR}/test.bpe.16k.MTNT2019.en

  
fi

# building vocabulary
#mkdir ${DATA_DIR}/sockeye
#python -m sockeye.prepare_data -s ${DATA_DIR}/train.char.fr \
#							   -t ${DATA_DIR}/train.bpe.16k.en \
#							   --max-seq-len 400:70 \
#							   -o ${DATA_DIR}/sockeye
							   


# training
CUDA_VISIBLE_DEVICES=1
# same as char2char( for CNN embedding setting)
python -m sockeye.train -d ${DATA_DIR}/sockeye \
						-o models \
						-vs ${DATA_DIR}/valid.char.fr \
						-vt ${DATA_DIR}/valid.bpe.16k.en \
						--encoder rnn-with-conv-embed \
						--decoder rnn \
						--num-layers 2:2 \
						--rnn-num-hidden 1024 \
						--conv-embed-output-dim 128 \
						--rnn-cell-type lnlstm \
						--rnn-residual-connections \
						--layer-normalization \
						--rnn-decoder-hidden-dropout 0.3 \
						--batch-size 40 \
						--batch-type sentence \
						--metrics perplexity accuracy \
						--label-smoothing 0.1 \
						--max-num-checkpoint-not-improved 12 \
						--conv-embed-dropout 0.1 \
						--optimizer adam \
						--initial-learning-rate 0.001 \
						--learning-rate-reduce-factor 0.7 \
						--checkpoint-interval 5000 \
						--learning-rate-reduce-num-not-improved 3 \
						--decode-and-evaluate 0 \
						--keep-last-params 15
						
