#!/bin/bash
# building pipeline for baseline on fr-en translation
# train: newscommentary + europarl
# valid: newsdiscussdev2015
# test: newstest2014/newsdiscusstest2015/MTNT-test-fr-en
source path.config

mkdir ${VOCAB_DIR}
mkdir ${DATA_DIR}
mkdir result

# cp -r ../base-fr-en+/data ./
# cp -r ../base-fr-en+/models ./

# perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/train/train.fr-en.en > ${VOCAB_DIR}/train.finetune.tok.en
# perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/train/train.fr-en.fr > ${VOCAB_DIR}/train.finetune.tok.fr
# perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/valid/valid.fr-en.en > ${VOCAB_DIR}/valid.finetune.tok.en
# perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/valid/valid.fr-en.fr > ${VOCAB_DIR}/valid.finetune.tok.fr

# python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.en.bpe.16k < ${VOCAB_DIR}/train.finetune.tok.en > ${DATA_DIR}/train.finetune.bpe.16k.en
# python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.en.bpe.16k < ${VOCAB_DIR}/valid.finetune.tok.en > ${DATA_DIR}/valid.finetune.bpe.16k.en
# python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.fr.bpe.16k < ${VOCAB_DIR}/train.finetune.tok.fr > ${DATA_DIR}/train.finetune.bpe.16k.fr
# python ${TOOL_DIR}/apply_bpe.py -c ${DATA_DIR}/fr-en.fr.bpe.16k < ${VOCAB_DIR}/valid.finetune.tok.fr > ${DATA_DIR}/valid.finetune.bpe.16k.fr

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
mkdir ${DATA_DIR}/sockeye
python -m sockeye.prepare_data -s ${DATA_DIR}/train.finetune.bpe.16k.fr \
   					 		   -t ${DATA_DIR}/train.finetune.bpe.16k.en \
   					 		   --num-words 16000:16000 \
 			  				   --max-seq-len 70:70 \
  							   --num-samples-per-shard 1000000 \
  							   --seed 13 \
							   --source-vocab data/sockeye/vocab.src.0.json \
							   --target-vocab data/sockeye/vocab.trg.0.json \
  					 		   -o data/sockeye-tune


# training
python -m sockeye.train -d data/sockeye-tune \
                        -vs ${DATA_DIR}/valid.finetune.bpe.16k.fr \
                        -vt ${DATA_DIR}/valid.finetune.bpe.16k.en \
                        -o models-tune \
                        --encoder rnn \
                        --decoder rnn \
                        --rnn-cell-type lnlstm \
						--layer-normalization \
						--rnn-residual-connections \
                        --num-layers 4:2 \
                        --rnn-num-hidden 512 \
                        --rnn-decoder-hidden-dropout 0.3 \
                        --num-embed 512 \
                        --rnn-attention-type mlp \
                        --batch-size 4096 \
                        --batch-type word \
                        --label-smoothing 0.1 \
                        --metrics perplexity accuracy \
                        --checkpoint-interval 500 \
                        --max-num-checkpoint-not-improved 5 \
                        --max-num-epochs 50 \
                        --optimizer adam \
                        --initial-learning-rate 0.0002 \
                        --learning-rate-reduce-factor 0.5 \
                        --learning-rate-reduce-num-not-improved 2 \
                        --decode-and-evaluate 0 \
                        --seed 13 \
						--params models/params.best \
                        --keep-last-params 9
