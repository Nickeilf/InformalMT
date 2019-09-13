#!/bin/bash
ONMT_DIR=../tools/OpenNMT-py
NAME=tagtune_external-en-fr
DATA_DIR=/data/zli/InformalMT/Back-translation/data
TOOL_DIR=../tools
RAW_DATA_DIR=../data

CUDA_VISIBLE_DEVICES=2
CHECKPOINT=136500
# evaluation on MTNT
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src ${DATA_DIR}/tag/test.en-fr.bpe.en \
                                -tgt ${DATA_DIR}/bpe/test.en-fr.bpe.fr \
                                -output result/MTNT.pred.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32
cat result/MTNT.pred.txt | sed -E 's/(@@ )|(@@ ?$)//g' > result/MTNT.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l fr < result/MTNT.join.txt > result/MTNT.detok.txt


# evaluation on newstest2014
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src ${DATA_DIR}/tag/test.bpe.news2014.en \
                                -tgt ${DATA_DIR}/bpe/test.bpe.news2014.fr \
                                -output result/news2014.pred.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32
cat result/news2014.pred.txt | sed -E 's/(@@ )|(@@ ?$)//g' > result/news2014.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l fr < result/news2014.join.txt > result/news2014.detok.txt


# evaluation on newsdiscusstest2015
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src $DATA_DIR/tag/test.bpe.newsdiscuss2015.en \
                                -tgt ${DATA_DIR}/bpe/test.bpe.newsdiscuss2015.fr \
                                -output result/newsdiscuss2015.pred.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32
cat result/newsdiscuss2015.pred.txt | sed -E 's/(@@ )|(@@ ?$)//g' > result/newsdiscuss2015.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l fr < result/newsdiscuss2015.join.txt > result/newsdiscuss2015.detok.txt


# evaluation on MTNT2019 test
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src ${DATA_DIR}/tag/test.en-fr.bpe.MTNT2019.en \
                                -tgt ${DATA_DIR}/bpe/test.en-fr.bpe.MTNT2019.fr \
                                -output result/MTNT2019.pred.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32
cat result/MTNT2019.pred.txt | sed -E 's/(@@ )|(@@ ?$)//g' > result/MTNT2019.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l fr < result/MTNT2019.join.txt > result/MTNT2019.detok.txt
python $TOOL_DIR/fix_punc.py -f result/MTNT2019.detok.txt

echo "BLEU score on MTNT test set"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/fine-tune/test/test.en-fr.fr < result/MTNT.detok.txt
echo "BLEU score on MTNT2019 test set"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.en-fr.fr < result/MTNT2019.detok.txt
echo "BLEU score on newstest2014"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/test/newstest2014.fr < result/news2014.detok.txt
echo "BLEU score on newsdiscusstest2015"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/test/newsdiscusstest2015.fr < result/newsdiscuss2015.detok.txt

