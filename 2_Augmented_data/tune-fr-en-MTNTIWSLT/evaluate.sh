#!/bin/bash
ONMT_DIR=../../tools/OpenNMT-py
NAME=tuneboth-fr-en
DATA_DIR=../data/bpe
TOOL_DIR=../../tools
RAW_DATA_DIR=../../data

CHECKPOINT=155900
# evaluation on MTNT
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src ${DATA_DIR}/test.fr-en.bpe.fr \
                                -tgt ${DATA_DIR}/test.fr-en.bpe.en \
                                -output result/MTNT.pred.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32
cat result/MTNT.pred.txt | sed -E 's/(@@ )|(@@ ?$)//g' > result/MTNT.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l en < result/MTNT.join.txt > result/MTNT.detok.txt


# evaluation on newstest2014
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src ${DATA_DIR}/test.bpe.news2014.fr \
                                -tgt ${DATA_DIR}/test.bpe.news2014.en \
                                -output result/news2014.pred.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32
cat result/news2014.pred.txt | sed -E 's/(@@ )|(@@ ?$)//g' > result/news2014.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l en < result/news2014.join.txt > result/news2014.detok.txt


# evaluation on newsdiscusstest2015
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src ${DATA_DIR}/test.bpe.newsdiscuss2015.fr \
                                -tgt ${DATA_DIR}/test.bpe.newsdiscuss2015.en \
                                -output result/newsdiscuss2015.pred.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32
cat result/newsdiscuss2015.pred.txt | sed -E 's/(@@ )|(@@ ?$)//g' > result/newsdiscuss2015.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l en < result/newsdiscuss2015.join.txt > result/newsdiscuss2015.detok.txt


# evaluation on MTNT2019 test
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src ${DATA_DIR}/test.fr-en.bpe.MTNT2019.fr \
                                -tgt ${DATA_DIR}/test.fr-en.bpe.MTNT2019.en \
                                -output result/MTNT2019.pred.txt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32
cat result/MTNT2019.pred.txt | sed -E 's/(@@ )|(@@ ?$)//g' > result/MTNT2019.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l en < result/MTNT2019.join.txt > result/MTNT2019.detok.txt

echo "BLEU score on MTNT test set"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/fine-tune/test/test.fr-en.en < result/MTNT.detok.txt
echo "BLEU score on MTNT2019 test set"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.fr-en.en < result/MTNT2019.detok.txt
echo "BLEU score on newstest2014"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/test/newstest2014.en < result/news2014.detok.txt
echo "BLEU score on newsdiscusstest2015"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/test/newsdiscusstest2015.en < result/newsdiscuss2015.detok.txt

