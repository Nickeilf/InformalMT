ONMT_DIR=../../tools/OpenNMT-py
NAME=tuneboth-fr-en
DATA_DIR=../data/bpe
TOOL_DIR=../../tools
RAW_DATA_DIR=../../data

# tokenize monolingual data
echo "tokenizing monolingual data"
# perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/monolingual/fr.txt > ../data/translate/src.fr.tok
echo "applying BPE"
# python ${TOOL_DIR}/apply_bpe.py -i ../data/translate/src.fr.tok -c ${DATA_DIR}/en.bpe.50k -o ../data/translate/src.fr.bpe

CHECKPOINT=155900
# translate

# split -d -l 10000 -a 5 ../data/translate/src.fr.bpe ../data/translate/fr/fr
for file in ../data/translate/fr/*
do
filename="${file##*/}"
echo "translating $filename"
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src $file \
                                -output ../data/translate/trans.fr/trans.$filename \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
								-fp32 \
                                -batch_size 64
done