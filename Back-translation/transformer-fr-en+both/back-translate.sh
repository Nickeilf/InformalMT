ONMT_DIR=../../tools/OpenNMT-py
NAME=tuneboth-fr-en
DATA_DIR=../data/bpe
TOOL_DIR=../../tools
RAW_DATA_DIR=../../data

# tokenize monolingual data
echo "tokenizing monolingual data"
# perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/monolingual/fr.txt > ../data/translate/src.fr.tok
echo "applying BPE"
# python ${TOOL_DIR}/apply_bpe.py -i ../data/translate/src.fr.tok -c ${DATA_DIR}/fr.bpe.50k -o ../data/translate/src.fr.bpe

# filter out sentences longer than 70
# awk 'NF<=70' ../data/translate/src.fr.bpe > ../data/translate/src.fr.bpe.filter

#mkdir ../data/translate/fr
#mkdir ../data/translate/trans.fr

#split -d -l 200000 -a 4 ../data/translate/src.fr.bpe.filter ../data/translate/fr/fr

CHECKPOINT=155900
# translate


for number in {0..206}
do
printf -v num "%04d"  "${number}"
file="../data/translate/fr/"fr$num
echo "translating $file"
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src $file \
                                -output ../data/translate/trans.fr/trans.fr$num \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
                                                                -fp32 \
                                -batch_size 256
done

