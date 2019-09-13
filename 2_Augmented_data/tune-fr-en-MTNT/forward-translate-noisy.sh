ONMT_DIR=../../tools/OpenNMT-py
NAME=tuneMTNT-fr-en
DATA_DIR=../data/bpe
TOOL_DIR=../../tools
RAW_DATA_DIR=../../data

# tokenize monolingual data
echo "tokenizing monolingual data"
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/monolingual/train.fr > ../data/tok/noisy.fr.tok
echo "applying BPE"
python ${TOOL_DIR}/apply_bpe.py -i ../data/tok/noisy.fr.tok -c ${DATA_DIR}/fr.bpe.50k -o ../data/bpe/noisy.mono.fr.bpe

# filter out sentences longer than 70
awk 'NF<=70' ../data/bpe/noisy.mono.fr.bpe > ../data/bpe/noisy.fr.bpe.src


CHECKPOINT=155500
# translate


python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src ../data/bpe/noisy.fr.bpe.src \
                                -output ../data/bpe/noisy.en.bpe.tgt \
                                -seed 1234 \
                                -beam_size 5 \
                                -gpu 0 \
                                                                -fp32 \
                                -batch_size 256