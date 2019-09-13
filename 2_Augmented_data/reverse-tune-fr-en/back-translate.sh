NAME=reverse-fr-en
ONMT_DIR=../tools/OpenNMT-py

CHECKPOINT=155150
CUDA_VISIBLE_DEVICES=0,1

for number in {0..22}
do
printf -v num "%02d"  "${number}"
file="../bt-data/"BT.en-fr.fr$num
echo "translating $file"
python ${ONMT_DIR}/translate.py -model models/${NAME}_step_${CHECKPOINT}.pt \
                                -src /data/zli/InformalMT/Back-translation/data/bpe/noisy.fr.bpe.src \
                                -output ../noisy.en.bpe.tgt \
                                -seed 1234 \
                                -beam_size 5 \
                                -replace_unk \
                                -fp32 \
                                -gpu 1 \
                                -batch_size 256
done