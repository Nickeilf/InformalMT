#!/bin/bash
source path.config

CUDA_VISIBLE_DEVICES=1
# evaluation on MTNT
cat data/test.bpe.16k.fr | fairseq-interactive data/fairseq --path models/checkpoint_best.pt --beam 5 --remove-bpe --batch-size 30 --buffer-size 50 |tail -n +7| tee result/MTNT.pred.txt
cut -f3 result/MTNT.pred.txt > result/MTNT.txt
sed '/^$/d' result/MTNT.txt > result/MTNT.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l en < result/MTNT.join.txt > result/MTNT.detok.txt


# evaluation on newstest2014 
cat data/test.bpe.16k.news2014.fr | fairseq-interactive data/fairseq --path models/checkpoint_best.pt --beam 5 --remove-bpe --batch-size 30 --buffer-size 50 |tail -n +7| tee result/news2014.pred.txt
cut -f3 result/news2014.pred.txt > result/news2014.txt
sed '/^$/d' result/news2014.txt > result/news2014.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l en < result/news2014.join.txt > result/news2014.detok.txt


# evaluation on newsdiscusstest2015
cat data/test.bpe.16k.newsdiscuss2015.fr | fairseq-interactive data/fairseq --path models/checkpoint_best.pt --beam 5 --remove-bpe --batch-size 30 --buffer-size 50 |tail -n +7| tee result/newsdiscuss2015.pred.txt
cut -f3 result/newsdiscuss2015.pred.txt > result/newsdiscuss2015.txt
sed '/^$/d' result/newsdiscuss2015.txt > result/newsdiscuss2015.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l en < result/newsdiscuss2015.join.txt > result/newsdiscuss2015.detok.txt

# evaluation on MTNT2019 test
cat data/test.bpe.16k.MTNT2019.fr | fairseq-interactive data/fairseq --path models/checkpoint_best.pt --beam 5 --remove-bpe --batch-size 30 --buffer-size 50 |tail -n +7| tee result/MTNT2019.pred.txt
cut -f3 result/MTNT2019.pred.txt > result/MTNT2019.txt
sed '/^$/d' result/MTNT2019.txt > result/MTNT2019.join.txt
perl ${TOOL_DIR}/detokenizer.perl -l en < result/MTNT2019.join.txt > result/MTNT2019.detok.txt

echo "BLEU score on MTNT test set"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/fine-tune/test/test.fr-en.en < result/MTNT.detok.txt
echo "BLEU score on MTNT2019 test set"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.fr-en.en < result/MTNT2019.detok.txt
echo "BLEU score on newstest2014"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/test/newstest2014.en < result/news2014.detok.txt
echo "BLEU score on newsdiscusstest2015"
perl ${TOOL_DIR}/multi-bleu-detok.perl ${RAW_DATA_DIR}/test/newsdiscusstest2015.en < result/newsdiscuss2015.detok.txt

