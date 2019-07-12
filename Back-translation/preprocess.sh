DATA_DIR=data
TOOL_DIR=../tools
RAW_DATA_DIR=../data
mkdir -p ${DATA_DIR}/tok ${DATA_DIR}/bpe

echo "--------start tokenization----------"
# WMT data
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/train/train.large.en > ${DATA_DIR}/tok/train.large.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/train/train.large.fr > ${DATA_DIR}/tok/train.large.tok.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/valid/dev.en > ${DATA_DIR}/tok/valid.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/valid/dev.fr > ${DATA_DIR}/tok/valid.tok.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/test/newstest2014.en > ${DATA_DIR}/tok/test.tok.news2014.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/test/newstest2014.fr > ${DATA_DIR}/tok/test.tok.news2014.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/test/newsdiscusstest2015.en > ${DATA_DIR}/tok/test.tok.newsdiscuss2015.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/test/newsdiscusstest2015.fr > ${DATA_DIR}/tok/test.tok.newsdiscuss2015.fr
# MTNT data
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/train/train.en > ${DATA_DIR}/tok/finetune.train.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/train/train.fr > ${DATA_DIR}/tok/finetune.train.tok.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/valid/valid.en > ${DATA_DIR}/tok/finetune.valid.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/valid/valid.fr > ${DATA_DIR}/tok/finetune.valid.tok.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/test/test.fr-en.en > ${DATA_DIR}/tok/test.fr-en.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/test/test.fr-en.fr > ${DATA_DIR}/tok/test.fr-en.tok.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/test/test.en-fr.en > ${DATA_DIR}/tok/test.en-fr.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/test/test.en-fr.fr > ${DATA_DIR}/tok/test.en-fr.tok.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.fr-en.en > ${DATA_DIR}/tok/test.fr-en.tok.MTNT2019.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.fr-en.fr > ${DATA_DIR}/tok/test.fr-en.tok.MTNT2019.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.en-fr.en > ${DATA_DIR}/tok/test.en-fr.tok.MTNT2019.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/test/MTNT2019.en-fr.fr > ${DATA_DIR}/tok/test.en-fr.tok.MTNT2019.fr
#IWSLT data
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/train/iwslt.fr-en.en.txt > ${DATA_DIR}/tok/finetune.iwslt.fr-en.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/train/iwslt.fr-en.fr.txt > ${DATA_DIR}/tok/finetune.iwslt.fr-en.tok.fr
perl ${TOOL_DIR}/tokenizer.perl -l en < ${RAW_DATA_DIR}/fine-tune/train/iwslt.en-fr.en.txt > ${DATA_DIR}/tok/finetune.iwslt.en-fr.tok.en
perl ${TOOL_DIR}/tokenizer.perl -l fr < ${RAW_DATA_DIR}/fine-tune/train/iwslt.en-fr.fr.txt > ${DATA_DIR}/tok/finetune.iwslt.en-fr.tok.fr
echo "--------finish tokenization----------"


echo "--------start learning byte pair encoding----------"
python ${TOOL_DIR}/learn_bpe.py -i ${DATA_DIR}/tok/train.large.tok.en -s 50000 -o ${DATA_DIR}/bpe/en.bpe.50k
python ${TOOL_DIR}/learn_bpe.py -i ${DATA_DIR}/tok/train.large.tok.fr -s 50000 -o ${DATA_DIR}/bpe/fr.bpe.50k
echo "--------finish learning byte pair encoding----------"
echo "--------start applying byte pair encoding----------"
# apply BPE on all English files
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/train.large.tok.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/train.bpe.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/valid.tok.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/valid.bpe.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.tok.news2014.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/test.bpe.news2014.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.tok.newsdiscuss2015.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/test.bpe.newsdiscuss2015.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/finetune.train.tok.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/finetune.train.bpe.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/finetune.valid.tok.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/finetune.valid.bpe.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.fr-en.tok.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/test.fr-en.bpe.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.en-fr.tok.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/test.en-fr.bpe.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.fr-en.tok.MTNT2019.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/test.fr-en.bpe.MTNT2019.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.en-fr.tok.MTNT2019.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/test.en-fr.bpe.MTNT2019.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/finetune.iwslt.fr-en.tok.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/finetune.iwslt.fr-en.bpe.en
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/finetune.iwslt.en-fr.tok.en -c ${DATA_DIR}/bpe/en.bpe.50k -o ${DATA_DIR}/bpe/finetune.iwslt.en-fr.bpe.en
# apply BPE on all French files
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/train.large.tok.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/train.bpe.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/valid.tok.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/valid.bpe.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.tok.news2014.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/test.bpe.news2014.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.tok.newsdiscuss2015.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/test.bpe.newsdiscuss2015.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/finetune.train.tok.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/finetune.train.bpe.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/finetune.valid.tok.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/finetune.valid.bpe.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.fr-en.tok.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/test.fr-en.bpe.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.en-fr.tok.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/test.en-fr.bpe.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.fr-en.tok.MTNT2019.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/test.fr-en.bpe.MTNT2019.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/test.en-fr.tok.MTNT2019.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/test.en-fr.bpe.MTNT2019.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/finetune.iwslt.fr-en.tok.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/finetune.iwslt.fr-en.bpe.fr
python ${TOOL_DIR}/apply_bpe.py -i ${DATA_DIR}/tok/finetune.iwslt.en-fr.tok.fr -c ${DATA_DIR}/bpe/fr.bpe.50k -o ${DATA_DIR}/bpe/finetune.iwslt.en-fr.bpe.fr

echo "--------finish applying byte pair encoding----------"

