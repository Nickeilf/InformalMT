mkdir -p data/train data/valid data/test data/monolingual
mkdir -p data/fine-tune/train data/fine-tune/valid data/fine-tune/test data/fine-tune/monolingual

# download clean data(Europarl+NewsCommentary)
wget https://github.com/pmichel31415/mtnt/releases/download/v1.1/clean-data-en-fr.tar.gz
tar -xvzf clean-data-en-fr.tar.gz
python tools/shuffle.py -src train.en -tgt train.fr
mv train.* data/train
mv dev.* data/valid
mv news* data/test
rm clean-data-en-fr.tar.gz

# CommonCrawl
wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
tar -xvzf training-parallel-commoncrawl.tgz commoncrawl.fr-en.fr commoncrawl.fr-en.en
rm training-parallel-commoncrawl.tgz

# UN corpus
wget http://www.statmt.org/wmt13/training-parallel-un.tgz
tar -xvzf training-parallel-un.tgz un/undoc.2000.fr-en.fr un/undoc.2000.fr-en.en
mv un/* ./
rmdir un
rm training-parallel-un.tgz

# giga-fr-en
wget http://www.statmt.org/wmt10/training-giga-fren.tar
tar xvf training-giga-fren.tar
rm training-giga-fren.tar
gzip -d giga-fren.release2.fixed.en.gz
gzip -d giga-fren.release2.fixed.fr.gz

# clean data
cat commoncrawl.fr-en.en giga-fren.release2.fixed.en undoc.2000.fr-en.en > raw.en
cat commoncrawl.fr-en.fr giga-fren.release2.fixed.fr undoc.2000.fr-en.fr > raw.fr
rm commoncrawl.fr-en.en giga-fren.release2.fixed.en undoc.2000.fr-en.en
rm commoncrawl.fr-en.fr giga-fren.release2.fixed.fr undoc.2000.fr-en.fr
perl tools/clean-corpus-n.perl raw fr en clean 1 70
rm raw.*
mv clean.* data/train/
cat data/train/*.en > data/train/train.large.en
cat data/train/*.fr > data/train/train.large.fr
python tools/shuffle.py -src data/train/train.large.fr -tgt data/train/train.large.en



# download MTNT dataset
wget https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz
tar -xvzf MTNT.1.1.tar.gz MTNT/train MTNT/valid MTNT/test MTNT/monolingual MTNT/split_tsv.sh
cd MTNT
sh split_tsv.sh
mv train/train.en-fr.* ../data/fine-tune/train
mv train/train.fr-en.* ../data/fine-tune/train
mv valid/valid.en-fr.* ../data/fine-tune/valid
mv valid/valid.fr-en.* ../data/fine-tune/valid
mv monolingual/train.* ../data/fine-tune/monolingual
mv test/test.en-fr.* ../data/fine-tune/test
mv test/test.fr-en.* ../data/fine-tune/test
cd ..
rm -rf MTNT
rm MTNT.1.1.tar.gz

cat data/fine-tune/train/train.en-fr.en data/fine-tune/train/train.fr-en.en > data/fine-tune/train/train.en
cat data/fine-tune/train/train.en-fr.fr data/fine-tune/train/train.fr-en.fr > data/fine-tune/train/train.fr
cat data/fine-tune/valid/valid.en-fr.en data/fine-tune/valid/valid.fr-en.en > data/fine-tune/valid/valid.en
cat data/fine-tune/valid/valid.en-fr.fr data/fine-tune/valid/valid.fr-en.fr > data/fine-tune/valid/valid.fr

# download WMT19 Robustness test set
wget http://www.cs.cmu.edu/~pmichel1/hosting/MTNT2019.tar.gz
tar -xvzf MTNT2019.tar.gz
# en -> fr
cut -f3 MTNT2019/en-fr.final.tsv > data/fine-tune/test/MTNT2019.en-fr.en
cut -f4 MTNT2019/en-fr.final.tsv > data/fine-tune/test/MTNT2019.en-fr.fr
# fr -> en
cut -f4 MTNT2019/fr-en.final.tsv > data/fine-tune/test/MTNT2019.fr-en.en
cut -f3 MTNT2019/fr-en.final.tsv > data/fine-tune/test/MTNT2019.fr-en.fr
rm -rf MTNT2019
rm MTNT2019.tar.gz

# download IWSLT2017 data for tuning (fr-en and en-fr)
wget wit3.fbk.eu/archive/2017-01-trnted//texts/fr/en/fr-en.tgz
tar -xvzf fr-en.tgz
cd fr-en

# training data
for train in train.tags.fr-en.fr train.tags.fr-en.en; do
	cat $train | \
	grep -v '<url>' | \
	grep -v '<talkid>' | \
	grep -v '<keywords>' | \
	grep -v '<doc' | \
	grep -v '<speaker>' | \
	grep -v '<reviewer' | \
	grep -v '<translator' | \
	grep -v '<\/doc>' | \
	sed -e 's/<title>//g' | \
	sed -e 's/<\/title>//g' | \
	sed -e 's/<description>//g' | \
	sed -e 's/<\/description>//g' > $train.txt
done


cat *.en.txt > ../data/fine-tune/train/iwslt.fr-en.en.txt
cat *.fr.txt > ../data/fine-tune/train/iwslt.fr-en.fr.txt
cd ..
rm -rf fr-en
rm fr-en.tgz


wget wit3.fbk.eu/archive/2017-01-trnted//texts/en/fr/en-fr.tgz
tar -xvzf en-fr.tgz
# train
cd en-fr
for train in train.tags.en-fr.fr train.tags.en-fr.en; do
	cat $train | \
	grep -v '<url>' | \
	grep -v '<talkid>' | \
	grep -v '<keywords>' | \
	grep -v '<doc' | \
	grep -v '<speaker>' | \
	grep -v '<reviewer' | \
	grep -v '<translator' | \
	grep -v '<\/doc>' | \
	sed -e 's/<title>//g' | \
	sed -e 's/<\/title>//g' | \
	sed -e 's/<description>//g' | \
	sed -e 's/<\/description>//g' > $train.txt
done


cat *.en.txt > ../data/fine-tune/train/iwslt.en-fr.en.txt
cat *.fr.txt > ../data/fine-tune/train/iwslt.en-fr.fr.txt
cd ..
rm -rf en-fr
rm en-fr.tgz

# we use OpenNMT-py for training so you have to clone the repository
# uncomment the following lines if you it is not installed

cd tools
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
pip install -r requirements.txt
cd ../..

