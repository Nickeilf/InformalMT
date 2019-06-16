mkdir -p data/train data/valid data/test
mkdir -p data/fine-tune/train data/fine-tune/valid data/fine-tune/test

# download clean data
wget https://github.com/pmichel31415/mtnt/releases/download/v1.1/clean-data-en-fr.tar.gz
tar -xvzf clean-data-en-fr.tar.gz
python shuffle.py -src train.en -tgt train.fr
mv train.* data/train
mv dev.* data/valid
mv news* data/test
rm clean-data-en-fr.tar.gz

# download MTNT dataset
wget https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz
tar -xvzf MTNT.1.1.tar.gz MTNT/train MTNT/valid MTNT/test MTNT/split_tsv.sh
cd MTNT
sh split_tsv.sh
mv train/train.en-fr.* ../data/fine-tune/train
mv train/train.fr-en.* ../data/fine-tune/train
mv valid/valid.en-fr.* ../data/fine-tune/valid
mv valid/valid.fr-en.* ../data/fine-tune/valid
mv test/test.en-fr.* ../data/fine-tune/test
mv test/test.fr-en.* ../data/fine-tune/test
cd ..
rm -rf MTNT
rm MTNT.1.1.tar.gz

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



# we use OpenNMT-py for training so you have to clone the repository
# uncomment the following lines if you it is not installed


