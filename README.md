# InformalMT
Project for my master thesis "Improving Nerual Machine Translation Robustness via Data Augmentation"

We experimented with data augmentation methods (Back-translation, forward-translation, fuzzy match) and external datasets from speech transcripts to improve the neural machine translation model's performance on noisy test sets. We followed the [WMT19 Robustness Shared Task](http://www.statmt.org/wmt19/robustness.html) in Fr-En directions.

The training and preprocessing scripts for all systems are provided in this repository.

### Tools used
```
OpenNMT-py
PyTorch
fairseq
```

### Data Preparation
You may run this script and it will download data needed automatically.

```bash prepare_data.sh```

Datasets used in the experiments can be catogorized as in-domain and out-of-domain. The in-domain data is [MTNT](https://github.com/pmichel31415/mtnt) dataset. For out-of-domain data, we use WMT15 fr-en News Translation data.

### Preprocessing
The preprocessing include tokenization with Moses tokenizer.perl along with [BPE](https://github.com/rsennrich/subword-nmt).

### Experiments
We conducted 4 experiments, namely:
1. Model comparison (RNN, CNN and Transformer) on noisy texts
2. Data agumentation (back-translation, forward-translation, fuzzy match)
3. External data (human transcripts from IWSLT and MuST-C, ASR generated transcripts)
4. Submissions to WMT19 Leaderboard

### Citation
Details about the experiments and results can be found here (TODO: add thesis link)

