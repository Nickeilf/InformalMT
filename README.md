# InformalMT
Comparison on approaches to Neural Machine Translation on informal languages

### Data Preparation
You may run this script and it will download data needed automatically.

```bash prepare_data.sh```

For baseline model, we use clean data [here](https://github.com/pmichel31415/mtnt/releases/download/v1.1/clean-data-en-fr.tar.gz) for initial training. The [MTNT](https://github.com/pmichel31415/mtnt) corpus is used for fine-tuning. We use [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) for training. The clone and install of this toolkit is included in the script.


### Preprocessing
The preprocessing include tokenization with Moses tokenizer.perl along with [BPE](https://github.com/rsennrich/subword-nmt).

### Models
The baseline model follows the same parameter setting as did in [MTNT: A Testbed for Machine Translation of Noisy Text](http://www.cs.cmu.edu/~pmichel1/hosting/mtnt-emnlp.pdf) except that we change sentence batch into token batch.

The `runs/` folder contains different models. You may replicate the result with `bash pipeline.sh`, the scripts includes all necessary steps (tokenization, BPE, building vocabulary, training).

### Evaluation