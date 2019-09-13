import os
import argparse
import re
from tqdm import tqdm


def replace_punc(path):
    with open(path, 'r') as file:
        sentences = file.readlines()

    results = []
    for sentence in tqdm(sentences):
        sentence = sentence.replace("'", "’")

        left = True
        while(sentence.find("\"") != -1):
            index = sentence.find("\"")
            if left:
                sentence = sentence.replace('\"','«', 1)
            else:
                sentence = sentence.replace('\"', '»', 1)
            left = not left
        results.append(sentence)
    
    with open(path, 'w') as file:
        file.writelines(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="replace punctuation for MTNT2019 test output")
    parser.add_argument("-f", help="input file", required=True)
    args = parser.parse_args()
    replace_punc(args.f)
