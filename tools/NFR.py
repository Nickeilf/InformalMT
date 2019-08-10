from SetSimilaritySearch import SearchIndex
from SetSimilaritySearch import all_pairs
import editdistance
import argparse
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter


def find_fuzzy_matches(src_path, tgt_path, threshold, n_best, output_src, output_tgt):
    with open(src_path, 'r') as file:
        src = file.readlines()

    with open(tgt_path, 'r') as file:
        tgt = file.readlines()
    
    set = []
    for line in src:
        toks = line.split()
        set.append(toks)
    
    pairs = all_pairs(set, similarity_func_name="containment_min",
                  similarity_threshold=threshold)
    
    tmp = list(pairs)
    tmp = sorted(tmp, key=lambda x: x[0])
    
    
    new_list = []
    for k, g in groupby(tmp, itemgetter(0)):
        new_list += sorted(list(g), key=lambda x: x[2], reverse=True)[:n_best]

    matches = []
    references = []
    
    for index1, index2, similarity in tqdm(new_list):
        sent1 = set[index1]
        sent2 = set[index2]
        score = editdistance.eval(sent1, sent2)
        
        min_length = min(len(sent1), len(sent2))
        score_threshold = min_length * (1-threshold)
        if score < score_threshold and src[index1] != src[index2] and tgt[index1] != tgt[index2]:
            matches.append(src[index1])
            references.append(tgt[index2])
    
    with open(output_src, 'w') as file:
        file.writelines(matches)
    
    with open(output_tgt, 'w') as file:
        file.writelines(references)



# find fuzzy matches from extra monolingual data
def find_fuzzy_matches_from_external(src_path, tgt_path, external_path, threshold, n_best, output_src, output_tgt):
    with open(src_path, 'r') as file:
        src = file.readlines()

    with open(tgt_path, 'r') as file:
        tgt = file.readlines()

    with open(external_path, 'r') as file:
        external = file.readlines()

    set = []
    for line in src:
        toks = line.split()
        set.append(toks)

    match_sets = SearchIndex(set, similarity_func_name="containment_min",
                        similarity_threshold=threshold)
    
    matches = []
    references = []
    for sentence in tqdm(external):
        tokens = sentence.split()
        result = match_sets.query(tokens)
        
        result = sorted(result, key=lambda x: x[1], reverse=True)[:n_best]
        for index, _ in result:
            src_tok = set[index]
            candidate_tok = sentence.split()

            score = editdistance.eval(candidate_tok, src_tok)
            min_length = min(len(src_tok), len(candidate_tok))
            score_threshold = min_length * (1-threshold)
            

            if score < score_threshold and src[index] != sentence and min_length > 5:
                matches.append(sentence)
                references.append(tgt[index])

    with open(output_src, 'w') as file:
        file.writelines(matches)

    with open(output_tgt, 'w') as file:
        file.writelines(references)
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Script for finding fuzzy matches in parallel data (or from external monolingual data)")
    parser.add_argument("-s", help="source parallel data", required=True)
    parser.add_argument("-t", help="target parallel data", required=True)
    parser.add_argument("-os", help="output source file path", required=True)
    parser.add_argument("-ot", help="output target file path", required=True)
    parser.add_argument("-e", help="external corpus on source language (if this is not provided, matches will be found inside source parallel data)")
    parser.add_argument("-l", type=float, default=0.5, help="similarity_threshold, larger lambda means higher quality")
    parser.add_argument("-n", type=int, default=10, help="only select n_best match candidates from SetSimilaritySearch, use to boost searching")
    args = parser.parse_args()

    assert args.l > 0 and args.l < 1

    if args.e is None:
        find_fuzzy_matches(args.s, args.t, args.l, args.n, args.os, args.ot)
    else:
        find_fuzzy_matches_from_external(args.s, args.t, args.e, args.l, args.n, args.os, args.ot)

