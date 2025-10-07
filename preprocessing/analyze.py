import csv
import numpy as np
import pandas as pd
from mwtokenizer.tokenizer import Tokenizer

def analyze(corpus_path: str, field_names: list) -> dict:
    """
    calculates the statistics of a parallelized corpus
    args:
        corpus_path: path to parallelized CSV
        field_names: language labels for the CSV
    returns:
        dictionary containing average token number, total token count, and the
        number of unique tokens for each language

    """
    stats = {ln: {'token_count': 0} for ln in field_names}
    unique = {ln: set() for ln in field_names}
    tokenizers = {ln: Tokenizer(language_code=ln) for ln in field_names}
    corpus = pd.read_csv(corpus_path, names=field_names)
    for i, row in corpus.iterrows():
        for ln in field_names:
            tokenized = [x for x in tokenizers[ln].word_tokenize(str(row[ln]))]
            stats[ln]['token_count'] += len(tokenized)
            unique[ln].update(set(tokenized))
        if i % 10000 == 0:
            print(f"{i:07}")
    for ln in field_names:
        stats[ln]['avg'] = np.divide(stats[ln]['token_count'], corpus.index.size)
        stats[ln]['unique'] = len(unique[ln])
        
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.to_latex('corpus_stats.tex', float_format="%.2f")
    
    return stats 

print(analyze('data/sent_speeches_all.csv', ['en', 'es', 'ru', 'zh', 'fr']))
print(analyze('data/europarl/europarl_4way.csv', ['de', 'en', 'es', 'fr', 'pl']))
print(analyze('data/apa-rst_3way.csv', ['A2 sentence', 'B1 sentence', 'OR sentence']))
#print(analyze('data/arendt_2wayforstats.csv', ['de', 'en']))
    