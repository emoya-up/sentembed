import pandas as pd
import torch.nn as nn

## adequate versions of cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CosineEmbeddingLoss

# internal
from .context import Context

dummy_df = pd.DataFrame({
    'esperanto' : ['spam', 'eggs', 'spam', 'eggs'] * 6,
    'latin' : ['alpha', 'beta', 'gamma'] * 8})

## Jaccard metric (mostly pairwise)
def jaccard_metric(context: Context,
                   data: pd.DataFrame,
                   decision_func=cosine_similarity,
                   threshold=.5,
                   yields=False):
    # TODO embed the text segments with variable encoders
    # TODO enable comparison of one or more
    '''
    This Jaccard metric measures the distance of n sets.
    The function reports the average of a binary decision on all rows
    in a pd.DataFrame.
    args:
        data: pd.DataFrame with source/target language features and rows of
        text segments
        threshold: binary decision threshold
    returns (or yields):
        dict with averages of the binary Jaccard
    '''
    cols = data.columns
    seen_pairs = {}
    
     # fixed-schedule pair-processing
    for ln1 in cols:
        avg_sim = 0
        for ln2 in cols:
            if f'{ln1, ln2}' not in seen_pairs and ln1 != ln2:
                # apply the decision function to two text segments
                try:
                    pairwise_cos = data[[ln1, ln2]].apply(
                        lambda x: decision_func(x),
                        axis=0,
                        raw=True)
                except TypeError as e:
                    raise NotImplementedError(
                        'Encoder retrieval not yet implemented')
                
                
                # register the language pair as seen
                seen_pairs[f'{ln1, ln2}'] = avg_sim
        
    print(seen_pairs)
    
    if yields:
        #TODO implement mode that yields the similarities in a loop
        # this allows for a pairwise jaccard metric
        pass
    else:
        return seen_pairs

# print output (formats)
if __name__ == "__main__":
    import sys
    
    print(dummy_df.head())
    pairwise_dict = jaccard_metric(context=Context(), data=dummy_df)
    print(pairwise_dict)


## TODO implement trivial version of the BLEU metric

## TODO implement modified version of the BLEU metric

## TODO implement pairwise inner product for matrices
# see torch.nn.CosineEmbeddingsLoss
