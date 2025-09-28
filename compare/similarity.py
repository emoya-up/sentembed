import pandas as pd
import torch.nn as nn

# adequate versions of cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CosineEmbeddingLoss

# trivial version of the BLEU score (takes string input)
from nltk.translate.bleu_score import sentence_bleu
from docx import Document

# internal
from .context import Context

dummy_df = pd.DataFrame({
    'esperanto' : ['spam', 'eggs', 'spam', 'eggs'] * 6,
    'latin' : ['alpha', 'beta', 'gamma'] * 8})

## Jaccard metric (mostly pairwise)
def jaccard_metric(#context: Context,
                   data: pd.DataFrame,
                   decision_func=cosine_similarity,
                   threshold=.5,
                   yields=False):
    '''
    This Jaccard metric measures the distance of 2 sets.
    The function reports the average of a binary decision on all rows
    in a pd.DataFrame.
    args:
        data: pd.DataFrame with source/target language features and rows of
        text segments
        threshold: binary decision threshold
    returns (or yields):
        dict with averages of the binary Jaccard
    '''

    # # confirm that the context is empty
    # if len(context.embeddings) != 0:
    #     return False
    
    cols = data.columns
    seen_pairs = {}
    
     # fixed-schedule pair-processing
    for ln1 in cols:
        avg_sim = 0
        for ln2 in cols:
            if f'{ln1, ln2}' not in seen_pairs and ln1 != ln2:
                # encode the two text segments
                context = Context(data[ln1].to_list()[1:], data[ln2].to_list()[1:])

                # apply the decision function to two text segments
                try:
                    avg_sim = decision_func(context.embeddings[0], context.embeddings[1])
                except TypeError as e:
                    print(e)
                
                # register the language pair as seen
                seen_pairs[f'{ln1, ln2}'] = avg_sim
    
    if yields:
        # TODO (optional) implement mode that yields the similarities in a loop
        # this allows for a pairwise jaccard metric
        pass
    else:
        # returns similarity based on decision function
        return seen_pairs
    

# pairwise inner product for matrices
def pairwiseInnerProduct():
    pass

# report on similarity
def report_similarity(similarity_dict, output_format='docx', filename='similarity_report'):
    """
    reports similarity scores in either docx or latex format
    args:
        similarity_dict: dict with language pair keys and similarity values
        output_format: 'docx' or 'latex'
        filename: output file name (without extension)
    returns:
        True
    """
    if output_format == 'docx':
        doc = Document()
        doc.add_heading('Similarity Report', 0)
        table = doc.add_table(rows=1, cols=2)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Language Pair'
        hdr_cells[1].text = 'Similarity Score'
        for pair, score in similarity_dict.items():
            row_cells = table.add_row().cells
            row_cells[0].text = str(pair)
            row_cells[1].text = f"{score:.4f}" if isinstance(score, float) else str(score)
        doc.save(f"{filename}.docx")

    elif output_format == 'latex':
        with open(f"{filename}.tex", "w") as f:
            f.write("\\begin{tabular}{|l|c|}\n\\hline\n")
            f.write("Language Pair & Similarity Score \\\\\n\\hline\n")
            for pair, score in similarity_dict.items():
                score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                f.write(f"{pair} & {score_str} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{document}\n")
    else:
        raise ValueError("Unsupported format. Use 'docx' or 'latex'.")

# print output (formats)
if __name__ == "__main__":
    import sys

    arendt_df = pd.read_csv('data/arendt_kafka_1to1.csv', names=['de1', 'de2'], index_col=0)
    arendt_acs = jaccard_metric(arendt_df, cosine_similarity)

    report_similarity(arendt_acs, 'latex', )
