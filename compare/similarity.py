import os
import argparse
import pandas as pd

# adequate versions of cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# trivial version of the BLEU score (takes string input)
from nltk.translate.bleu_score import sentence_bleu
from docx import Document

# internal
from .context import Context

# handle the different modalities based on command-line input
parser = argparse.ArgumentParser(
                    prog='compare.similarity',
                    description='Compare the similarity of different embeddings')

parser.add_argument('-e', '--encoder')
parser.add_argument('-d', '--dataset')
parser.add_argument('-s', '--similarity_function')

args = parser.parse_args()

# paths to the relevant data
paths = {
    'europarl': [f'data/europarl/{file}' for file in os.listdir('data/europarl/')],
    'apa-rst': ['data/apa-rst_3way.csv'],
    'arendt': ['data/arendt/essay1.csv', 'data/arendt/arendt_kafka_1to1.csv', ]}

# dummy dataframe for testing
# dummy_df = pd.DataFrame({
#     'esperanto' : ['spam', 'eggs', 'spam', 'eggs'] * 6,
#     'latin' : ['alpha', 'beta', 'gamma'] * 8})

## Jaccard metric (mostly pairwise)
def jaccard_metric(data: pd.DataFrame,
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
    
    data_df = data.fillna('')  # replace NaN with empty strings
    cols = data_df.columns
    seen_pairs = {}
    
     # fixed-schedule pair-processing
    for ln1 in cols:
        for ln2 in cols:
            if f'{ln1, ln2}' not in seen_pairs.keys() and \
                    f'{ln2, ln1}' not in seen_pairs.keys() and \
                    ln1 != ln2:
                # encode the two text segments
                context = Context(data_df[ln1].to_list()[1:], data_df[ln2].to_list()[1:])

                # apply the decision function to two text segments
                average_sim = 0
                for x in range(context.embeddings[0].shape[0]):
                    source = context.embeddings[0][x].reshape(1, -1)
                    target = context.embeddings[1][x].reshape(1, -1)
                    
                    sim = decision_func(source, target)
                    # output for latex
                    # print(f"{x} & {context.token_lists[0][x]} & {context.token_lists[1][x]} & {sim[0,0]:.2f}\\\\")
                    average_sim += sim
                        
                average_sim = average_sim / context.embeddings[0].shape[0]
                
                # register the language pair as seen
                seen_pairs[f'{ln1, ln2}'] = average_sim
    
    if yields:
        # TODO (optional) implement mode that yields the similarities for a loop
        pass
    else:
        # returns similarity based on decision function
        return seen_pairs
    

# pairwise inner product for matrices
def pairwiseInnerProduct():
    raise NotImplementedError()

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

def visualize_invariance(dict):
    raise NotImplementedError()
    

if __name__ == "__main__":
    import sys

    for i, filepath in enumerate(paths[args.dataset]):
        aligned_df = pd.read_csv(filepath, header=0, index_col=0)
        print(aligned_df.head())
        
        average_similarity = jaccard_metric(aligned_df, cosine_similarity)
        report_similarity(average_similarity, 'latex', f'results/{args.dataset}_{i}_indepth')
        
        # visualize the invariance with a seaborn heatmap
        # visualize_invariance()

