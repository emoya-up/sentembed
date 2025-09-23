import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from compare.context import Context  # Adjust import as needed
from wakepy import keep

def align_csv_files(ground_truth, field_names):
    """
    Trains on the ground truth by maximizing the alignment between two correct
    sentences
    """
    encoder = SentenceTransformer('sentence-transformers/LaBSE')

    ground_truth_df = pd.read_csv(ground_truth, header=0, index_col=0)
    ground_truth_df = ground_truth_df[:5000]
    source = ground_truth_df.loc[:,field_names[0]]
    target = ground_truth_df.loc[:,field_names[1]]
    
    ground_truth_df['ln1_embeddings'] = encoder.encode(source.tolist())
    print(ground_truth_df.head())
    return None
    
    alignments = []
    for s_idx, s_row in source.items():
        
        similarities = []
        for t_idx, t_row in target[s_idx:s_idx+100 if s_idx+100 < len(target) else len(target)-1].items():
            context = Context(s_row, t_row)
            similarities.append((s_idx, t_idx, context(True)))
        max_sim = max(similarities, key=lambda x: x[2])
        source.drop(max_sim[0])
        target.drop(max_sim[1])
        alignments.append(max_sim)
    return alignments
        
if __name__ == "__main__":
    with keep.running():
        
        for pair in ['DE-EN', 'FR-EN', 'ES-EN', 'PL-EN']:
            print(f'Aligning {pair}...')
            alignments = align_csv_files(f'data/{pair}.csv', ['ln1', 'ln2'])
            # alignments_df = pd.DataFrame(alignments, columns=['source_index', 'target_index', 'similarity'])
            # alignments_df.to_csv(f'data/alignments_{pair}.csv', index=False)