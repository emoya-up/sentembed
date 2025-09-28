import os
import json
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from compare.context import Context  # Adjust import as needed
from wakepy import keep

def encode_csv_files(ground_truth, field_names) -> np.ndarray:
    """
    Trains on the ground truth by minimizing the loss between two matching
    sentences
    """
    encoder = SentenceTransformer('sentence-transformers/LaBSE')
    joint_array = np.array([0, 0])

    ground_truth_df = pd.read_csv(ground_truth, header=0, index_col=0)
    ground_truth_df = ground_truth_df[:5000]
    
    source = ground_truth_df.loc[:,field_names[0]]
    target = ground_truth_df.loc[:,field_names[1]]
    
    # encode the sentences
    source_mat = encoder.encode(source.tolist())
    target_mat = encoder.encode(target.tolist())

    # return the matrices as a 3D numpy array
    joint_array = np.stack([source_mat, target_mat], axis=0)

    return joint_array
    
        
if __name__ == "__main__":
    with keep.running():
        
        path = 'data/train/'
        for pair in os.listdir(path):
            print(f'Aligning {pair}...')
            alignments = encode_csv_files(f'{path}{pair}', ['ln1', 'ln2'])

            # Convert the numpy array to a nested list and save as JSON
            with open(f'embeddings/train/{pair[:5]}.json', 'w') as file:
                json.dump(alignments.tolist(), file)
            