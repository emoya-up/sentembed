import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from wakepy import keep

def embed_json(file_path, model_name, save_path):
    """
    Embeds all sentences in a CSV file and saves the embeddings as JSON files.
    """
    data_df = pd.read_csv(file_path)
    encoder = SentenceTransformer(model_name)
    
    data_dict = data_df[:5000].to_dict('records')
    for i, row in enumerate(data_dict):
        data_dict[i]['ln1_embedding'] = encoder.encode(row['ln1']).tolist()
        data_dict[i]['ln2_embedding'] = encoder.encode(row['ln2']).tolist()
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
if __name__ == "__main__":
    with keep.running():
        for pair in ['DE-EN', 'FR-EN', 'ES-EN', 'PL-EN']:
            print(f'Embedding EuroParl {pair}...')
            embed_json(f'data/{pair}.csv',
                       'sentence-transformers/LaBSE',
                       f'data/embedded_{pair}.json')
            