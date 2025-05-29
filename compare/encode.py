import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from similarity import cosine_similarity
from torch import Tensor, tensor
from numpyencoder import NumpyEncoder

order = ['en', 'es', 'ru', 'zh', 'fr']

def encode(datapath, savepath, modelname) -> dict[str, list]:
    embeddings = {}
    data = pd.read_csv(datapath, names=order)
    model = SentenceTransformer(modelname)
    
    for ln, sentences in data.items():
        emb = model.encode(sentences.to_numpy(str))
        embeddings[ln] = emb
        
        # save to json file
        with open(f"{savepath}/embeddings_{ln}.json", 'w') as file:
            json.dump(emb.tolist(), file)
        
    return embeddings


def compare(datapath: str, sim_function):
    '''
    compare every line:
    however, do not be surprised if the examples seem too good to be true
    The LABSE model was trained on this dataset
    '''
    
    sim_dict = {}
    compared_pairs = []
    
    for ln1 in order:
        for ln2 in order:
            if {ln1, ln2} not in compared_pairs:
                sims = 0
                emb1 = pd.read_json(f"{datapath}/embeddings_{ln1}.json")
                emb2 = pd.read_json(f"{datapath}/embeddings_{ln2}.json")
                for x in range(emb1.shape[1]):
                    sims += cosine_similarity(
                        emb1[x].to_numpy().reshape(1,-1),
                        emb2[x].to_numpy().reshape(1,-1))
                avg_sim = sims/emb1.shape[1]
                #print(f"{ln1} and {ln2} average similarity: {sims/emb1.shape[1]}")
                sim_dict[ln1+'-'+ln2] = avg_sim
                compared_pairs.append({ln1, ln2})
                
    return {k: v[0, 0] for k, v in sorted(sim_dict.items(), reverse=True, key=lambda x: x[1])}
    
encode('data/sent_speeches_all.csv',
       'data',
       'sentence-transformers/LaBSE')

print('Similarities:')
for k, v in compare('data', cosine_similarity).items():
    print(f'{k}: {v}')