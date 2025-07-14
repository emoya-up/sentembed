import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from similarity import cosine_similarity
from torch import Tensor, tensor, matmul, concat
from types import FunctionType
from numpyencoder import NumpyEncoder

# order of languages for toy data
#order = ['en', 'es', 'ru', 'zh', 'fr']

# set of languages for EuroParl
#order = {'EN', 'ES', 'FR', 'DE', 'PL'}

# set of languages for APA-RST
#order = {'OR', 'B2', 'A2'}

# set of languages for CrEd-HA
#order = {'en', 'de', 'fr', 'yi'}

# supersets & Boolean differences

class Context():
    '''
    This object represents the context between two distinct text witnesses, a
    pair of text segments that serves as a candidate for alignment.
    '''
    embeddings = [tensor(0), tensor(0)]
    hadamard = tensor(0)
    concat = tensor(0)
    
    pair = [tensor(0), tensor(0)]
    
    def __init__(self, tokens1, tokens2):
        # the intertextual context is generated based on the dimensions
        # of the two target sentences
        self.embeddings = [tokens1, tokens2]
    
    def _hadamard(self):
        # copy embeddings
        placeholder = [x.clone().detach() for x in self.embeddings]
        
        # element-wise multiplication of pairs
        self.hadamard = matmul(placeholder[0], placeholder[1])
        
        return self.hadamard
    
    def _concatenate(self):
        # copy embeddings
        placeholder = self.embeddings
        
        # concatenation of pairs
        self.concat = concat(placeholder)
        
        return self.concat
    
    def _compare(self, datapath: str, embeddings: Tensor):
        # copy embeddings
        placeholder = embeddings.clone().detach()
        # compare (and align) the two text witnesses
        embeddings = encode(datapath=datapath,
               savepath='data',
               modelname='sentence-transformers/LaBSE')
        
        print('Similarities:')
        for k, v in compare(embeddings, 'data', cosine_similarity).items():
            print(f'{k}: {v}')
        
        return True
    
    def __call__(self, *args, **kwds):
        # calculates basic measures and the alignment quality
        if (len(args) >= 2) or (args[0] not in [0, 1]):
            raise ValueError('Value must be a Boolean')
        
        encode()
        
        self._hadamard()
        self._concatenate()
        
        # args can only be the selection of hadamard or concatenation
        if args[0]:
            self._compare()
            return self.hadamard
        else:
            self._compare()
            return self.concat

def encode(datapath, savepath, modelname) -> dict[str, list]:
    '''
    encode a sequence of tokens with a specific model
    '''
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

def compare(data, datapath: str, sim_function: FunctionType):
    '''
    compare every line:
    However, do not be surprised if the examples seem too good to be true.
    The LABSE model was trained on the toy data and this is generally
    possible with pretrained models.
    
    args:
        data: if available, sentences from working memory
        datapath: path where sentences are stored
        sim_function: similarity function to be used
    
    return:
        Any
    '''
    
    sim_dict = {}
    compared_pairs = []
    
    # fixed-schedule pair-processing
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
    
    # returns dictionary for each language pair sorted by similarity
    return {k: v[0, 0] for k, v in sorted(sim_dict.items(),
                                          reverse=True,
                                          key=lambda x: x[1])}

'''
embs = encode('data/sent_speeches_all.csv',
              'data',
              'sentence-transformers/LaBSE')

print('Similarities:')
for k, v in compare(embs, 'data', cosine_similarity).items():
    print(f'{k}: {v}')

'''

data = ['This is one sentence', 'This is the other']
data_ = ["C'est une phrase", "Ca, c'est, l'autre"]
order = ['en', 'fr']
pair = Context(data[0], data_[0])
print(type(pair(True)))
