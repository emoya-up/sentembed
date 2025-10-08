import pandas as pd
import os

path = 'data/europarl/'
for bitext in os.listdir(path):
    df = pd.read_csv(f'{path}{bitext}', index_col=0)[:5000]
    df.to_csv(f'{path}{bitext[:-4]}_short.csv')