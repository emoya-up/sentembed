import os
import pandas as pd

#path = '../../../europarl-extract-0.9/corpora/corpora/parallel'
path = 'data/europarl_parallel'
for pair in os.listdir(path):
    print(pair)
    pairDf = pd.DataFrame(columns=['ln1', 'ln2'])
    tabs = os.listdir(f'{path}/{pair}/tab/')
    for t in tabs:
        try:
            pairDf = pd.concat(
                [pairDf, pd.read_table(f'{path}/{pair}/tab/{t}', names=['ln1', 'ln2'])],
                axis=0,
                ignore_index=True
                )
        except KeyboardInterrupt:
            pairDf.to_csv(f'data/train/{pair}.csv')
            exit()
        except pd.errors.ParserError:
            continue
    pairDf.to_csv(f'data/train/{pair}.csv')
