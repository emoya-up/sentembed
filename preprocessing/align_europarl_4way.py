import pandas as pd
import os

# Path to bitext files (dissimilar)
path = 'data/europarl/'

fourway_df = 0

for file in os.listdir(path):
    print(file)
    if type(fourway_df) == pd.DataFrame:
        fourway_df = pd.concat(
            [fourway_df, pd.read_csv(f'{path}{file}').loc[:5000, 'ln1']],
            axis=1)
    else:
        fourway_df = pd.read_csv(f'{path}{file}', index_col=0)[:5000]

# Save to CSV
fourway_df.to_csv(f'{path}europarl_4way.csv')