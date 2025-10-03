import pandas as pd
import os

path_a2 = 'data/apa-rst/a2-b1/'
path_or = 'data/apa-rst/b1-or/'
threeway_df = pd.DataFrame([])

rows = []
for file_a, file_or in zip(os.listdir(path_a2), os.listdir(path_or)):
    row_a = pd.read_csv(f'{path_a2}{file_a}')[["A2 sentence", "B1 sentence"]]
    row_o = pd.read_csv(f'{path_or}{file_or}')[["OR sentence"]]
    row = pd.concat([row_a, row_o], axis=1)
    
    rows.append(row)
    
threeway_df = pd.concat(rows, axis=0, ignore_index=True)
threeway_df.to_csv('data/apa-rst_3way.csv')