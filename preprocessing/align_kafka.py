import numpy as np
import pandas as pd

# find all alternative text segments for a given row
def alt_segments(l2_witness: pd.Series, l1_equivalent: pd.Series, index: int) \
    -> pd.Series:
    ans = l2_witness.where(l1_equivalent == str(index))
    ans = ans.dropna()
    ans = '\n'.join(ans)
    return ans


def align_indexed(original_df: pd.DataFrame, main_column: str) -> pd.DataFrame:
    '''
    takes all alternative text segments from the secondary text witness and
    aligns it with the first text witness's lines to arrive at a 1-1 ground
    truth
    args:
        original_df: the original dataframe with columns containing 1) the
            primary text 2) the secondary text 3) alignments
        main_column: string indicating the column name for the primary text
    '''
    aligned_df = pd.DataFrame(original_df.loc[:, main_column])

    # add any alternative text segments as new columns
    l2_segments = []
    for i, row in original_df.iterrows():
        alternatives = alt_segments(original_df.loc[:,"1"],
                                    original_df.loc[:,"Alignierung"],
                                    i)
        l2_segments.append(alternatives)
        
    l2_segments = pd.Series(l2_segments)
    aligned_df = pd.concat([aligned_df, l2_segments], axis=1, names=['index', 'l1', 'l2'])
    
    return aligned_df

exampleDf = pd.read_csv('data/kafka_man_ann_aligniert.csv',
                        header=0,
                        index_col=0)

oneTo1Df = align_indexed(exampleDf, "0")
oneTo1Df.to_csv('data/arendt_kafka_1to1.csv')
