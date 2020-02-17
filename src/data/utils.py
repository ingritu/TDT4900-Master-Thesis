from pathlib import Path
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


def max_length_caption(df_path):
    df = pd.read_csv(df_path)
    captions = set(df.loc[:, 'clean_caption'])
    length = max(len(c.split()) for c in captions)
    print('DATAFRAME PATH', df_path)
    print('MAX LENGTH OF CAPTION', length)
    return length