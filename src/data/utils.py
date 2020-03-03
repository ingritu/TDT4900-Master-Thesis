from pathlib import Path
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


def max_length_caption(df_path, df=None):
    if df is None:
        df = pd.read_csv(df_path)
    captions = set(df.loc[:, 'clean_caption'])
    length = max(len(c.split()) for c in captions)
    print('DATAFRAME PATH', df_path)
    print('MAX LENGTH OF CAPTION', length)
    return length


def load_vocabulary(voc_path):
    # vocabulary need to be consistent whenever this function is called
    # vocabulary must therefore be loaded as a list
    with open(voc_path, 'r') as voc_file:
        vocabulary = [word.strip() for word in voc_file.readlines()
                      if len(word) > 0]
        max_len = int(vocabulary[0])
        vocabulary = vocabulary[1:]
    vocabulary.insert(0, 'UNK')
    wordtoix = {}
    ixtoword = {}
    for i, word in enumerate(vocabulary):
        wordtoix[word] = i
        ixtoword[i] = word
    return wordtoix, ixtoword, max_len
