
from pathlib import Path
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


def basic_data_cleaning(df_path):
    """
    When we deal with text, we generally perform some basic cleaning
    like lower-casing all the words (otherwise“hello” and “Hello” will
    be regarded as two separate words), removing special tokens
    (like ‘%’, ‘$’, ‘#’, etc.), eliminating words which contain
    numbers (like ‘hey199’, etc.).
    """
    caption_df = pd.read_csv(df_path)
    # make a new column for the cleaned version of the caption
    # I do this because I want to keep the original version
    caption_df['clean_caption'] = ''
    # corpus of words, {word: occurrence in corpus}
    corpus = {}

    for i in range(len(caption_df)):






    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)






