
from pathlib import Path
import pandas as pd
import string

ROOT_PATH = Path(__file__).absolute().parents[2]

THRESHOLD = 3


def basic_data_cleaning(df_path, save_path, voc_save_path):
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
    table = str.maketrans('', '', string.punctuation)
    for i in range(len(caption_df)):
        cap_tokens = caption_df.loc[i, 'caption'].split()
        # convert to lower
        cap_tokens = [word.lower() for word in cap_tokens]
        # remove punctuation from each token
        cap_tokens = [word.translate(table) for word in cap_tokens]
        # remove hanging 's'
        cap_tokens = [word for word in cap_tokens
                      if len(word) > 1 or word == 'a']
        # remove tokens with numbers in them
        cap_tokens = [word for word in cap_tokens if word.isalpha()]

        # add words to corpus
        for word in cap_tokens:
            if word not in corpus.keys():
                corpus[word] = 1
            else:
                corpus[word] += 1

        # add the cleaned caption to df
        caption_df.at[i, 'clean_caption'] = ' '.join(cap_tokens)
    print("Full vocabulary size:", len(corpus))

    # replace words with less than 3 occurrences with a UNK token
    replace_corpus = set([key for key in corpus if corpus[key] < THRESHOLD])
    for i in range(len(caption_df)):
        cap_tokens = caption_df.loc[i, 'clean_caption'].split()
        cap_tokens = [word if word not in replace_corpus else 'UNK'
                      for word in cap_tokens]
        cap_tokens = ['startseq'] + cap_tokens + ['endseq']
        caption_df.at[i, 'clean_caption'] = ' '.join(cap_tokens)

    print('vocabulary size that will be used:',
          len(corpus) - len(replace_corpus))

    caption_df.to_csv(save_path)
    vocabulary = [key for key in corpus.keys() if corpus[key] >= THRESHOLD]
    with open(voc_save_path, 'w') as voc_file:
        for word in vocabulary:
            voc_file.write(word + '\n')


if __name__ == '__main__':
    df_path_ = ROOT_PATH.joinpath('data', 'interim', 'Flickr8k',
                                  'Flickr8k_train.csv')
    save_path_ = ROOT_PATH.joinpath('data', 'interim', 'Flickr8k',
                                    'Flickr8k_train_clean.csv')
    voc_save_path_ = ROOT_PATH.joinpath('data', 'interim', 'Flickr8k',
                                        'Flickr8k_vocabulary.csv')
    basic_data_cleaning(df_path_, save_path_, voc_save_path_)
