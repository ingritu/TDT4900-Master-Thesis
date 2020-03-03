
from pathlib import Path
import pandas as pd
import string

from src.data.utils import max_length_caption

ROOT_PATH = Path(__file__).absolute().parents[2]


def basic_data_cleaning(df_path, save_path, voc_save_path, threshold=3,
                        unk_percentage=0.4):
    """
    Basic data cleaning entails:
    - Converting to lower.
    - Removing punctuation.
    - Converting numbers to number words and removing large numbers.
    - Removing one letter words that are not 'a'.
    - Replacing uncommon words with UNK token.
    - Removing captions with too many UNK tokens.
    - Removing weired mmmmm mm mmm captions.

    Parameters
    ----------
    df_path : Path or str. Where the .csv file containing unprocessed
        cases is located.
    save_path : Path or str. Where the new .csv file containing
        pre-processed cases will be saved.
    voc_save_path : Path or str. Where the vocabulary will be saved.
    threshold : int. Word frequency threshold. Default value is 3.
    unk_percentage : float. Remove captions where the percentage of UNK tokens
        in the caption is greater than unk_percentage. Default value is 0.4.

    Returns
    -------
    Saves cleaned dataset at save_path. saves vocabulary at
    voc_save_path.
    """
    df_path = Path(df_path)
    save_path = Path(save_path)
    voc_save_path = Path(voc_save_path)
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
        # converting numbers to number words and remove large numbers
        cap_tokens = [word if word.isalpha() else number_to_word(word)
                      for word in cap_tokens]

        # remove hanging 's' and other possible weired one letter words
        cap_tokens = [word for word in cap_tokens
                      if len(word) > 1 or word == 'a']
        # add words to corpus
        for word in cap_tokens:
            if word not in corpus.keys():
                corpus[word] = 1
            else:
                corpus[word] += 1

        # add the cleaned caption to df
        caption_df.at[i, 'clean_caption'] = ' '.join(cap_tokens)
    print("Full vocabulary size:", len(corpus))

    replace_corpus, caption_df = replace_uncommon_words(caption_df,
                                                        corpus,
                                                        threshold)

    # remove captions with more than 40% UNK tokens in captions
    count_before = len(caption_df)
    caption_df, remove_caps_len = remove_too_many_unk(caption_df,
                                                      unk_percentage)
    count_after = len(caption_df)
    if count_after != count_before - remove_caps_len:
        print("Did not remove 40% UNK captions!!"
              "\nOr something else went wrong.")

    # remove bad mm mmmmm mmmm captions if they still exist
    count_before = count_after
    caption_df, remove_caps_len = remove_bad_captions(caption_df)
    count_after = len(caption_df)
    if count_after != count_before - remove_caps_len:
        print("Did not remove bad mmmm captions!!"
              "\nOr something else went wrong.")

    print('vocabulary size that will be used:',
          len(corpus) - len(replace_corpus))

    # check save path
    if not save_path.parent.is_dir():
        save_path.parent.mkdir(parents=True)

    max_len = max_length_caption("", caption_df)
    # save captions
    caption_df.to_csv(save_path)

    # vocabulary stuff
    vocabulary = [key for key in corpus.keys() if corpus[key] >= threshold]
    vocabulary.append('endseq')
    # add startseq last so that we can remove it easily from models vocabulary
    vocabulary.append('startseq')

    # check voc save path
    if not voc_save_path.parent.is_dir():
        voc_save_path.parent.mkdir(parents=True)

    with open(voc_save_path, 'w') as voc_file:
        # add maxlen to first line in vocabulary file
        voc_file.write(str(max_len) + '\n')
        for word in vocabulary:
            voc_file.write(word + '\n')


def replace_uncommon_words(caption_df, corpus, threshold=3):
    """
    Replace words with less than THRESHOLD occurrences with an UNK token.

    Parameters
    ----------
    caption_df : DataFrame.
    corpus : dict.
    threshold : int.
    """
    #
    # token
    replace_corpus = set([key for key in corpus if corpus[key] < threshold])
    for i in range(len(caption_df)):
        cap_tokens = caption_df.loc[i, 'clean_caption'].split()
        cap_tokens = [word if word not in replace_corpus else 'UNK'
                      for word in cap_tokens]
        cap_tokens = ['startseq'] + cap_tokens + ['endseq']
        caption_df.at[i, 'clean_caption'] = ' '.join(cap_tokens)
    return replace_corpus, caption_df


def remove_too_many_unk(caption_df, unk_percentage=0.4):
    """
    Remove captions with too many UNK tokens in the caption.

    Parameters
    ----------
    caption_df : DataFrame. contains pre-processed cases.
    unk_percentage : float. Remove captions where the percentage of UNK tokens
        in the caption is greater than unk_percentage. Default value is 0.4.
    Returns
    -------
    caption_df : DataFrame. Where captions with too many UNKs have
        been removed.
    remove_caps : list. caption_ids of the captions that were removed.
    """
    remove_caps = []
    for i in range(len(caption_df)):
        caption = caption_df.loc[i, 'clean_caption'].split()
        length = len(caption)
        unks = sum([1 if w == 'UNK' else 0 for w in caption])
        if unks / length > unk_percentage:
            print("removes too many UNKs")
            remove_caps.append(caption_df.loc[i, 'caption_id'])
    caption_df = caption_df.loc[
                 ~caption_df.loc[:, 'caption_id'].isin(remove_caps), :]
    caption_df = caption_df.reset_index(drop=True)
    return caption_df, len(remove_caps)


def remove_bad_captions(caption_df):
    """
    Remove objectively bad captions that do not contain real words.
    Here we only remove captions that only consists of the letter m,
    since our exploration found that such cases exist in MS COCO.

    Parameters
    ----------
    caption_df : DataFrame. contains pre-processed cases.

    Returns
    -------
    caption_df : DataFrame. Where bad captions have been removed.
    remove_caps : list. caption_ids of the captions that were removed.
    """
    remove_caps = []
    for i in range(len(caption_df)):
        caption = caption_df.loc[i, 'clean_caption'].split()
        if is_all_one_letter(caption, 'm'):
            print("adds mmmmm mmmm to remove")
            remove_caps.append(caption_df.loc[i, 'caption_id'])
    caption_df = caption_df.loc[
                 ~caption_df.loc[:, 'caption_id'].isin(remove_caps), :]
    caption_df = caption_df.reset_index(drop=True)
    return caption_df, len(remove_caps)


def is_all_one_letter(caption, letter):
    """
    Check whether caption is only a string consisting of one letter.

    Parameters
    ----------
    caption : str.
    letter : str.
    """
    tmp_caption = caption[1:len(caption) - 1]
    for word in tmp_caption:
        for char in word:
            if char != letter:
                return False
    return True


def number_to_word(word):
    """
    Transform number to number word.

    Parameters
    ----------
    word : str.
    """
    numbers = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
               '5': 'five', '6': 'six', '7': 'seven', '8': 'eight',
               '9': 'nine'}
    if word in numbers.keys():
        return numbers[word]
    return ''
