
from pathlib import Path
import pandas as pd
import string

ROOT_PATH = Path(__file__).absolute().parents[2]

THRESHOLD = 3
UNK_PERCENTAGE = 0.4


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

    # replace words with less than 3 occurrences with a UNK token
    replace_corpus = set([key for key in corpus if corpus[key] < THRESHOLD])
    for i in range(len(caption_df)):
        cap_tokens = caption_df.loc[i, 'clean_caption'].split()
        cap_tokens = [word if word not in replace_corpus else 'UNK'
                      for word in cap_tokens]
        cap_tokens = ['startseq'] + cap_tokens + ['endseq']
        caption_df.at[i, 'clean_caption'] = ' '.join(cap_tokens)

    # remove captions with more than 40% UNK tokens in captions
    count_before = len(caption_df)
    remove_caps = []
    for i in range(len(caption_df)):
        caption = caption_df.loc[i, 'clean_caption'].split()
        length = len(caption)
        unks = sum([1 if w == 'UNK' else 0 for w in caption])
        if unks/length > UNK_PERCENTAGE:
            print("removes too many UNKs")
            remove_caps.append(caption_df.loc[i, 'caption_id'])

    caption_df = caption_df.loc[
                 ~caption_df.loc[:, 'caption_id'].isin(remove_caps), :]
    caption_df = caption_df.reset_index(drop=True)
    count_after = len(caption_df)
    if count_after != count_before - len(remove_caps):
        print("Did not remove 40% UNK captions!!"
              "\nOr something else went wrong.")

    # remove bad mm mmmmm mmmm captions if they still exist
    count_before = count_after
    remove_caps = []
    for i in range(len(caption_df)):
        caption = caption_df.loc[i, 'clean_caption'].split()
        if is_all_one_letter(caption, 'm'):
            print("adds mmmmm mmmm to remove")
            remove_caps.append(caption_df.loc[i, 'caption_id'])

    caption_df = caption_df.loc[
                 ~caption_df.loc[:, 'caption_id'].isin(remove_caps), :]
    caption_df = caption_df.reset_index(drop=True)
    count_after = len(caption_df)
    if count_after != count_before - len(remove_caps):
        print("Did not remove bad mmmm captions!!"
              "\nOr something else went wrong.")

    print('vocabulary size that will be used:',
          len(corpus) - len(replace_corpus))
    # save captions
    caption_df.to_csv(save_path)

    # vocabulary stuff
    vocabulary = [key for key in corpus.keys() if corpus[key] >= THRESHOLD]
    vocabulary.insert(0, 'startseq')
    vocabulary.insert(1, 'endseq')
    with open(voc_save_path, 'w') as voc_file:
        for word in vocabulary:
            voc_file.write(word + '\n')


def is_all_one_letter(caption, letter):
    tmp_caption = caption[1:len(caption) - 1]
    for word in tmp_caption:
        for char in word:
            if char != letter:
                return False
    return True


def number_to_word(word):
    numbers = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
               '5': 'five', '6': 'six', '7': 'seven', '8': 'eight',
               '9': 'nine'}
    if word in numbers.keys():
        return numbers[word]
    return ''
