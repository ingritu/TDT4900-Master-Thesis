
def load_vocabulary(voc_path):
    # vocabulary need to be consistent whenever this function is called
    # vocabulary must therefore be loaded as a list
    with open(voc_path, 'r') as voc_file:
        vocabulary = [word.strip() for word in voc_file.readlines()
                      if len(word) > 0]
    vocabulary.insert(0, 'UNK')
    wordtoix = {}
    ixtoword = {}
    for i, word in enumerate(vocabulary):
        wordtoix[word] = i
        ixtoword[i] = word
    return wordtoix, ixtoword
