
def load_vocabulary(voc_path):
    vocabulary = set()
    with open(voc_path, 'r') as voc_file:
        vocabulary.add([word.strip() for word in voc_file.readlines()
                        if len(word) > 0])
    vocabulary.add('UNK')
    wordtoix = {}
    ixtoword = {}
    for i, word in enumerate(vocabulary):
        wordtoix[word] = i
        ixtoword[i] = word
    return wordtoix, ixtoword
