
def load_vocabulary(voc_path):
    vocabulary = {}
    with open(voc_path, 'r') as voc_file:
        for i, word in enumerate(voc_file.readlines()):
            vocabulary[word] = i
    return vocabulary
