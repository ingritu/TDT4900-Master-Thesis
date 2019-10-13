import numpy as np


def load_glove_vectors(glove_path):
    # Load Glove vectors
    embeddings_index = {}  # empty dictionary
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def embeddings_matrix(vocab_size, wordtoix, embeddings_index, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
        # if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
