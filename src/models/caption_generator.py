from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.merge import add
from keras.models import Model
from keras import Input


def simple_tutorial_model(max_length, vocab_size, embedding_dim):
    # TODO: write functionalty to attach the pre trained embeddings
    # image feature extractor model
    inputs1 = Input(shape=(1536,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # partial caption sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # decoder (feed forward) model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # merge the two input models
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.summary()
    return model
