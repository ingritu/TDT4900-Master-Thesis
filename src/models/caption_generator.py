from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.merge import add
from keras.models import Model
from keras import Input
from pathlib import Path
from src.features.glove_embeddings import embeddings_matrix
from src.features.glove_embeddings import load_glove_vectors
from src.data.load_vocabulary import load_vocabulary
from src.data.data_generator import data_generator
import pandas as pd

ROOT_PATH = Path(__file__).absolute().parents[2]


class CaptionGenerator:

    def __init__(self, max_length,
                 voc_path,
                 embedding_dim,
                 pre_trained_embeddings=True,
                 em_path=ROOT_PATH.joinpath('data',
                                            'processed',
                                            'glove',
                                            'glove.42B.300d.txt'),
                 feature_path=ROOT_PATH.joinpath('data',
                                                 'processed',
                                                 'Flickr8k',
                                                 'Images',
                                                 '299x299'),
                 verbose=True
                 ):
        self.model = None
        self.model_name = 'AbstractSuper'
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.wordtoix, self.ixtoword = load_vocabulary(voc_path)
        self.vocab_size = len(self.wordtoix)
        self.pre_trained_embeddings = pre_trained_embeddings
        self.em_path = em_path
        self.feature_path = feature_path
        self.max_length = max_length
        self.build_model()

    def compile(self, optimizer='adam'):
        self.model.compile(optimizer=optimizer,
                           loss='categorical_cross_entropy')

    def build_model(self):
        pass

    def train(self,
              train_path,
              val_path,
              batch_size,
              epochs,
              seed=2222):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        steps_per_epoch = len(train_df) // batch_size
        val_steps = len(val_df) // batch_size
        train_generator = data_generator(train_df,
                                         batch_size=batch_size,
                                         steps_per_epoch=steps_per_epoch,
                                         wordtoix=self.wordtoix,
                                         feature_path=self.feature_path,
                                         seed=seed)
        val_generator = data_generator(val_df,
                                       batch_size=batch_size,
                                       steps_per_epoch=val_steps,
                                       wordtoix=self.wordtoix,
                                       feature_path=self.feature_path)
        self.model.fit_generator(train_generator,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=val_generator,
                                 validation_steps=val_steps)

    def predict(self):
        self.model.predict()


class TutorialModel(CaptionGenerator):

    def __init__(self, max_length,
                 voc_path,
                 embedding_dim,
                 pre_trained_embeddings=True,
                 em_path=ROOT_PATH.joinpath('data',
                                            'processed',
                                            'glove',
                                            'glove.42B.300d.txt'),
                 feature_path=ROOT_PATH.joinpath('data',
                                                 'processed',
                                                 'Flickr8k',
                                                 'Images',
                                                 '299x299'),
                 verbose=True
                 ):
        super().__init__(max_length,
                         voc_path,
                         embedding_dim,
                         pre_trained_embeddings,
                         em_path,
                         feature_path,
                         verbose)
        self.model_name = 'Tutorial'

    def build_model(self):
        # image feature extractor model
        inputs1 = Input(shape=(1536,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        # partial caption sequence model
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size,
                        self.embedding_dim,
                        mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        # decoder (feed forward) model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)

        # merge the two input models
        self.model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        if self.pre_trained_embeddings:
            # Load gloVe embeddings
            embeddings_index = load_glove_vectors(self.em_path)
            embedding_matrix = embeddings_matrix(self.vocab_size,
                                                 self.wordtoix,
                                                 embeddings_index)
            # Attach pre-trained embeddings to embeddings layer
            self.model.layers[2].set_weights([embedding_matrix])
            self.model.layers[2].trainable = False
        if self.verbose:
            self.model.summary()
