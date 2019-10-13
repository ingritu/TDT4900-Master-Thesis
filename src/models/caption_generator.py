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
from src.features.Resnet_features import load_visual_features
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from datetime import datetime

ROOT_PATH = Path(__file__).absolute().parents[2]


class CaptionGenerator:

    def __init__(self, max_length,
                 voc_path=ROOT_PATH.joinpath('data',
                                             'interim',
                                             'Flickr8k',
                                             'Flickr8k_vocabulary.csv'),
                 embedding_dim=200,
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
                 save_path=ROOT_PATH.joinpath('models'),
                 verbose=True
                 ):
        self.model = None
        self.model_name = 'AbstractSuper'
        self.model_save_path = save_path
        self.verbose = verbose
        self.embedding_dim = embedding_dim
        self.wordtoix, self.ixtoword = load_vocabulary(voc_path)
        self.encoded_features = load_visual_features(feature_path)
        self.vocab_size = len(self.wordtoix)
        self.pre_trained_embeddings = pre_trained_embeddings
        self.em_path = em_path
        self.feature_path = feature_path  # redundant now
        self.max_length = max_length

    def set_embedding_dim(self, value):
        # ony intended to be used before compile
        self.embedding_dim = value

    def set_max_length(self, value):
        # only intended to be used before compile
        self.max_length = value

    def compile(self, optimizer='adam'):
        # build and compile the model
        self.build_model()
        self.model.compile(optimizer=optimizer,
                           loss='categorical_cross_entropy')

    def build_model(self):
        pass

    def load_model(self, load_path):
        self.model = load_model(load_path)

    def train(self,
              train_path,
              val_path,
              batch_size=32,
              epochs=30,
              seed=2222):
        # TODO: implement early stopping
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        steps_per_epoch = len(train_df) // batch_size
        val_steps = len(val_df) // batch_size
        train_generator = data_generator(train_df,
                                         batch_size=batch_size,
                                         steps_per_epoch=steps_per_epoch,
                                         wordtoix=self.wordtoix,
                                         features=self.encoded_features,
                                         seed=seed)
        val_generator = data_generator(val_df,
                                       batch_size=batch_size,
                                       steps_per_epoch=val_steps,
                                       wordtoix=self.wordtoix,
                                       features=self.encoded_features)
        # train model
        self.model.fit_generator(train_generator,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=val_generator,
                                 validation_steps=val_steps)
        # save model
        date_time_obj = datetime.now()
        timestamp_str = date_time_obj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
        self.model.save(self.model_name + '_' + str(timestamp_str) +
                        '_' + '.h5')

    def predict(self, test_path):
        # TODO: implement beam search instead of greedy search/MLE
        test_df = pd.read_csv(test_path)
        predicted_captions = {}
        images = test_df.loc[:, 'image_id']
        for image_name in images:
            image = self.encoded_features[image_name]
            prediction = self.greedy_search(image)
            predicted_captions[image_name] = prediction
        return predicted_captions

    def greedy_search(self, image):
        # Maximum likelihood estimation
        in_tokens = ['startseq']
        for i in range(self.max_length):
            sequence = [self.wordtoix[w] for w in in_tokens if w in self.wordtoix]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            y_predicted = self.model.predict([image, sequence], verbose=0)
            y_predicted = np.argmax(y_predicted)
            word = self.ixtoword[y_predicted]
            in_tokens.append(word)
            if word == 'endseq':
                break
        final = in_tokens[1:-1]
        final = ' '.join(final)
        return final


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
                 save_path=ROOT_PATH.joinpath('models'),
                 verbose=True
                 ):
        super().__init__(max_length,
                         voc_path,
                         embedding_dim,
                         pre_trained_embeddings,
                         em_path,
                         feature_path,
                         save_path,
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
        # pre-trained embeddings
        if self.pre_trained_embeddings:
            # Load gloVe embeddings
            embeddings_index = load_glove_vectors(self.em_path)
            embedding_matrix = embeddings_matrix(self.vocab_size,
                                                 self.wordtoix,
                                                 embeddings_index)
            # Attach pre-trained embeddings to embeddings layer
            self.model.layers[2].set_weights([embedding_matrix])
            self.model.layers[2].trainable = False
        # print summary
        if self.verbose:
            self.model.summary()
