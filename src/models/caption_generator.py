from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.merge import concatenate
from keras.models import Model
from keras import Input
from keras.layers import RepeatVector
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
from copy import deepcopy

from src.models.custom_layers2 import LSTMWithVisualSentinel
from src.models.custom_layers2 import AdaptiveAttention
from src.models.custom_layers import SentinelLSTM

ROOT_PATH = Path(__file__).absolute().parents[2]


def model_switcher(model_str, max_len):
    switcher = {
        "TestModel": TestModel,
        "TutorialModel": TutorialModel,
        "AdaptiveModel": AdaptiveModel,
        "Default": TestModel
    }
    return switcher[model_str](max_len)


class CaptionGenerator:

    def __init__(self, max_length,
                 voc_path=ROOT_PATH.joinpath('data',
                                             'interim',
                                             'Flickr8k',
                                             'Flickr8k_vocabulary.csv'),
                 embedding_dim=300,
                 pre_trained_embeddings=False,
                 em_path=ROOT_PATH.joinpath('data',
                                            'processed',
                                            'glove',
                                            'glove.42B.300d.txt'),
                 feature_path=ROOT_PATH.joinpath('data',
                                                 'processed',
                                                 'Flickr8k',
                                                 'Images',
                                                 'encoded_full_images.pkl'),
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

    def compile(self,
                optimizer='adam',
                loss='categorical_crossentropy',
                weights=None):
        # build and compile the model
        self.build_model(weights=weights)
        self.model.compile(optimizer=optimizer,
                           loss=loss)

    def build_model(self, weights=None):
        pass

    def load_model(self, load_path):
        load_path = Path(load_path)
        self.model = load_model(load_path)

    def save_model(self, save_path):
        save_path = Path(save_path)
        self.model.save(save_path)

    def get_model(self):
        return self.model

    def train(self,
              train_path,
              val_path,
              batch_size=32,
              epochs=30,
              seed=2222):
        # TODO: implement early stopping
        train_df = pd.read_csv(train_path)

        steps_per_epoch = len(train_df) // batch_size

        train_generator = data_generator(train_df,
                                         batch_size=batch_size,
                                         steps_per_epoch=steps_per_epoch,
                                         wordtoix=self.wordtoix,
                                         features=self.encoded_features,
                                         seed=seed)
        # hold out on validation for now
        '''''''''
        val_df = pd.read_csv(val_path)
        val_steps = len(val_df) // batch_size
        val_generator = data_generator(val_df,
                                       batch_size=batch_size,
                                       steps_per_epoch=val_steps,
                                       wordtoix=self.wordtoix,
                                       features=self.encoded_features)
        '''''''''
        # train model
        self.model.fit_generator(train_generator,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch)
        # save model
        date_time_obj = datetime.now()
        timestamp_str = date_time_obj.strftime("%d-%b-%Y_(%H:%M:%S)")
        save_path = self.model_save_path.joinpath(
            self.model_name + '_' + str(timestamp_str) + '.h5')
        self.save_model(save_path)

    def predict_greedy(self, test_path):
        """
        Make predictions using greedy search as inference.

        Parameters
        ----------
        test_path : Path or str. Path to the .csv file containing test
            cases.

        Returns
        -------
        predicted_captions : dict. Dictionary containing one prediction
            per image.
        """
        test_path = Path(test_path)
        test_df = pd.read_csv(test_path)
        predicted_captions = {}
        images = test_df.loc[:, 'image_id']
        for image_name in images:
            image = self.encoded_features[image_name]
            prediction = self.greedy_search(image)
            predicted_captions[image_name] = prediction
        return predicted_captions

    def predict_beam(self, test_path, b=3):
        """
        Make predictions using beam search as inference.

        Parameters
        ----------
        test_path : Path or str. Path to the .csv file containing test
            cases.
        b : int. beam size. Default is 3.

        Returns
        -------
        predicted_captions : dict. Dictionary containing all b
            predictions per image, including the probabilities.
        """
        test_path = Path(test_path)
        test_df = pd.read_csv(test_path)
        predicted_captions = {}
        images = test_df.loc[:, 'image_id']
        for image_name in images:
            image = self.encoded_features[image_name]
            predictions = self.beam_search(image, b=b)
            predicted_captions[image_name] = predictions
        return predicted_captions

    def beam_search(self, image, b=3):
        """
        Beam search inference. Keep track on the b most probable
        captions at any step, until no more predictions can be made.

        Parameters
        ----------
        image : Numpy array
        b : int beam size. Default value is 3.

        Returns
        -------
        captions : list. contains b captions.
        """
        # initialization
        in_token = ['startseq']
        captions = [[in_token, 0.0]]
        for _ in range(self.max_length):
            # check if all captions have their endseq token.
            all_done = True
            for caption in captions:
                # check for at least one caption without endseq token.
                if caption[0][-1] != 'endseq' and all_done:
                    all_done = False
            if all_done:
                break

            # size of tmp_captions is max b^2
            tmp_captions = []
            for caption in captions:
                if caption[0][-1] == 'endseq':
                    # skip expanding if caption has an 'endseq' token.
                    tmp_captions.append(caption)
                    continue
                # if this process proves to be too computationally heavy
                # then consider trade off with memory, by having extra
                # variable with both index rep and string rep.
                sequence = [self.wordtoix[w] for w in caption[0]
                            if w in self.wordtoix]
                sequence = pad_sequences([sequence], maxlen=self.max_length)
                # get predictions
                y_predictions = self.model.predict([[image], sequence],
                                                   verbose=0)
                # get the b most probable indices
                words_predicted = np.argsort(y_predictions)[-b:]
                for word in words_predicted:
                    new_partial_cap = deepcopy(caption[0])
                    # add the predicted word to the partial caption
                    new_partial_cap.append(self.ixtoword[word])
                    new_partial_cap_prob = caption[1] + y_predictions[word]
                    # add cap and prob to tmp list
                    tmp_captions.append([new_partial_cap,
                                         new_partial_cap_prob])
            captions = tmp_captions
            captions.sort(key=lambda l: l[1])
            captions = captions[-b:]

        return captions

    def greedy_search(self, image):
        """
        Greedy search inference. Always choose the most probable word as
        the next in the sequence.

        Parameters
        ----------
        image : Numpy array.

        Returns
        -------
        caption : str. caption generates given image.
        """
        # Maximum likelihood estimation
        in_tokens = ['startseq']
        for i in range(self.max_length):
            sequence = [self.wordtoix[w] for w in in_tokens
                        if w in self.wordtoix]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            y_predicted = self.model.predict([[image], sequence], verbose=0)
            y_predicted = np.argmax(y_predicted)
            word = self.ixtoword[y_predicted]
            in_tokens.append(word)
            if word == 'endseq':
                break
        caption = in_tokens[1:-1]
        caption = ' '.join(caption)
        return caption


class TutorialModel(CaptionGenerator):

    def __init__(self, max_length,
                 voc_path=ROOT_PATH.joinpath('data',
                                             'interim',
                                             'Flickr8k',
                                             'Flickr8k_vocabulary.csv'),
                 embedding_dim=300,
                 pre_trained_embeddings=False,
                 em_path=ROOT_PATH.joinpath('data',
                                            'processed',
                                            'glove',
                                            'glove.42B.300d.txt'),
                 feature_path=ROOT_PATH.joinpath('data',
                                                 'processed',
                                                 'Flickr8k',
                                                 'Images',
                                                 'encoded_full_images.pkl'),
                 save_path=ROOT_PATH.joinpath('models'),
                 verbose=True
                 ):
        super().__init__(max_length,
                         voc_path=voc_path,
                         embedding_dim=embedding_dim,
                         pre_trained_embeddings=pre_trained_embeddings,
                         em_path=em_path,
                         feature_path=feature_path,
                         save_path=save_path,
                         verbose=verbose)
        self.model_name = 'Tutorial'

    def build_model(self, weights=None):
        # this is run when model.compile() is called
        if weights is not None:
            # load saved model
            self.load_model(weights)
        else:
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
                                                     embeddings_index,
                                                     self.embedding_dim)
                # Attach pre-trained embeddings to embeddings layer
                self.model.layers[2].set_weights([embedding_matrix])
                self.model.layers[2].trainable = False

        # print summary
        if self.verbose:
            self.model.summary()


class AdaptiveModel(CaptionGenerator):

    def __init__(self, max_length,
                 voc_path=ROOT_PATH.joinpath('data',
                                             'interim',
                                             'Flickr8k',
                                             'Flickr8k_vocabulary.csv'),
                 embedding_dim=300,
                 pre_trained_embeddings=False,
                 em_path=ROOT_PATH.joinpath('data',
                                            'processed',
                                            'glove',
                                            'glove.42B.300d.txt'),
                 feature_path=ROOT_PATH.joinpath('data',
                                                 'processed',
                                                 'Flickr8k',
                                                 'Images',
                                                 'encoded_full_images.pkl'),
                 save_path=ROOT_PATH.joinpath('models'),
                 verbose=True
                 ):
        super().__init__(max_length,
                         voc_path=voc_path,
                         embedding_dim=embedding_dim,
                         pre_trained_embeddings=pre_trained_embeddings,
                         em_path=em_path,
                         feature_path=feature_path,
                         save_path=save_path,
                         verbose=verbose)
        self.model_name = 'AdaptiveModel'

    def build_model(self, weights=None):
        # This is run when model.compile() is called
        if weights is not None:
            # load saved model
            self.load_model(weights)
        else:
            # build model
            encoder_inputs = Input(shape=(1536,))
            # resize to global features
            # TODO

            decoder_inputs = Input(shape=(self.max_length,))
            # resize word input

            # TODO merge global visual features and word

            lstm_layer = LSTMWithVisualSentinel(2048)

            # TODO merge visual, s_t and h_t

            attention_layer = AdaptiveAttention(2048, 30)

            # TODO merge z_t and h_t

            # can add more Dense layers for decoding here

            output = Dense(self.vocab_size, activation='softmax')

            self.model = Model(input=[encoder_inputs, decoder_inputs],
                               output=output)

            # pre-trained embeddings
            if self.pre_trained_embeddings:
                # Load gloVe embeddings
                embeddings_index = load_glove_vectors(self.em_path)
                embedding_matrix = embeddings_matrix(self.vocab_size,
                                                     self.wordtoix,
                                                     embeddings_index,
                                                     self.embedding_dim)
                # Attach pre-trained embeddings to embeddings layer
                self.model.layers[2].set_weights([embedding_matrix])
                self.model.layers[2].trainable = False

        # print summary
        if self.verbose:
            self.model.summary()


class TestModel(CaptionGenerator):

    def __init__(self, max_length,
                 voc_path=ROOT_PATH.joinpath('data',
                                             'interim',
                                             'Flickr8k',
                                             'Flickr8k_vocabulary.csv'),
                 embedding_dim=300,
                 pre_trained_embeddings=False,
                 em_path=ROOT_PATH.joinpath('data',
                                            'processed',
                                            'glove',
                                            'glove.42B.300d.txt'),
                 feature_path=ROOT_PATH.joinpath('data',
                                                 'processed',
                                                 'Flickr8k',
                                                 'Images',
                                                 'encoded_full_images.pkl'),
                 save_path=ROOT_PATH.joinpath('models'),
                 verbose=True
                 ):
        super().__init__(max_length,
                         voc_path=voc_path,
                         embedding_dim=embedding_dim,
                         pre_trained_embeddings=pre_trained_embeddings,
                         em_path=em_path,
                         feature_path=feature_path,
                         save_path=save_path,
                         verbose=verbose)
        self.model_name = 'TestModel'

    def build_model(self, weights=None):
        # This is run when model.compile() is called
        if weights is not None:
            # load saved model
            self.load_model(weights)
        else:
            # build model
            encoder_inputs = Input(shape=(1536,))
            enc_dr = Dropout(0.5)(encoder_inputs)
            encoder_fc1 = Dense(512, activation='relu')(enc_dr)

            encoder_rep = RepeatVector(self.max_length)(encoder_fc1)

            decoder_inputs = Input(shape=(self.max_length,))
            em = Embedding(self.vocab_size,
                            self.embedding_dim,
                            mask_zero=True)(decoder_inputs)
            print(em.shape)
            dec_dr = Dropout(0.5)(em)
            dec_fc1 = Dense(512, activation='relu')(dec_dr)
            print(dec_fc1.shape)

            # merge global visual features and word
            concat1 = concatenate([encoder_rep, dec_fc1])  # (, 248)

            print(concat1.shape)

            h_t, s_t, states = LSTMWithVisualSentinel(512)(concat1)

            print(h_t.shape)
            print(s_t.shape)
            print(states.shape)

            # probably redundant if custom layer is working
            #concat2 = concatenate([lstm_layer[0], lstm_layer[1]])

            # can add more Dense layers for decoding here

            output = \
                Dense(self.vocab_size, activation='softmax')(h_t)

            self.model = Model(input=[encoder_inputs, decoder_inputs],
                               output=output)

            # pre-trained embeddings
            if self.pre_trained_embeddings:
                # Load gloVe embeddings
                embeddings_index = load_glove_vectors(self.em_path)
                embedding_matrix = embeddings_matrix(self.vocab_size,
                                                     self.wordtoix,
                                                     embeddings_index,
                                                     self.embedding_dim)
                # Attach pre-trained embeddings to embeddings layer
                self.model.layers[2].set_weights([embedding_matrix])
                self.model.layers[2].trainable = False

        # print summary
        if self.verbose:
            self.model.summary()
