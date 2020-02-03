import torch
from torch import nn
from torch import optim
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.data.data_generator import data_generator
from src.data.load_vocabulary import load_vocabulary
from src.features.Resnet_features import load_visual_features

ROOT_PATH = Path(__file__).absolute().parents[2]


def loss_switcher(loss_string):
    switcher = {
        'cross_entropy': nn.CrossEntropyLoss,
        'MSE': nn.MSELoss,
        'Default': nn.CrossEntropyLoss,
    }

    return switcher[loss_string]


def optimizer_switcher(optimizer_string):
    switcher = {
        'adam': optim.Adam,
        'SGD': optim.SGD,
        'Default': optim.SGD
    }
    return switcher[optimizer_string]


class Generator:

    def __init__(self,
                 input_shape,
                 vocab_size,
                 voc_path,
                 feature_path,
                 loss_function='cross_entropy',
                 optimizer='adam',
                 lr=0.0001,
                 embedding_size=300,
                 seed=222):
        # initialize model as None
        self.model = None

        # delete if not used in this class
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.wordtoix, self.ixtoword = load_vocabulary(voc_path)
        self.encoded_features = load_visual_features(feature_path)

        # initialize loss function
        self.criterion = loss_switcher(loss_function)()

        # set up optimizer
        self.optimizer_string = optimizer
        self.optimizer = optimizer_switcher(optimizer)  # not initialized
        self.lr = lr

        # misc
        self.model_name = 'CaptionGeneratorFramework'
        self.random_seed = seed

    def initialize_optimizer(self):
        self.optimizer = self.optimizer(self.model.parameters(), self.lr)

    def train(self, data_path, epochs, batch_size):
        # TODO: implement early stopping on CIDEr metric
        data_path = Path(data_path)
        train_df = pd.read_csv(data_path)

        steps_per_epoch = len(train_df) // batch_size

        train_generator = data_generator(train_df, batch_size,
                                         steps_per_epoch,
                                         self.wordtoix,
                                         self.encoded_features,
                                         seed=self.random_seed)

        for e in range(epochs):
            for s in range(steps_per_epoch):
                # zero the gradient buffers
                self.optimizer.zero_grad()

                # get minibatch from data generator
                # TODO: modify datagenerator to give right output
                x, target = next(train_generator)
                # get predictions from network
                output = self.model(x)
                # get loss
                loss = self.criterion(output, target)
                print('loss', '(' + self.optimizer_string + '):', loss)

                loss.backward()
                self.optimizer.step()
        # end of training
        # save model to file
        date_time_obj = datetime.now()
        timestamp_str = date_time_obj.strftime("%d-%b-%Y_(%H:%M:%S)")
        # TODO: add timestamp to filename
        self.save_model()

    def predict(self, data_df, beam_size):
        # TODO: implement this function
        pass

    def beam_search(self, beam_size):
        # TODO: implement this function
        pass

    def load_model(self):
        # TODO: implement this function
        pass

    def save_model(self):
        # TODO: implement this function
        pass

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, string):
        assert isinstance(string, str)
        self.model_name = string
