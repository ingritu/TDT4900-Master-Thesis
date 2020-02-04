import torch
from torch import nn
from torch import optim
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.data.data_generator import data_generator
from src.data.load_vocabulary import load_vocabulary
from src.features.Resnet_features import load_visual_features
from src.models.torch_generators import model_switcher

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
                 model_name,
                 input_shape,
                 voc_path,
                 feature_path,
                 save_path,
                 loss_function='cross_entropy',
                 optimizer='adam',
                 lr=0.0001,
                 embedding_size=300,
                 seed=222):
        # delete if not used in this class
        self.input_shape = input_shape

        self.embedding_size = embedding_size

        self.save_path = save_path
        self.random_seed = seed

        self.wordtoix, self.ixtoword = load_vocabulary(voc_path)
        self.vocab_size = len(self.wordtoix)
        self.encoded_features = load_visual_features(feature_path)

        # initialize model
        self.model = model_switcher(model_name)(self.input_shape,
                                                self.vocab_size,
                                                embedding_size=
                                                self.embedding_size,
                                                seed=self.random_seed)
        print(self.model)
        self.model_name = model_name

        # initialize loss function
        self.criterion = loss_switcher(loss_function)()

        # set up optimizer
        self.optimizer_string = optimizer
        self.optimizer = None  # not initialized
        self.lr = lr
        self.initialize_optimizer()

        # misc
        self.framework_name = 'CaptionGeneratorFramework'

    def initialize_optimizer(self):
        self.optimizer = optimizer_switcher(self.optimizer_string)(
            self.model.parameters(), self.lr)

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
            print('Epoch: #' + str(e + 1))
            for s in range(steps_per_epoch):
                print('Step: #' + str(s + 1) + '/' + str(steps_per_epoch))
                # zero the gradient buffers
                self.optimizer.zero_grad()

                # get minibatch from data generator
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
        path = self.save_path.joinpath(self.model_name + '_' + timestamp_str +
                                       '.pth')
        self.save_model(path)

    def predict(self, data_df, beam_size):
        # TODO: implement this function
        pass

    def beam_search(self, image, beam_size):
        # TODO: implement this function
        pass

    def load_model(self, path):
        self.model = model_switcher(self.model_name)(self.input_shape,
                                                     self.vocab_size,
                                                     embedding_size=
                                                     self.embedding_size,
                                                     seed=self.random_seed)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def get_model(self):
        return self.model

    def set_model(self, model):
        # expects a pytorch model
        self.model = model

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, string):
        assert isinstance(string, str)
        self.model_name = string
