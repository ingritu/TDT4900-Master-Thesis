import torch
from torch import nn
from torch import optim
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
from copy import deepcopy

from src.data.data_generator import data_generator
from src.data.data_generator import pad_sequences
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
        self.max_length = 0  # initialize as 0

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
        # initialize max_length of training set
        self.max_length = max([len(c.split())
                               for c in set(train_df.loc[:, 'clean_caption'])])

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

    def predict(self, data_path, beam_size=1):
        """
        Function to make self.model make predictions given some data.

        Parameters
        ----------
        data_path : Path or str.
            Path to csv file containing the test set.
        beam_size : int.
            Default is 1, which is the same as doing greedy inference.

        Returns
        -------
        predicted_captions : dict.
            Dictionary image_name as keys and predicted captions through
            beam search are the values.
        """
        data_path = Path(data_path)
        data_df = pd.read_csv(data_path)
        predicted_captions = {}
        images = data_df.loc[:, 'image_id']
        for image_name in images:
            image = self.encoded_features[image_name]
            predictions = self.beam_search(image, beam_size=beam_size)
            predicted_captions[image_name] = predictions
        return predicted_captions


    def beam_search(self, image, beam_size):
        # TODO: implement this function
        # initialization
        image = torch.tensor(image)  # convert to tensor
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
                sequence = torch.tensor(sequence)  # convert to tensor
                # pad sequence
                sequence = pad_sequences([sequence], maxlen=self.max_length)
                
                # get predictions
                y_predictions = self.model([[image], sequence])
                # get the b most probable indices
                words_predicted = np.argsort(y_predictions)[-beam_size:]
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
            captions = captions[-beam_size:]

        return captions

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
