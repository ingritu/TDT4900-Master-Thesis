import torch
from torch import nn
from torch import optim


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
                 loss_function='cross_entropy',
                 optimizer='adam',
                 lr=0.0001,
                 embedding_size=300,
                 seed=222):
        self.model = None
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # initialize loss function
        self.criterion = loss_switcher(loss_function)()

        # set up optimizer
        self.optimizer_string = optimizer
        self.optimizer = optimizer_switcher(optimizer)  # not initialized
        self.lr = lr

        # misc
        self.model_name = 'AbstractModel'
        self.random_seed = seed

    def initialize_optimizer(self):
        self.optimizer = self.optimizer(self.model.parameters(), self.lr)

    def train(self, data_path, epochs, batch_size):
        # TODO: implement this function

        steps_per_epoch = 1

        for e in range(epochs):

            for s in range(steps_per_epoch):
                # zero the gradient buffers
                self.optimizer.zero_grad()

                # get minibatch from data generator
                x = torch.rand((3, 4))  # fake data in wrong dim
                target = torch.rand((1, 4))
                # get predictions from network
                output = self.model(x)
                # get loss
                loss = self.criterion(output, target)
                print('loss', '(' + self.optimizer_string + '):', loss)

                loss.backward()
                self.optimizer.step()
                
    def predict(self, data_df, beam_size):
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
