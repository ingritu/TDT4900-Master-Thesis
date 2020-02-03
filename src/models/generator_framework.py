class Generator():

    def __init__(self,
                 input_shape,
                 vocab_size,
                 embedding_size=300,
                 seed=222):
        self.model = None
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.model_name = 'AbstractModel'
        self.embedding_size = embedding_size
        self.random_seed = seed

    def train(self, data_df):
        # TODO: implement this function
        pass

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

