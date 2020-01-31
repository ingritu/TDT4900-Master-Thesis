from keras.layers import Layer
from keras.layers import RNN
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K


class LSTMWithVisualSentinelCell(Layer):

    def __init__(self, units, **kwargs):
        # initialize variables
        self.units = units
        self.state_size = (self.units, self.units)
        self.lstm_cell = None
        self.x_gate = None
        self.h_gate = None
        super(LSTMWithVisualSentinelCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lstm_cell = LSTM(self.units,
                              input_shape=input_shape,
                              return_state=True)
        self.x_gate = Dense(self.units,
                            input_shape=input_shape,
                            activation='sigmoid')
        self.h_gate = Dense(self.units,
                            input_dim=self.units,
                            activation='sigmoid')
        super(LSTMWithVisualSentinelCell, self).build(input_shape)

    def call(self, inputs,
             states,
             mask=None,
             **kwargs):
        print(inputs)
        h_old, c_old = states
        ht, ct = self.lstm_cell(inputs, (h_old, c_old))
        sen_gate = K.sigmoid(self.x_gate(inputs) + self.h_gate(h_old))
        st = sen_gate * K.tanh(ct)
        return ht, [ht, ct], st


class LSTMWithVisualSentinel(RNN):

    def __init__(self, units, **kwargs):
        self.units = units
        cell = LSTMWithVisualSentinelCell(self.units, **kwargs)
        super(LSTMWithVisualSentinel, self).__init__(cell, **kwargs)

    def call(self, inputs,
             mask=None,
             training=None,
             initial_state=None,
             **kwargs):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(LSTMWithVisualSentinel, self).call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state)


class AdaptiveAttention(Layer):

    def __init__(self, hidden_size, att_dim, **kwargs):
        # initialize variables
        self.sen_affine = None
        self.sen_att = None
        self.h_affine = None
        self.h_att = None
        self.v_att = None
        self.alphas = None
        self.context_hidden = None
        self.hidden_size = hidden_size
        self.att_dim = att_dim
        super(AdaptiveAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input should be [attention vectors + sentinel vector, hidden state]
        self.sen_affine = Dense(self.hidden_size, self.hidden_size)
        self.sen_att = Dense(self.hidden_size, self.att_dim)
        self.h_affine = Dense(self.hidden_size, self.hidden_size)
        self.h_att = Dense(self.hidden_size, self.att_dim)
        self.v_att = Dense(self.hidden_size, self.att_dim)
        self.alphas = Dense(self.att_dim, 1)
        self.context_hidden = Dense(self.hidden_size, self.hidden_size)

    def call(self, inputs, **kwargs):
        # input should be [attention vectors + sentinel vector, hidden state]
        assert isinstance(inputs, list)
        pass
