from keras import backend as K
from keras.layers import Layer


class LSTMWithVisualSentinel(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(LSTMWithVisualSentinel, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1],
                                             self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        # Be sure to call this at the end
        super(LSTMWithVisualSentinel, self).build(input_shape)

    def call(self, x, mask=None):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]
