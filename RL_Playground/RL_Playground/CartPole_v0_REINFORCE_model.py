import tensorflow as tf


class CartPole_v0_model(tf.keras.Model):
    """
    """
    def __init__(self):
      super(CartPole_v0_model, self).__init__()
      self.input_dense = tf.keras.layers.Dense(units = 32,
                                              activation = tf.nn.relu,
                                              use_bias = True,
                                              input_dim = 4)
      self.dense_out = tf.keras.layers.Dense(units = 2,
                                           activation = tf.nn.softmax,
                                           use_bias = False)
    @tf.function
    def call(self, input_features):
        d_input = self.input_dense(input_features)
        d_out = self.dense_out(d_input)
        return d_out
