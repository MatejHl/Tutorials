"Tesorflow 2.0:"
import tensorflow as tf

'''
https://towardsdatascience.com/implementing-an-autoencoder-in-tensorflow-2-0-5e86126e9f7
'''

class Encoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim):
    super(Encoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu,
      kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.sigmoid
    )
    
  def call(self, input_features):
    x_hidden = self.hidden_layer(input_features)
    return self.output_layer(x_hidden)



class Decoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim):
    super(Decoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu,
      kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=original_dim,
      activation=tf.nn.sigmoid
    )
  
  def call(self, code):
    z_hidden = self.hidden_layer(code)
    return self.output_layer(z_hidden)


class Autoencoder(tf.keras.Model):
    '''
    Autoencoder model.
    '''
    def __init__(self, intermediate_dim, original_dim):
      super(Autoencoder, self).__init__()
      self.encoder = Encoder(intermediate_dim=intermediate_dim)
      self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
    
    def call(self, input_features):
      x_code = self.encoder(input_features)
      reconstructed = self.decoder(x_code)
      return reconstructed


def loss(model, x_original):
  reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(x_original), x_original)))
  return reconstruction_error

def train(loss, model, opt, x_original):
  with tf.GradientTape() as tape:
    gradients = tape.gradient(loss(model, x_original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)
