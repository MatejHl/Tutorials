import os
import tensorflow as tf

from Test_train import training_features


# BUG: See Test_Train for INFO
# loaded_autoencoder = tf.saved_model.load(os.path.join("tmp","models","1"))
# print(loaded_autoencoder.encoder(training_features))