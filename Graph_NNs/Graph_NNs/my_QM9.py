import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # This supress debugging information so UNcoment when debugging ! ! !
import tensorflow as tf
import spektral
import networkx as nx
import pandas
import numpy

import datetime

from matplotlib import pyplot as plt

# Preprocessing:
# If done in TensorFlow then delete this.
from spektral.utils import label_to_one_hot
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
      print(spektral.datasets.qm9.NODE_FEATURES)
      print(spektral.datasets.qm9.EDGE_FEATURES)
      
      # H atoms are implicit:
      A, X, E, y = spektral.datasets.qm9.load_data(nf_keys = 'atomic_num',
                                                  ef_keys = 'type',
                                                  auto_pad = True, 
                                                  self_loops = True, 
                                                  amount = 10000,
                                                  return_type = 'numpy')
      y = y[['cv']].values  # Heat capacity at 298.15K
      
      # Preprocessing
      X_uniq = numpy.unique(X)
      X_uniq = X_uniq[X_uniq != 0]
      E_uniq = numpy.unique(E)
      E_uniq = E_uniq[E_uniq != 0]
      
      X = label_to_one_hot(X, X_uniq)
      E = label_to_one_hot(E, E_uniq)
      
      # Parameters
      N = X.shape[-2]       # Number of nodes in the graphs
      F = X[0].shape[-1]    # Dimension of node features
      S = E[0].shape[-1]    # Dimension of edge features
      n_out = y.shape[-1]   # Dimension of the target
      
      # Train/test split
      A_train, A_test, \
      X_train, X_test, \
      E_train, E_test, \
      y_train, y_test = train_test_split(A, X, E, y, test_size=0.1, random_state=0)
      
      # This can be done in tensorflow instead
      # See  https://towardsdatascience.com/building-a-one-hot-encoding-layer-with-tensorflow-f907d686bf39
      
      learning_rate = 0.001
      batch_size = 200
      epochs = 200
      log_dir = os.path.join('tf_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
      
      X_in = tf.keras.Input(shape=(N, F))
      A_in = tf.keras.Input(shape=(N, N))
      E_in = tf.keras.Input(shape=(N, N, S))
      
      X_1 = spektral.layers.EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
      X_2 = spektral.layers.EdgeConditionedConv(32, activation='relu')([X_1, A_in, E_in])
      X_3 = spektral.layers.EdgeConditionedConv(32, activation='relu')([X_1 + X_2, A_in, E_in])
      # X_3 = spektral.layers.GlobalSumPool()(X_2)
      X_4 = spektral.layers.GlobalAttnSumPool()(X_3)
      output = tf.keras.layers.Dense(n_out)(X_4)
      
      # Build model
      model = tf.keras.Model(inputs=[X_in, A_in, E_in], outputs=output)
      optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
      model.compile(optimizer=optimizer, 
                        loss='mse',
                        metrics = [tf.keras.metrics.MeanAbsoluteError()])
      model.summary()
      
      print(X_train.shape)
      
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



      model.fit([X_train, A_train, E_train],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data = ([X_test, A_test, E_test], y_test),
            callbacks=[tensorboard_callback])
      
      ################################################################################
      # EVALUATE MODEL
      ################################################################################
      print('Testing model')
      model_loss = model.evaluate([X_test, A_test, E_test],
                                  y_test,
                                  batch_size=batch_size)
      print('Done. Test loss: {}'.format(model_loss))