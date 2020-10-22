from spektral.layers import GraphConv

import tensorflow as tf
import numpy

from tensorflow.keras.layers import Input, Dropout

# Download dataset
from spektral.datasets import citation

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = numpy.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

A,X,y, train_mask, val_mask, test_mask = citation.load_data('cora')

N = A.shape[0]
F = X.shape[-1]
n_classes = y.shape[-1]

# Create networks
X_in = Input(shape = (F, ))
A_in = Input(shape = (N, ), sparse = True)

# Functional API 
X_1 = GraphConv(16, activation = 'relu')([X_in, A_in])
X_1 = Dropout(0.5)(X_1)
X_2 = GraphConv(n_classes, activation = 'softmax')([X_1, A_in])


print(X[0,:])

print(A.shape)
print(X.shape)
print(X_2.shape)

raise Exception('-----')

model = tf.keras.Model(inputs = [X_in, A_in], outputs = X_2)

# Preprocess Adjecency matrix (add self-loops and scale connections according to degree)
A = GraphConv.preprocess(A).astype('f4')


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

X.toarray()
A = A.astype('f4')

X = tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(X))
A = tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(A))

valid_data = ([X, A], y, val_mask)



model.fit([X, A], y,
            sample_weight = train_mask,
            validation_data = valid_data,
            batch_size = N,
            shuffle = False,
            epochs = 20)

eval_results = model.evaluate([X, A],
                                y,
                                sample_weight = test_mask,
                                batch_size = N)

print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))