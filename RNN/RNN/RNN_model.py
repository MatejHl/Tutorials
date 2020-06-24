import tensorflow as tf


class RNN_model(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
        super(RNN_model, self).__init__()
        
        self.embedding_input = tf.keras.layers.Embedding(input_dim = vocab_size, 
                                               output_dim = embedding_dim,
                                               batch_input_shape=[batch_size, None],
                                               embeddings_initializer='uniform',
                                               embeddings_regularizer=None, 
                                               activity_regularizer=None,
                                               embeddings_constraint=None, 
                                               mask_zero=False, 
                                               input_length=None)
        self.GRU = tf.keras.layers.GRU(units = rnn_units,  # dim of layer "inside GRU". Number of GRU in unrolled network is seq_length
                                       activation='tanh', 
                                       recurrent_activation='sigmoid', 
                                       use_bias=True,
                                       return_sequences=True, # Whether to return the last output in the output sequence, or the full sequence.
                                       stateful=True,
                                       kernel_initializer='glorot_uniform',
                                       recurrent_initializer='glorot_uniform')
        self.dense_out = tf.keras.layers.Dense(units = vocab_size,  # We predict probability distribution over next letter
                                           activation = None,
                                           use_bias = False)

    @tf.function
    def call(self, input_features):
        embed = self.embedding_input(input_features)
        gru = self.GRU(embed)
        d_out = self.dense_out(gru)
        return d_out


def loss(labels, logits):
  return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))


def train_step(loss, model, opt, batch_labels, batch_input):
    with tf.GradientTape() as tape:
        predictions = model(batch_input)
        loss_val = loss(batch_labels, predictions)
    grads = tape.gradient(loss_val, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val
    
