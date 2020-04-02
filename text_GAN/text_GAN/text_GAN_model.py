import tensorflow as tf 


def gen_text(start_string, Generator_model, end_char, idx2char, char2idx, return_text = True):
    """
    generate text until end_char.

    Returns:
    --------
    text : str
        generated text including start_string

    length : int
        length of generated text
    """
    
    start_encode = [char2idx[s] for s in start_string]
    input_eval = start_encode
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    done = False
    Generator_model.reset_states()
    while not done:
        predictions = Generator_model(input_eval)
        # predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0] # .numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        if return_text:
            text_generated.append(idx2char[predicted_id])
        else:
            text_generated.append(predicted_id)

        if idx2char[predicted_id] == end_char:
            done = True

    if return_text:
        return (start_string + ''.join(text_generated)), len(start_encode + text_generated)
    else:
        return tf.convert_to_tensor(start_encode + text_generated, dtype=tf.float32), len(start_encode + text_generated)
    

def _tf_gen_text_while_body(input_eval, predicted_id, Generator_model):
    """
    """
    predictions = Generator_model(tf.expand_dims(input_eval, 0))
    predicted_id = tf.random.categorical(predictions, num_samples=1, dtype = tf.int32)
    predicted_id = tf.reshape(predicted_id, shape = (1,))
    input_eval = tf.concat([input_eval, predicted_id], axis = 0, name='tf_gen_text_concat')
    return (input_eval, predicted_id)

@tf.function
def tf_gen_text(start_seq, Generator_model, end_char_id, maximum_iterations):
    """
    generate indices of words in tf.while_loop

    start_seq needs to have dim (None, )
    """
    # input_eval = tf.expand_dims(start_seq, 0)
    input_eval = start_seq
    predicted_id = tf.constant([-1], name = 'predicted_id')

    Generator_model.reset_states()
    seq_generated, last_predicted_id = tf.while_loop(cond = lambda input_eval, predicted_id: tf.math.not_equal(predicted_id, end_char_id), 
                  body =  lambda input_eval, predicted_id: _tf_gen_text_while_body(input_eval, predicted_id, Generator_model), 
                  loop_vars = [input_eval, predicted_id],
                  shape_invariants=[tf.TensorShape([None]), predicted_id.get_shape()],
                  name = 'while_tf_gen_text',
                  maximum_iterations = maximum_iterations)
    return seq_generated

# @tf.function
def tf_MC_rollouts(start_seq, num_trajectories, Generator_model, end_char_id, maximum_iterations):
    """
    Monte Carlo rollouts in tensorflow friendly implementation.

    Parameters:
    -----------
    start_seq : tf.Tensor, shape = (None, )
        is sentence up to now.

    Notes:
    ------
    In future change to tf.while_loop ? ?
    """
    trajectories = []
    for i in range(num_trajectories):
        trajectory = tf_gen_text(start_seq, Generator_model, end_char_id, maximum_iterations)
        trajectories.append(trajectory)
    return trajectories


class RNN_generator(tf.keras.Model):
    """
    """
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(RNN_generator, self).__init__()
        self.embedding_input = tf.keras.layers.Embedding(input_dim = vocab_size,
                                           output_dim = embedding_dim,
                                           # batch_input_shape = (1, None),
                                           embeddings_initializer = 'uniform',
                                           embeddings_regularizer = None,
                                           activity_regularizer = None,
                                           embeddings_constraint = None,
                                           mask_zero = True,
                                           input_length = None)
        self.GRU_last = tf.keras.layers.GRU(units = rnn_units,
                                       activation = 'tanh',
                                       use_bias = True,
                                       return_sequences = False, # Whether to return the last output in the output sequence, or the full sequence.
                                       stateful = False, # See https://stackoverflow.com/questions/39681046/keras-stateful-vs-stateless-lstms
                                       kernel_initializer = 'glorot_uniform',
                                       recurrent_initializer = 'glorot_uniform')
        self.dense_out = tf.keras.layers.Dense(units = vocab_size,
                                               activation = None,  # we use tf.random.categorical to sample from distribution and this function needs logits.
                                               use_bias = False)

    @tf.function
    def call(self, input_features):
        embed = self.embedding_input(input_features)
        gru = self.GRU_last(embed)
        d_out = self.dense_out(gru)
        return d_out


class RNN_discriminator(tf.keras.Model):
    """
    """
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
        super(RNN_discriminator, self).__init__()
        self.embedding_input = tf.keras.layers.Embedding(input_dim = vocab_size,
                                           output_dim = embedding_dim,
                                           # batch_input_shape = (batch_size, None),
                                           embeddings_initializer = 'uniform',
                                           embeddings_regularizer = None,
                                           activity_regularizer = None,
                                           embeddings_constraint = None,
                                           mask_zero = True,
                                           input_length = None)
        self.GRU_last = tf.keras.layers.GRU(units = rnn_units,
                                       activation = 'tanh',
                                       use_bias = True,
                                       return_sequences = False, # Whether to return the last output in the output sequence, or the full sequence.
                                       stateful = False, # See https://stackoverflow.com/questions/39681046/keras-stateful-vs-stateless-lstms
                                       kernel_initializer = 'glorot_uniform',
                                       recurrent_initializer = 'glorot_uniform')
        # see if Bidirectional makes more sense ??
        self.dense_out = tf.keras.layers.Dense(units = 1,
                                               activation = tf.nn.sigmoid,    # in Reinforce we use mean of result. What would happen if we use logits instead ? ?
                                               use_bias = True) # Probability of original
    
    @tf.function
    def call(self, input_features):
        """
        Notes:
        ------
        return shape is (batch_size, 1)

        B E W A R E: in case of model change, loss_discriminator_GP_0_line must be changed
        """
        # if tf.keras.backend.is_sparse(input_features):
        #     embed = tf.nn.safe_embedding_lookup_sparse(embedding_weights = discriminator.embedding_input.trainable_variables, 
        #                                            sparse_ids = input_features, 
        #                                            sparse_weights=None, combiner='mean', name=None)
        # else:    
        embed = self.embedding_input(input_features)
        gru = self.GRU_last(embed)
        d_out = self.dense_out(gru)
        return d_out

@tf.function
def loss_discriminator_GP_0_line(Discriminator_model, batch_original_x, batch_gen_x, lam, alpha = None):
    """
    Binary cross-entropy loss with gradient penalty (GP-0) proposed in ref [1].

    Reference:
    ----------
    [1] Improving Generalization and Stability of Generative Adversarial Networks
        https://arxiv.org/abs/1902.03984
    """
    embed_batch_original_x = Discriminator_model.embedding_input(batch_original_x)
    embed_batch_gen_x = Discriminator_model.embedding_input(batch_gen_x)

    # embed_batch_original_x = tf.nn.safe_embedding_lookup_sparse(embedding_weights = Discriminator_model.embedding_input.trainable_variables, 
    #                                    sparse_ids = batch_original_x, 
    #                                    sparse_weights=None, combiner='mean', name='GP_0_embed_batch_original_x')
    # 
    # embed_batch_gen_x = tf.nn.safe_embedding_lookup_sparse(embedding_weights = Discriminator_model.embedding_input.trainable_variables, 
    #                                    sparse_ids = batch_gen_x, 
    #                                    sparse_weights=None, combiner='mean', name='GP_0_embed_batch_gen_x')

    embed_batch_shape = tf.shape(embed_batch_original_x)

    alpha_shape = [embed_batch_shape[0]] + [1 for _ in range(len(embed_batch_original_x.shape)-1)]

    if alpha is None:
        alpha = tf.random.uniform(shape = alpha_shape,
                                  minval = 0.0,
                                  maxval = 1.0,
                                  name = 'alpha')
    else:
        alpha = tf.reshape(tensor = alpha,
                           shape = alpha_shape,
                           name = 'alpha')


    interpolates = alpha*embed_batch_original_x + (1-alpha)*embed_batch_gen_x
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        gru = Discriminator_model.GRU_last(interpolates)
        disc_value = Discriminator_model.dense_out(gru)
    grad = tape.gradient(disc_value, interpolates)
    grad = tf.reshape(grad, shape=(embed_batch_shape[0], -1))
    grad_sq_norm = tf.math.square(tf.norm(grad, axis=-1))
    GP_0 = tf.math.scalar_mul(lam, tf.reduce_mean(grad_sq_norm), name='Gradient_penalty_0')
    BCE_loss = loss_discriminator_cross_entropy(Discriminator_model, batch_original_x, batch_gen_x)
    GP_0_loss = BCE_loss + GP_0
    return GP_0_loss


@tf.function
def loss_discriminator_cross_entropy(Discriminator_model, batch_original_x, batch_gen_x):
    """
    Binary cross entropy loss for Discriminator.
    """
    batch_shape = tf.shape(batch_original_x)
    loss_original = tf.keras.losses.binary_crossentropy(y_true = tf.ones(batch_shape[0]),
                                                        y_pred = tf.reshape(Discriminator_model(batch_original_x), shape = (batch_shape[0],)),
                                                        from_logits=False)
    loss_gen = tf.keras.losses.binary_crossentropy(y_true = tf.zeros(batch_gen_x.shape[0]),
                                                   y_pred = tf.reshape(Discriminator_model(batch_gen_x), shape = (batch_gen_x.shape[0],)) ,
                                                   from_logits=False)
    BCE_loss = tf.math.add(loss_original, loss_gen, name="disc_loss_cross_entropy")
    return BCE_loss

@tf.function
def train_discriminator(loss, Discriminator_model, opt, batch_original_x, batch_gen_x, **kwargs):
    """
    """
    with tf.GradientTape() as tape:
        loss_disc = loss(Discriminator_model, batch_original_x, batch_gen_x, **kwargs)
    grad = tape.gradient(loss_disc, Discriminator_model.trainable_variables)
    opt.apply_gradients(zip(grad, Discriminator_model.trainable_variables))
    return loss_disc