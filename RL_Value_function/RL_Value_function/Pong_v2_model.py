"Tesorflow 2.0:"
import tensorflow as tf
import numpy as np

'''
https://towardsdatascience.com/implementing-an-autoencoder-in-tensorflow-2-0-5e86126e9f7
'''

# class Dummy_layer(tf.keras.layers.Layer):
#     """
#     """
#     def __init__(self, intermediate_dim):
#       super(Dummy_layer, self).__init__()
#       self.output = tf.keras.layers.Dense(
#         units=intermediate_dim,
#         activation=tf.nn.relu,
#         kernel_initializer='he_uniform'
#       )
#       
#     def call(self, input_features):
#       x_hidden = self.hidden_layer(input_features)
#       return self.output_layer(x_hidden)


class Q_function(tf.keras.Model):
    '''
    Parameterized model of Q(s, a) function

    Notes:
    ------
    Resulting model has to be of size (n_possible_actions, None), because y has size (None, ) which is row vector [so it is (1,None)],
    if "InvalidArgumentError: Incompatible shapes" error occures, possible solution is tf.transpose
    '''
    def __init__(self, n_possible_actions, n_frame_window):
      super(Q_function, self).__init__()
      # input shape : (None, 80, 80, n_frame_window)
      self.conv_input = tf.keras.layers.Conv2D(filters = 5,
                                            kernel_size = (3, 3),
                                            strides = (2, 2),
                                            padding = 'same',
                                            data_format = 'channels_last',
                                            use_bias = True,
                                            activation = tf.nn.relu,
                                            input_shape = (80, 80, n_frame_window) )

      # input shape : (None, 40, 40, 5)
      self.conv_1 = tf.keras.layers.Conv2D(filters = 7,
                                            kernel_size = (4,4),
                                            strides = (2,2),
                                            padding = 'same',
                                            data_format = 'channels_last',
                                            use_bias = True,
                                            activation = tf.nn.relu)
      
      # input shape : (None, 20, 20, 7)
      # self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size = (2,2),
      #                                             strides = (2,2),
      #                                             padding = 'same',
      #                                             data_format = 'channels_last')

      # input shape : (None, 20, 20, 7)
      # ( Pouziva array_ops.reshape. Posibly can be done by tf.reshape() )
      self.reshape_to_dense = tf.keras.layers.Reshape(target_shape = (20*20*7,))
      
      # input shape : (None, 2800)
      self.input_dense = tf.keras.layers.Dense(units = 400,
                                              activation = tf.nn.relu,
                                              use_bias = True)
      self.input_dense_dropout = tf.keras.layers.Dropout(rate = 0.2)

      # input shape : (None, 400)
      self.dense_1 = tf.keras.layers.Dense(units = 70,
                                              activation = tf.nn.relu,
                                              use_bias = True)

      # input shape : (None, 400)
      self.output_dense = tf.keras.layers.Dense(units = n_possible_actions,
                                               activation = None,
                                               use_bias = False)




    def call(self, input_features):
      c_input = self.conv_input(input_features)
      c_1 = self.conv_1(c_input)
      # mp_1 = self.max_pool_1(c_1)
      reshape = self.reshape_to_dense(c_1)
      d_input = self.input_dense(reshape)
      d_input_dropout = self.input_dense_dropout(d_input)
      d_1 = self.dense_1(d_input_dropout)
      d_output = self.output_dense(d_1)
      return d_output


def loss(model, x_batch, y_batch):
    mse_loss = tf.reduce_mean(tf.square(tf.subtract(model(x_batch), y_batch)))
    return mse_loss


def train(loss, model, opt, x_batch, y_batch):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, x_batch, y_batch), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)



def eps_greedy_policy(current_state, Q_fun, eps, possible_actions):
    """
    (It is not generator)

    Paramters:
    ----------
    current_state : 
        one state from which

    eps : float
        probability of choosing different poslicy than argmax_a

    possible_actions : nd.array
        one dimensional array (R^n) of possible actions

    Returns:
    --------
    action : object
        action name (number) with the same type as element of possible_actions
    
    Notes:
    ------
    It is possible that expand dimensions will be needed for current_state
    """

    # fn = np.expand_dims(next_states, 0)
    expand = tf.expand_dims(current_state, axis=0, name=None)
    Q_all_actions = Q_fun(expand)
    argmax_a = np.argmax(Q_all_actions, axis = 1)
    if np.random.uniform(size=1) > eps:
        action = possible_actions[argmax_a]
    else:
        action = np.random.choice(possible_actions.flatten(), size=(1,))
    return action


def TD_transform_sample(reward, new_state, gamma, n_possible_actions, Q_fun_fixed):
    """
    Notes:
    ------
    It is possible that expand dimensions will be needed for new_state.

    np.tile asi nie je potrebne, lebo y by sa mal odcitat elementwise.
    """
    expand = tf.expand_dims(new_state, axis=0, name=None)
    max_a = tf.math.reduce_max(Q_fun_fixed(expand)) # max_a{Q(s', a, w-)}
    y = tf.add(reward, tf.math.scalar_mul(gamma, max_a))
    y_tiled = tf.squeeze(tf.tile(input=[y], multiples=[n_possible_actions], name=None))
    return y_tiled


def prepro(I):
    """ 
    prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector 
    
    """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float)# .ravel()


def init_game(n_frame_window, env):
    """
    """
    replay_buffer=[]
    observation = env.reset()
    new_frame = prepro(observation)
    for i in range(n_frame_window):
        old_frame = new_frame
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done: 
            observation = env.reset()
        new_frame = prepro(observation)
        replay_buffer.append((old_frame, action, reward, new_frame))
    current_state = np.stack([row[0] for row in replay_buffer], axis = -1)
    return current_state, replay_buffer


def get_samples_from_buffer(n_frame_window, batch_size, replay_buffer):
    """
    get samples from replay_buffer

    Returns:
    --------
    train_dataset : tf.data.Dataset

    """
    states = []
    rewards = []
    new_states = []
    for i in np.random.choice(len(replay_buffer), replace = False, size=batch_size):
        if i >= n_frame_window-1:
            states.append(np.stack([replay_buffer[j][0] for j in range(i-(n_frame_window-1), i+1)], axis = -1))
            rewards.append(replay_buffer[i][2])
            new_states.append(np.stack([replay_buffer[j][3] for j in range(i-(n_frame_window-1), i+1)], axis = -1))
    return tf.data.Dataset.from_tensor_slices((states, rewards, new_states)) 




if __name__ == '__main__':
    import gym
    n_frame_window = 4
    batch_size = 4
    env = gym.make("Pong-v0")
    current_state, replay_buffer = init_game(n_frame_window=7, env = env)
    train_dataset = get_samples_from_buffer(n_frame_window, batch_size, replay_buffer)

    