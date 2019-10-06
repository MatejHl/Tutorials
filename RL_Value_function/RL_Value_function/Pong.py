import os
import numpy as np
import pickle
import gym
import math
import tensorflow as tf 

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float)# .ravel()

#Define 2D convolutional function
def conv2d(x, W, b, strides=[1,1,1,1]):
    x = tf.nn.conv2d(x, W, strides=strides, padding='SAME', data_format='NHWC') # 'NHWC' stands for [batch_size, Height, Width, Channels]
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv_net(data, out_dim, dropout_rate, frame_window):
    '''
    Ak by bolo namiesto tf.get_variable pouzite tf.Variable, tak by to nefungovalo,
    lebo pri kazdom zavolani funkcie conv_net by sa vytvorili nove vahy. Naproti tomu
    v pripade ze uz vahy existuju tak sa pri tf.get_variable iba nacitaju a nevytvoria sa nove.

    - shape = [filter_height, filter_width, in_channels, out_channels]

    Assuming input pictures of size 80x80
    '''
    with tf.name_scope("conv_input"):
        W = tf.get_variable('c_in_W', shape=(3,3,frame_window,16),
            initializer=tf.contrib.layers.xavier_initializer()) # shape = [filter_height, filter_width, in_channels, out_channels]
        b = tf.get_variable('c_in_b', shape=(16), initializer=tf.zeros_initializer())
        conv_input = conv2d(data, W, b) # [80,80,16]
    with tf.name_scope("conv_1"):
        W = tf.get_variable('c_1_W', shape=(3,3,16,16),
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('c_1_b', shape=(16), initializer=tf.zeros_initializer())
        conv_1 = conv2d(conv_input, W, b) # [80,80,16]
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # [40,40,16]
    with tf.name_scope("conv_2"):
        W = tf.get_variable('c_2_W', shape=(3,3,16,32),
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('c_2_b', shape=(32), initializer=tf.zeros_initializer())
        conv_2 = conv2d(pool_1, W, b) # [40,40,32]
    with tf.name_scope("conv_3"):
        W = tf.get_variable('c_3_W', shape=(3,3,32,32),
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('c_3_b', shape=(32), initializer=tf.zeros_initializer())
        conv_3 = conv2d(conv_2, W, b) # [40,40,32]
        pool_3 = tf.nn.max_pool(conv_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.name_scope("conv_dense_1"):
        W = tf.get_variable('c_d1_W', shape=(20*20*32,128), 
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('c_d1_b', shape=(128), initializer=tf.zeros_initializer())
        # Flatten
        flat_conv = tf.reshape(pool_3, [-1, W.get_shape().as_list()[0]]) # [20*20*32]
        conv_dense_1 = tf.add(tf.matmul(flat_conv, W), b) # [128]
        conv_dense_1 = tf.nn.relu(conv_dense_1) # [128]
        conv_dense_1 = tf.nn.dropout(conv_dense_1, rate = dropout_rate)
    with tf.name_scope("conv_dense_output"):
        W = tf.get_variable('c_out_W', shape=(128, out_dim), 
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('c_out_b', shape=(out_dim), initializer=tf.zeros_initializer())
        conv_output = tf.add(tf.matmul(conv_dense_1, W), b) # [out_dim]
    return conv_output

# def Q(observation, action):
def Q_fun(state, number_of_actions, dropout_rate, frame_window):
    conv_out_dim = 32
    conv_output = conv_net(state, conv_out_dim, dropout_rate, frame_window)
    with tf.name_scope('Q_out'):
        W = tf.get_variable('q_out_W', shape=(conv_out_dim, number_of_actions), 
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('q_out_b', shape=(number_of_actions), initializer=tf.zeros_initializer())
        Q = tf.add(tf.matmul(conv_output, W), b) # [number_of_actions]
    return Q

def Q_train_model(y_values, Q_value, learning_rate):
    with tf.name_scope('train'):
        loss = tf.losses.mean_squared_error(y_values, Q_value)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
    return train_op

def create_graph_Q(Q_fun_model_path, number_of_actions, frame_window):
    with tf.name_scope('placeholders'):
        state = tf.placeholder(tf.float32, shape=(None, 80, 80, frame_window), name = 'state') # [batch_size, ]
        # number_of_actions = tf.placeholder(tf.int32, shape=(1, ), name = 'number_of_actions')
        dropout_rate = tf.placeholder(tf.float32, name = 'dropout_rate')
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        y_values = tf.placeholder(tf.float32, shape=(None, number_of_actions), name = 'y')
    Q_value = Q_fun(state, number_of_actions, dropout_rate, frame_window)
    train_op = Q_train_model(y_values, Q_value, learning_rate)
    train_writer = tf.summary.FileWriter(Q_fun_model_path, tf.get_default_graph())
    saver = tf.compat.v1.train.Saver()
    return (state, dropout_rate, learning_rate, y_values, Q_value, train_op, train_writer, saver)

if __name__ == '__main__':
    # ------ hyperparameters ------
    # env = gym.make('MsPacman-v0')
    env = gym.make("Pong-v0")
    learn_rate = 1e-3
    buffer_size_multiplier = 10
    gamma = 0.99 # discount factor for reward
    eps = 0.15 # 1/eps in theory, esp-greedy policy    
    Q_fun_model_path = os.path.join('models', 'Q_function', 'Q_fun') # F"/content/gdrive/My Drive/Colab Notebooks/Pong/models"
    n_epochs=3
    fixed_update_rate = 4
    restore = False
    frame_window = 4
    batch_size = 1200
    # ------------------------------
    

    graph_Q = tf.Graph()
    with graph_Q.as_default():
        state, dropout_rate, learning_rate, y_values, Q_value, train_op, train_writer, saver = create_graph_Q(Q_fun_model_path, env.action_space.n, frame_window)
        # tensorboard --logdir="models\Q_function"
    graph_fixed_Q = tf.Graph()
    with graph_fixed_Q.as_default():
        state_f, dropout_rate_f, learning_rate_f, y_values_f, Q_value_f, train_op_f, train_writer_f, saver_f = create_graph_Q(Q_fun_model_path, env.action_space.n, frame_window)
        # tensorboard --logdir="models\fixed_Q_function"

    eps_greedy_prob = np.repeat(eps,env.action_space.n, 0)
    eps_greedy_prob = np.insert(eps_greedy_prob, env.action_space.n, 1 - eps * env.action_space.n )
    if eps > 1 or eps < 0:
            raise ValueError('eps must be between 0 and 1')
    
    possible_actions = np.expand_dims(np.arange(env.action_space.n), 1)
    # batch_size = 0
    replay_buffer=[]
    # replay_buffer_sample = []
    observation = env.reset()
    init_observations = []
    # done = False
    # while not done:
    for i in range(frame_window):
        observation_old = observation
        a = env.action_space.sample()
        observation, reward, done, info = env.step(a)
        if not done:
            replay_buffer.append((prepro(observation_old), a, reward, prepro(observation)))
        else:
            observation = env.reset()
            replay_buffer.append((prepro(observation_old), a, reward, prepro(observation)))
        init_observations.append(prepro(observation))
    next_states = np.stack(init_observations, axis = -1)
    
    # replay_buffer = replay_buffer[(len(replay_buffer)-batch_size):]
    # for i in np.random.choice(len(replay_buffer), replace = False, size=batch_size):
    #                 if i >= frame_window-1:
    #                     states.append(np.stack([replay_buffer[j][0] for j in range(i-(frame_window-1), i+1)], axis = -1))
    #                     print(states[0].shape)
    #                     rewards.append(replay_buffer[i][2])
    #                     states_prime.append(np.stack([replay_buffer[j][3] for j in range(i-(frame_window-1), i+1)], axis = -1))
    # next_states = [replay_buffer_sample[i][3] for i in range(len(replay_buffer_sample))]
    print('batch_size:   ', batch_size)

    with tf.Session(graph=graph_Q) as sess_Q:
        if restore:
            saver.restore(sess_Q, Q_fun_model_path)
        else:
            sess_Q.run(tf.global_variables_initializer())
            save_path = saver.save(sess_Q, Q_fun_model_path)
        with tf.Session(graph=graph_fixed_Q) as sess_fixed_Q:
            for epoch in range(n_epochs):
                print('-------------')
                print(epoch)
                # eps_greedy_policy: -------- on batch
                '''
                Ked ma vystup z conv dimenziu 1xN a vstup do action je M, tak po nasobeni vektorom W_a bude mat scitanie tvar
                (1xN + MxN). V takomto pripade sa vektor 1XN pripocita ku kazdemu z M riadkov. => da sa do Q rovno pouzit input
                v podobe celeho vektora possible_actions, lebo s bude iba jedno. Alebo naopak sa da pouzit jedna akcia a vsetky observations
                '''
                if len(replay_buffer) >= buffer_size_multiplier*batch_size:
                    del replay_buffer[:batch_size]
                # feed_dict = {state: np.expand_dims(next_states, 3), dropout_rate: 0.0, learning_rate: learn_rate, y_values: np.zeros((1, env.action_space.n))}
                # Q_future_state_a = sess_Q.run(Q_value, feed_dict = feed_dict)
                # argmax_a = np.argmax(Q_future_state_a, axis = 1)
                # eps_greedy_action = np.random.choice(np.arange(env.action_space.n + 1), size=(batch_size,), p=eps_greedy_prob)
                # eps_greedy_action[eps_greedy_action >= env.action_space.n] = argmax_a[eps_greedy_action >= env.action_space.n]
                # Play and add to replay buffer
                for i in range(batch_size):
                    feed_dict = {state: [next_states], dropout_rate: 0.0, learning_rate: learn_rate, y_values: np.zeros((1, env.action_space.n))}
                    Q_future_state_a = sess_Q.run(Q_value, feed_dict = feed_dict)
                    argmax_a = np.argmax(Q_future_state_a, axis = 1)
                    eps_greedy_action = np.random.choice(np.arange(env.action_space.n + 1), size=(1,), p=eps_greedy_prob)
                    if eps_greedy_action >= env.action_space.n:
                        eps_greedy_action = argmax_a
                    observation, reward, done, info = env.step(eps_greedy_action)
                    env.render()
                    if not done:
                        replay_buffer.append((next_states[:,:,-1], eps_greedy_action, reward, prepro(observation) ))
                    else:
                        observation = env.reset()
                        replay_buffer.append((next_states[:,:,-1], eps_greedy_action, reward, prepro(observation) ))
                    observation_expand = np.expand_dims(prepro(observation), 2)
                    next_states = np.concatenate([next_states, observation_expand], axis = -1)
                    next_states = next_states[:, :, 1:]

                # Sample from replay_buffer:
                # replay_buffer_sample = []
                states = []
                rewards = []
                states_prime = []
                for i in np.random.choice(len(replay_buffer), replace = False, size=batch_size):
                    if i >= frame_window-1:
                        states.append(np.stack([replay_buffer[j][0] for j in range(i-(frame_window-1), i+1)], axis = -1))
                        rewards.append(replay_buffer[i][2])
                        states_prime.append(np.stack([replay_buffer[j][3] for j in range(i-(frame_window-1), i+1)], axis = -1))
                # replay_buffer_sample = [replay_buffer[i] for i in np.random.choice(len(replay_buffer), replace = False, size=batch_size)]
                # states, eps_greedy_actions, rewards, states_prime = zip(*replay_buffer_sample)
                # Generate ML standard training dataset (X, y):
                if epoch%fixed_update_rate == 0:
                    saver.restore(sess_fixed_Q, Q_fun_model_path)
                Q_future_state_a = np.empty((env.action_space.n, batch_size))
                feed_dict = {state_f: states_prime, dropout_rate_f: 0.0, learning_rate_f: learn_rate, y_values_f: np.zeros((1, env.action_space.n))}    
                Q_future_state_a = sess_fixed_Q.run(Q_value_f, feed_dict = feed_dict).flatten()
                max_a = np.max(Q_future_state_a, axis = 0) # max_a{Q(s', a, w-)}
                y = np.tile(np.array(rewards) + gamma*max_a, (env.action_space.n, 1)).T
                # Train Q
                feed_dict = {state: states, dropout_rate: 0.1, learning_rate: learn_rate, y_values: y}
                sess_Q.run(train_op, feed_dict = feed_dict)
                save_path = saver.save(sess_Q, Q_fun_model_path)