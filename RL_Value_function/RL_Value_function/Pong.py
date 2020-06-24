import os
import numpy as np
import pickle
import gym
import math
import tensorflow as tf 
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))


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
    x = tf.nn.conv2d(input=x, filters=W, strides=strides, padding='SAME', data_format='NHWC') # 'NHWC' stands for [batch_size, Height, Width, Channels]
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
    with tf.compat.v1.name_scope("conv_input"):
        W = tf.compat.v1.get_variable('c_in_W', shape=(4,4,frame_window,8),
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")) # shape = [filter_height, filter_width, in_channels, out_channels]
        b = tf.compat.v1.get_variable('c_in_b', shape=(8), initializer=tf.compat.v1.zeros_initializer())
        conv_input = conv2d(data, W, b) # [40,40,8] ... je to spravne?
    with tf.compat.v1.name_scope("conv_1"):
        W = tf.compat.v1.get_variable('c_1_W', shape=(4,4,8,8),
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b = tf.compat.v1.get_variable('c_1_b', shape=(8), initializer=tf.compat.v1.zeros_initializer())
        conv_1 = conv2d(conv_input, W, b) # [40,40,8]
        pool_1 = tf.nn.max_pool2d(input=conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # [20,20,8]
    with tf.compat.v1.name_scope("conv_2"):
        W = tf.compat.v1.get_variable('c_2_W', shape=(3,3,8,16),
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b = tf.compat.v1.get_variable('c_2_b', shape=(16), initializer=tf.compat.v1.zeros_initializer())
        conv_2 = conv2d(pool_1, W, b) # [20,20,16]
    with tf.compat.v1.name_scope("conv_3"):
        W = tf.compat.v1.get_variable('c_3_W', shape=(3,3,16,16),
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b = tf.compat.v1.get_variable('c_3_b', shape=(16), initializer=tf.compat.v1.zeros_initializer())
        conv_3 = conv2d(conv_2, W, b) # [20,20,16]
        pool_3 = tf.nn.max_pool2d(input=conv_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.compat.v1.name_scope("conv_dense_1"):
        W = tf.compat.v1.get_variable('c_d1_W', shape=(10*10*16,64), 
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b = tf.compat.v1.get_variable('c_d1_b', shape=(64), initializer=tf.compat.v1.zeros_initializer())
        # Flatten
        flat_conv = tf.reshape(pool_3, [-1, W.get_shape().as_list()[0]]) # [10*10*16]
        conv_dense_1 = tf.add(tf.matmul(flat_conv, W), b) # [64]
        conv_dense_1 = tf.nn.relu(conv_dense_1) # [64]
        conv_dense_1 = tf.nn.dropout(conv_dense_1, rate = dropout_rate)
    with tf.compat.v1.name_scope("conv_dense_output"):
        W = tf.compat.v1.get_variable('c_out_W', shape=(64, out_dim), 
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b = tf.compat.v1.get_variable('c_out_b', shape=(out_dim), initializer=tf.compat.v1.zeros_initializer())
        conv_output = tf.add(tf.matmul(conv_dense_1, W), b) # [out_dim]
    return conv_output

# def Q(observation, action):
def Q_fun(state, number_of_actions, dropout_rate, frame_window):
    conv_out_dim = 32
    conv_output = conv_net(state, conv_out_dim, dropout_rate, frame_window)
    with tf.compat.v1.name_scope('Q_out'):
        W = tf.compat.v1.get_variable('q_out_W', shape=(conv_out_dim, number_of_actions), 
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b = tf.compat.v1.get_variable('q_out_b', shape=(number_of_actions), initializer=tf.compat.v1.zeros_initializer())
        Q = tf.add(tf.matmul(conv_output, W), b) # [number_of_actions]
    return Q

def Q_train_model(y_values, Q_value, learning_rate):
    with tf.compat.v1.name_scope('train'):
        loss = tf.compat.v1.losses.mean_squared_error(y_values, Q_value)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
    return train_op

def create_graph_Q(Q_fun_model_path, number_of_actions, frame_window):
    with tf.compat.v1.name_scope('placeholders'):
        state = tf.compat.v1.placeholder(tf.float32, shape=(None, 80, 80, frame_window), name = 'state') # [batch_size, ]
        dropout_rate = tf.compat.v1.placeholder(tf.float32, name = 'dropout_rate')
        learning_rate = tf.compat.v1.placeholder(tf.float32, name = 'learning_rate')
        y_values = tf.compat.v1.placeholder(tf.float32, shape=(None, number_of_actions), name = 'y')
    Q_value = Q_fun(state, number_of_actions, dropout_rate, frame_window)
    train_op = Q_train_model(y_values, Q_value, learning_rate)
    train_writer = tf.compat.v1.summary.FileWriter(Q_fun_model_path, tf.compat.v1.get_default_graph())
    saver = tf.compat.v1.train.Saver()
    return (state, dropout_rate, learning_rate, y_values, Q_value, train_op, train_writer, saver)

def get_possible_actions():
    # print(env.unwrapped.get_action_meanings())
    possible_actions = np.expand_dims(np.array([2,3]), 1)
    possible_actions_n = len(possible_actions)
    return possible_actions, possible_actions_n
print('Init... ok')

if __name__ == '__main__':
    # ------ hyperparameters ------
    # env = gym.make('MsPacman-v0')
    # config = tf.compat.v1.ConfigProto()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # config.gpu_options.allow_growth = True

    env = gym.make("Pong-v0")
    learn_rate = 1e-3
    buffer_size_multiplier = 2
    gamma = 0.8 # discount factor for reward
    eps = 0.1 # 1/eps in theory, esp-greedy policy
    eps_decay = 1.0/3.0
    eps_decay_freq = 2000
    Q_fun_model_path = os.path.join('_model_files', 'models', 'Q_function', 'Q_fun') # F"/content/gdrive/My Drive/Colab Notebooks/Pong/models"
    # Q_fun_model_path = os.path.join('gdrive','My Drive', 'Colab Notebooks', 'Pong', 'models', 'Q_function', 'Q_fun')
    # print(os.path.abspath(Q_fun_model_path))
    n_epochs=10
    fixed_update_rate = 20
    restore = False
    render = False
    frame_window = 1
    batch_size = 1000
    possible_actions, possible_actions_n = get_possible_actions()
    print('Number of actions: {}'.format(possible_actions_n))
    # ------------------------------
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join('_model_files', 'models', 'logs', current_time, 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # ------------------------------
    

    graph_Q = tf.Graph()
    with graph_Q.as_default():
        state, dropout_rate, learning_rate, y_values, Q_value, train_op, train_writer, saver = create_graph_Q(Q_fun_model_path, possible_actions_n, frame_window)
    graph_fixed_Q = tf.Graph()
    with graph_fixed_Q.as_default():
        state_f, dropout_rate_f, learning_rate_f, y_values_f, Q_value_f, train_op_f, train_writer_f, saver_f = create_graph_Q(Q_fun_model_path, possible_actions_n, frame_window)


    eps_greedy_prob = np.repeat(eps, possible_actions_n, 0)
    eps_greedy_prob = np.insert(eps_greedy_prob, possible_actions_n, 1 - eps * possible_actions_n )
    if eps > 1 or eps < 0:
            raise ValueError('eps must be between 0 and 1')
    
    
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

    with tf.compat.v1.Session(graph=graph_Q) as sess_Q:
        if restore:
            saver.restore(sess_Q, Q_fun_model_path)
        else:
            sess_Q.run(tf.compat.v1.global_variables_initializer())
            save_path = saver.save(sess_Q, Q_fun_model_path)
        # with tf.compat.v1.Session(graph=graph_fixed_Q) as sess_fixed_Q:
        for epoch in range(n_epochs):
            if epoch%eps_decay_freq == 0 and epoch > 0:
                eps = eps*eps_decay
                print(eps)
                eps_greedy_prob = np.repeat(eps, possible_actions_n, 0)
                eps_greedy_prob = np.insert(eps_greedy_prob, possible_actions_n, 1 - eps * possible_actions_n )
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
                print(i)
                fn = np.expand_dims(next_states, 0)
                dummy_y = np.ones((3, possible_actions_n))
                print('Inputs: ')
                feed_dict = {state: fn, dropout_rate: 0.0, learning_rate: learn_rate, y_values: dummy_y}
                print(Q_value.shape)
                Q_future_state_a = sess_Q.run(Q_value, feed_dict = feed_dict)
                #
                for ele in feed_dict.values():
                    try:
                        print(ele.shape)
                    except:
                        continue
                print(Q_future_state_a.shape)
                #
                argmax_a = np.argmax(Q_future_state_a, axis = 1)
                print(argmax_a.shape)
                eps_greedy_action = np.random.choice(np.append(possible_actions, [-1]), size=(1,), p=eps_greedy_prob)
                if eps_greedy_action == -1:
                    eps_greedy_action = possible_actions[argmax_a, 0]
                observation, reward, done, info = env.step(eps_greedy_action)
                if render:
                    env.render()
                
                if reward == 1:
                    print("Point! ! !")
                if reward == 1 or reward == -1:
                    tf.summary.scalar('reward', reward, step=epoch*batch_size + i)
                if not done:
                    replay_buffer.append((next_states[:,:,-1], eps_greedy_action, reward, prepro(observation) ))
                else:
                    observation = env.reset()
                    replay_buffer.append((next_states[:,:,-1], eps_greedy_action, reward, prepro(observation) ))
                observation_expand = np.expand_dims(prepro(observation), 2)
                next_states = np.concatenate([next_states, observation_expand], axis = -1)
                next_states = next_states[:, :, 1:]

            # --------> ! ! !   ! ! !   ! ! !   ! ! ! <-------- #
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

            with tf.compat.v1.Session(graph=graph_fixed_Q) as sess_fixed_Q:
                if epoch%fixed_update_rate == 0:
                    saver.restore(sess_fixed_Q, Q_fun_model_path)
                Q_future_state_a = np.empty((possible_actions_n, batch_size))
                feed_dict = {state_f: states_prime, dropout_rate_f: 0.0, learning_rate_f: learn_rate, y_values_f: np.zeros((1, possible_actions_n))}    
                Q_future_state_a = sess_fixed_Q.run(Q_value_f, feed_dict = feed_dict).flatten()
                max_a = np.max(Q_future_state_a, axis = 0) # max_a{Q(s', a, w-)}
            y = np.tile(np.array(rewards) + gamma*max_a, (possible_actions_n, 1)).T
            # Train Q
            feed_dict = {state: states, dropout_rate: 0.7, learning_rate: learn_rate, y_values: y}
            sess_Q.run(train_op, feed_dict = feed_dict)
            save_path = saver.save(sess_Q, Q_fun_model_path)
    print(save_path)

