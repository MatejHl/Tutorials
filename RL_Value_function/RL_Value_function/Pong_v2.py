import tensorflow as tf
import gym
import time
import os
from math import ceil

from Pong_v2_model import *

# tf.config.experimental.list_physical_devices('GPU')
# tf.config.set_soft_device_placement(True) # Let tensorflow to choose device
tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model_path = os.path.join('models_v2', 'Q_fun')
env = gym.make("Pong-v0")
restore = True
render = False
n_frame_window = 4
eps = 0.1
eps_decay_freq = 2000
eps_decay = 1/3
Q_fixed_update_rate = 15
learning_rate = 0.1
batch_size = 2000
n_epochs = 1700
gamma = 0.95
max_buffer_size = 17000

def get_possible_actions():
    # print(env.unwrapped.get_action_meanings())
    possible_actions = np.array([2,3])
    n_possible_actions = len(possible_actions)
    return possible_actions, n_possible_actions

possible_actions, n_possible_actions = get_possible_actions()

Q_fun = Q_function(n_possible_actions, n_frame_window)
fixed_Q_fun = Q_function(n_possible_actions, n_frame_window)
opt = tf.optimizers.Adam(learning_rate=learning_rate)

# Initialize game:
current_state, replay_buffer = init_game(n_frame_window, env)

# Save model graph and init logging:
variables_path = os.path.join(model_path, 'variables', 'variables')
if restore:
    Q_fun.load_weights(variables_path)
    fixed_Q_fun.load_weights(variables_path)
else:
    expand = tf.expand_dims(current_state, axis=0, name=None)
    Q_fun._set_inputs(expand)
    Q_fun.save(model_path)
writer = tf.summary.create_file_writer(os.path.join(model_path, 'log'))

print(Q_fun.weights)
raise Exception()

game_no = 0
train_step = 0

with writer.as_default():
  with tf.summary.record_if(True):
        for epoch in range(n_epochs):
            # Decay eps:
            if epoch > 0 and epoch%eps_decay_freq == 0:
                eps = eps*eps_decay
                print(eps)
            
            # Update replay_buffer:
            start_play = time.time()
            i = 0
            done = False
            while not done:
                """
                Ak hra skonci tak namiesto posledneho frame sa ako "new_frame" ulozi env.reset() 
                a current_state bude mat n-1 framov z predoslej hry a init frame s novej.
                """
                if batch_size is not None:
                    if i > batch_size:
                        break
                action = eps_greedy_policy(current_state, Q_fun, eps, possible_actions)
                observation, reward, done, info = env.step(action)
                if render: env.render()
                if reward in [-1,1]: 
                    tf.summary.scalar('reward', reward, step = game_no)
                    game_no += 1
                if reward == 1 : print('Yeah ...')
                if done: 
                    observation = env.reset()
                old_frame = current_state[:,:,-1]
                new_frame = prepro(observation)
                replay_buffer.append((old_frame, action, reward, new_frame))
                new_frame = np.expand_dims(new_frame, 2)
                current_state = np.concatenate([current_state[:, :, 1:], new_frame], axis = -1) #dim: [height, width, num_of_frames]
                i += 1
            run_batch_size = i
            train_batch_size = ceil(run_batch_size/10) # train batch is batch_size/10
            print('\n batch_size: {}'.format(run_batch_size))
            end_play = time.time()
            print('TIME: play: {}'.format(end_play-start_play))
        
            # Check size of replay_buffer:
            if len(replay_buffer) > max_buffer_size:
                del replay_buffer[:len(replay_buffer) - max_buffer_size]
        
        
            # Update fixed model:
            if epoch%Q_fixed_update_rate == 0:
                Q_fun.save_weights(variables_path)
                fixed_Q_fun.load_weights(variables_path)
                def apply_TD_transform(state, reward, new_state):
                    return state, TD_transform_sample(reward, new_state, gamma, n_possible_actions, fixed_Q_fun)
        
            # Get samples from replay_buffer and transform them:
            start_sampling = time.time()
            train_dataset = get_samples_from_buffer(n_frame_window, run_batch_size, replay_buffer)
            train_dataset = train_dataset.map(apply_TD_transform)
            train_dataset = train_dataset.batch(train_batch_size)
            end_sampling = time.time()
            print('TIME sampling: {}'.format(end_sampling - start_sampling))
        
            # Train Q_fun model
            start_training = time.time()
            for x_batch, y_batch in train_dataset:
                train(loss, Q_fun, opt, x_batch, y_batch)
                loss_values = loss(Q_fun, x_batch, y_batch)
                tf.summary.scalar('loss', loss_values, step = train_step)
                train_step += 1
            end_training = time.time()
            print('TIME training: {}'.format(end_training - start_training))