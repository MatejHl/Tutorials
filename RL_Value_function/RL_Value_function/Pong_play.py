import tensorflow as tf
import numpy as np
import gym
import os
from Pong import create_graph_Q
from Pong import prepro
from Pong import get_possible_actions

env = gym.make("Pong-v0")
observation = env.reset()
done = False
frame_window = 4
Q_fun_model_path = os.path.join('_model_files', 'models', 'Q_function', 'Q_fun')
eps = 0.00

possible_actions, possible_actions_n = get_possible_actions()

print(possible_actions)
print(env.unwrapped.get_action_meanings()[possible_actions[0, 0]])
print(env.unwrapped.get_action_meanings()[possible_actions[1, 0]])
eps_greedy_prob = np.repeat(eps, possible_actions_n, 0)
eps_greedy_prob = np.insert(eps_greedy_prob, possible_actions_n, 1 - eps * possible_actions_n )

print(eps_greedy_prob)

graph_Q = tf.Graph()
with graph_Q.as_default():
    state, dropout_rate, learning_rate, y_values, Q_value, train_op, train_writer, saver = create_graph_Q(Q_fun_model_path, possible_actions_n, frame_window)

init_observations = []
for i in range(frame_window):
    observation_old = observation
    a = env.action_space.sample()
    observation, reward, done, info = env.step(a)
    init_observations.append(prepro(observation))
next_states = np.stack(init_observations, axis = -1)
    
with tf.compat.v1.Session(graph=graph_Q) as sess_Q:
    saver.restore(sess_Q, Q_fun_model_path)
    score = 0.0
    while not done:
        feed_dict = {state: [next_states], dropout_rate: 0.0, learning_rate: 0.99, y_values: np.zeros((1, possible_actions_n))}
        Q_future_state_a = sess_Q.run(Q_value, feed_dict = feed_dict)
        argmax_a = np.argmax(Q_future_state_a, axis = 1)
        eps_greedy_action = np.random.choice(np.append(possible_actions, [-1]), size=(1,), p=eps_greedy_prob)
        if eps_greedy_action == -1:
            eps_greedy_action = possible_actions[argmax_a, 0]
        print(env.unwrapped.get_action_meanings()[eps_greedy_action[0]])
        observation, reward, done, info = env.step(eps_greedy_action)
        score += reward
        env.render()

        observation_expand = np.expand_dims(prepro(observation), 2)
        next_states = np.concatenate([next_states, observation_expand], axis = -1)
        next_states = next_states[:, :, 1:]