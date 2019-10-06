import tensorflow as tf
import numpy as np
import gym
import os
from Pong import create_graph_Q
from Pong import prepro


env = gym.make("Pong-v0")
observation = env.reset()
done = False
frame_window = 4
Q_fun_model_path = os.path.join('models', 'Q_function', 'Q_fun')
eps = 0.05
eps_greedy_prob = np.repeat(eps,env.action_space.n, 0)
eps_greedy_prob = np.insert(eps_greedy_prob, env.action_space.n, 1 - eps * env.action_space.n )

graph_Q = tf.Graph()
with graph_Q.as_default():
    state, dropout_rate, learning_rate, y_values, Q_value, train_op, train_writer, saver = create_graph_Q(Q_fun_model_path, env.action_space.n, frame_window)

init_observations = []
for i in range(frame_window):
    observation_old = observation
    a = env.action_space.sample()
    observation, reward, done, info = env.step(a)
    init_observations.append(prepro(observation))
next_states = np.stack(init_observations, axis = -1)
    
with tf.Session(graph=graph_Q) as sess_Q:
    saver.restore(sess_Q, Q_fun_model_path)
    score = 0.0
    while not done:
        feed_dict = {state: [next_states], dropout_rate: 0.0, learning_rate: 0.99, y_values: np.zeros((1, env.action_space.n))}
        Q_future_state_a = sess_Q.run(Q_value, feed_dict = feed_dict)
        argmax_a = np.argmax(Q_future_state_a, axis = 1)
        eps_greedy_action = np.random.choice(np.arange(env.action_space.n + 1), size=(1,), p=eps_greedy_prob)
        if eps_greedy_action >= env.action_space.n:
            eps_greedy_action = argmax_a
        observation, reward, done, info = env.step(eps_greedy_action)
        score += reward
        env.render()
        observation_expand = np.expand_dims(prepro(observation), 2)
        next_states = np.concatenate([next_states, observation_expand], axis = -1)
        next_states = next_states[:, :, 1:]