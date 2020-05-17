import tensorflow as tf
import gym

from CartPole_v0_REINFORCE_model import CartPole_v0_model
from CartPole_v0_REINFORCE_config import *
from Algorithms.REINFORCE import REINFORCE


model = CartPole_v0_model()
env = gym.make("CartPole-v0")

# REINFORCE
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
REINFORCE(model, env, GAMMA, optimizer, n_epochs = N_EPOCHS, update_mod = UPDATE_MOD)

# + Add save model
