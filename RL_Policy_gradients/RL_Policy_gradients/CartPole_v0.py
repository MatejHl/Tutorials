import tensorflow as tf
import gym

from CartPole_v0_model import CartPole_v0_model
from REINFORCE import REINFORCE




model = CartPole_v0_model()
env = gym.make("CartPole-v0")

# REINFORCE
gamma = 0.8
optimizer = tf.optimizers.Adam(learning_rate=0.01)
REINFORCE(model, env, gamma, optimizer, n_epochs = 100000, update_mod = 100)

# + Add save model