from Agents.HumanUser import User
from Agents.AlphaZero import Agent
from Agents._memory import Memory
from Agents._self_play_funcs import playMatches
from Games.Connect4 import Game
from Connect4_MCTreeSearch_config import Config

from Connect4_MCTreeSearch_model import Connect4_MCTreeSearch_model

human_VS_human = False
human_VS_agent = True
agent_VS_agent = False

# -----------
episodes = 3
goes_first = -1
# -----------
env = Game()
config = Config()
memory = Memory(config)


# -------- Temporary solution:
import os
import tensorflow as tf
from Connect4_MCTreeSearch_train import get_ckpt_filename

date_fileformat = config.train_hparams['DATE_FILEFORMAT']
ckpt_filename = config.train_hparams['CKPT_FILENAME']
restore = True
logdir = config.train_hparams['LOGDIR']

if ckpt_filename is None:
    ckpt_filename = get_ckpt_filename(logdir, date_fileformat, restore)
logdir = os.path.join(logdir, ckpt_filename)

# model = tf.keras.models.load_model(logdir)
model = tf.saved_model.load(logdir)

print(model.signatures)

raise Exception('AA')

# model = Connect4_MCTreeSearch_model(board_shape = env.grid_shape + (1,),
#                                          filter_num = config.hparams['FILTER_NUM'],
#                                          kernel_size = config._resolve_hparams(config.hparams['KERNEL_SIZE'], int),
#                                          relu_negative_slope = config.hparams['RELU_NEGATIVE_SLOPE'],
#                                          num_residual_layers = config.hparams['NUM_RESIDUAL_LAYERS'],
#                                          residual_in_filter_num = config.hparams['RESIDUAL_IN_FILTER_NUM'],
#                                          residual_in_kernel_size = config._resolve_hparams(config.hparams['RESIDUAL_IN_KERNEL_SIZE'], int),
#                                          residual_in_relu_negative_slope = config.hparams['RESIDUAL_IN_RELU_NEGATIVE_SLOPE'],
#                                          residual_out_relu_negative_slope = config.hparams['RESIDUAL_OUT_RELU_NEGATIVE_SLOPE'],
#                                          n_actions = env.action_size,
#                                          policy_filter_num = config.hparams['POLICY_FILTER_NUM'],
#                                          policy_kernel_size = config._resolve_hparams(config.hparams['POLICY_KERNEL_SIZE'], int),
#                                          policy_relu_negative_slope = config.hparams['POLICY_RELU_NEGATIVE_SLOPE'],
#                                          value_filter_num = config.hparams['VALUE_FILTER_NUM'],
#                                          value_kernel_size = config._resolve_hparams(config.hparams['VALUE_KERNEL_SIZE'], int),
#                                          value_in_relu_negative_slope = config.hparams['VALUE_IN_RELU_NEGATIVE_SLOPE'],
#                                          value_in_dense_units = config.hparams['VALUE_IN_DENSE_UNITS'],
#                                          value_out_relu_negative_slope = config.hparams['VALUE_OUT_RELU_NEGATIVE_SLOPE'],
#                                          name = 'best_player_model')
# ckpt = tf.train.Checkpoint(step = tf.Variable(1, dtype=tf.int64),
#                                epoch = tf.Variable(1, dtype=tf.int64),
#                                version = tf.Variable(1, dtype=tf.int64),
#                                model = model)
# ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(logdir, 'ckpts'), max_to_keep=5)
#     
# ckpt.restore(ckpt_manager.latest_checkpoint)
# if ckpt_manager.latest_checkpoint: print("Restored from {}".format(ckpt_manager.latest_checkpoint))
# ---------------------



if human_VS_human:
    player1 = User('player 1', env.state_size, env.action_size)
    player2 = User('player 2', env.state_size, env.action_size)

elif human_VS_agent:
    player1 = User('player 1', env.state_size, env.action_size)
    player2 = Agent('player 2', env.state_size, env.action_size, 
                    model = model,
                    optimizer = None,
                    config = config)
    # player2.tau = 0

elif agent_VS_agent:
    player1 = Agent('player 1', env.state_size, env.action_size, 
                    model = model,
                    optimizer = None,
                    config = config)
    player2 = Agent('player 2', env.state_size, env.action_size, 
                    model = model,
                    optimizer = None,
                    config = config)


scores, memory, points =  playMatches(player1, 
                                      player2, 
                                      episodes = episodes, 
                                      goes_first = goes_first, 
                                      env = env,
                                      memory = None,
                                      render = True)
print(scores)
print(points)