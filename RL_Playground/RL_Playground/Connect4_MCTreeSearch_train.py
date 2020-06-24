import os
import tensorflow as tf
from datetime import datetime

from Agents.AlphaZero import Agent
from Agents._self_play_funcs import playMatches
from Agents._memory import Memory

from Games.Connect4 import Game

from Connect4_MCTreeSearch_config import Config
from Connect4_MCTreeSearch_model import Connect4_MCTreeSearch_model

def get_ckpt_filename(logdir, date_fileformat, restore = False):
    """
    Helping function to get ckpt_filename if none is given.
    """
    if restore:
        recent_time = datetime.min  # Get most recent model:
        for filename in os.listdir(logdir):
            try:
                tim = datetime.strptime(filename, date_fileformat)
            except:
                continue
            if tim >= recent_time: 
                recent_time = tim
        ckpt_filename = recent_time.strftime(date_fileformat)
    else:
        ckpt_filename = datetime.now().strftime(date_fileformat)
    return ckpt_filename



ckpt_filename = None
RESTORE = False

if __name__ == '__main__':
    env = Game()
    config = Config()
    memory = Memory(config)
  
    date_fileformat = config.train_hparams['DATE_FILEFORMAT']
    ckpt_filename = config.train_hparams['CKPT_FILENAME']
    restore = config.train_hparams['RESTORE']
    logdir = config.train_hparams['LOGDIR']

    if ckpt_filename is None:
        ckpt_filename = get_ckpt_filename(logdir, date_fileformat, restore)
    logdir = os.path.join(logdir, ckpt_filename)

    # Player:
    model = Connect4_MCTreeSearch_model(board_shape = env.grid_shape + (1,),
                                    filter_num = config.hparams['FILTER_NUM'],
                                    kernel_size = config._resolve_hparams(config.hparams['KERNEL_SIZE'], int),
                                    relu_negative_slope = config.hparams['RELU_NEGATIVE_SLOPE'],
                                    num_residual_layers = config.hparams['NUM_RESIDUAL_LAYERS'],
                                    residual_in_filter_num = config.hparams['RESIDUAL_IN_FILTER_NUM'],
                                    residual_in_kernel_size = config._resolve_hparams(config.hparams['RESIDUAL_IN_KERNEL_SIZE'], int),
                                    residual_in_relu_negative_slope = config.hparams['RESIDUAL_IN_RELU_NEGATIVE_SLOPE'],
                                    residual_out_relu_negative_slope = config.hparams['RESIDUAL_OUT_RELU_NEGATIVE_SLOPE'],
                                    n_actions = env.action_size,
                                    policy_filter_num = config.hparams['POLICY_FILTER_NUM'],
                                    policy_kernel_size = config._resolve_hparams(config.hparams['POLICY_KERNEL_SIZE'], int),
                                    policy_relu_negative_slope = config.hparams['POLICY_RELU_NEGATIVE_SLOPE'],
                                    value_filter_num = config.hparams['VALUE_FILTER_NUM'],
                                    value_kernel_size = config._resolve_hparams(config.hparams['VALUE_KERNEL_SIZE'], int),
                                    value_in_relu_negative_slope = config.hparams['VALUE_IN_RELU_NEGATIVE_SLOPE'],
                                    value_in_dense_units = config.hparams['VALUE_IN_DENSE_UNITS'],
                                    value_out_relu_negative_slope = config.hparams['VALUE_OUT_RELU_NEGATIVE_SLOPE'],
                                    name = 'player_model')
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001,
                                        beta_1 = 0.9,
                                        beta_2 = 0.999,
                                        epsilon = 1e-07,
                                        amsgrad = False,
                                        name = 'Adam')
    player = Agent(name = 'player', 
                   state_space_size = env.state_size, 
                   action_space_size = env.action_size, 
                   model = model,
                   optimizer = opt,
                   config = config)
    player_ckpt = tf.train.Checkpoint(step = tf.Variable(1, dtype=tf.int64),
                               epoch = tf.Variable(1, dtype=tf.int64),
                               version = tf.Variable(1, dtype=tf.int64),
                               model = player.model)
    player_ckpt_manager = tf.train.CheckpointManager(player_ckpt, os.path.join(logdir, 'ckpts'), max_to_keep=5)

    # Best player:
    best_model = Connect4_MCTreeSearch_model(board_shape = env.grid_shape + (1,),
                                         filter_num = config.hparams['FILTER_NUM'],
                                         kernel_size = config._resolve_hparams(config.hparams['KERNEL_SIZE'], int),
                                         relu_negative_slope = config.hparams['RELU_NEGATIVE_SLOPE'],
                                         num_residual_layers = config.hparams['NUM_RESIDUAL_LAYERS'],
                                         residual_in_filter_num = config.hparams['RESIDUAL_IN_FILTER_NUM'],
                                         residual_in_kernel_size = config._resolve_hparams(config.hparams['RESIDUAL_IN_KERNEL_SIZE'], int),
                                         residual_in_relu_negative_slope = config.hparams['RESIDUAL_IN_RELU_NEGATIVE_SLOPE'],
                                         residual_out_relu_negative_slope = config.hparams['RESIDUAL_OUT_RELU_NEGATIVE_SLOPE'],
                                         n_actions = env.action_size,
                                         policy_filter_num = config.hparams['POLICY_FILTER_NUM'],
                                         policy_kernel_size = config._resolve_hparams(config.hparams['POLICY_KERNEL_SIZE'], int),
                                         policy_relu_negative_slope = config.hparams['POLICY_RELU_NEGATIVE_SLOPE'],
                                         value_filter_num = config.hparams['VALUE_FILTER_NUM'],
                                         value_kernel_size = config._resolve_hparams(config.hparams['VALUE_KERNEL_SIZE'], int),
                                         value_in_relu_negative_slope = config.hparams['VALUE_IN_RELU_NEGATIVE_SLOPE'],
                                         value_in_dense_units = config.hparams['VALUE_IN_DENSE_UNITS'],
                                         value_out_relu_negative_slope = config.hparams['VALUE_OUT_RELU_NEGATIVE_SLOPE'],
                                         name = 'best_player_model')
    best_player = Agent(name = 'best_player', 
                        state_space_size = env.state_size, 
                        action_space_size = env.action_size, 
                        model = best_model,
                        optimizer = None,
                        config = config)
    best_player_ckpt = tf.train.Checkpoint(step = tf.Variable(1, dtype=tf.int64),
                                    epoch = tf.Variable(1, dtype=tf.int64),
                                    version = tf.Variable(1, dtype=tf.int64),
                                    model = best_player.model)
    
    if restore:
        player_ckpt.restore(player_ckpt_manager.latest_checkpoint)
        best_player_ckpt.restore(player_ckpt_manager.latest_checkpoint) # restoring from the same ckpt_manager ! !
        if ckpt_manager.latest_checkpoint: print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        save_path = player_ckpt_manager.save()
        best_player_ckpt.restore(player_ckpt_manager.latest_checkpoint)
        print("Initializing models from scratch.")

    config.save(logdir)


    for epoch in range(config.train_hparams['N_EPOCHS']):
        print('epoch:  {}'.format(epoch))
        _, memory, _ =  playMatches(best_player, 
                                    best_player, 
                                    episodes = config.train_hparams['N_EPISODES'], 
                                    goes_first = 0, 
                                    env = env,
                                    memory = memory)
        memory.clear_shortMemory()

        if len(memory.longMemory) > config.MEMORY_SIZE:
            player.train(memory.longMemory)

            # save memory HERE ! ! 


            # Evaluate which player is better
            scores, memory, points =  playMatches(player, 
                                                  best_player, 
                                                  episodes = config.train_hparams['N_EVAL_EPISODES'], 
                                                  goes_first = 0, 
                                                  env = env,
                                                  memory = None)
            if scores['player'] > scores['best_player'] * config.train_hparams['EVAL_SCORE_MULTIPLIER']:
                save_path = player_ckpt_manager.save()
                best_ckpt.restore(ckpt_manager.latest_checkpoint)
                player_ckpt.versoin.assign_add(1)


    # writer = tf.summary.create_file_writer(logdir) 