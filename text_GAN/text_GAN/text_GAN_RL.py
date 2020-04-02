import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
import json

from text_GAN_model import *
from REINFORCE import *


def preprocess_text(sentence, begin_char, end_char):
    return begin_char + sentence + end_char

def process_json_file(file_path):
    with open(file_path, 'r') as json_file:
        j = json.load(json_file)
    sentences = []
    for row in j:
        sentences += [inner_row.get('text').lower() for inner_row in row.get('utterances') if inner_row.get('speaker') == 'ASSISTANT']
    return sentences



PADDING_CHAR = '_'
BEGIN_CHAR = '<'
END_CHAR = '>'
PADDING_CONST = 500
LAM = tf.constant(100.0)
GAMMA = tf.constant(0.95)
NUM_MC_ROLLOUTS = 5
# REINFORCE_UPDATE_MOD = 2
BATCH_SIZE = 32
BUFFER_SIZE = 10000
gen_EMBEDDING_DIM = 32
gen_RNN_UNITS = 128
gen_OPT = tf.keras.optimizers.Adam()
gen_RESTORE = True
disc_EMBEDDING_DIM = 32
disc_RNN_UNITS = 128
disc_OPT = tf.keras.optimizers.Adam()
disc_RESTORE = True
N_EPOCHS = 10


# Data from https://github.com/google-research-datasets/Taskmaster
data = []
data_path = os.path.join("..", "..", "DATASETS", "Taskmaster-master", "TM-2-2020", "data")
for json_file in os.listdir(data_path):
    data += process_json_file(os.path.join(data_path, json_file))

vocab = [PADDING_CHAR, BEGIN_CHAR, END_CHAR] + sorted(set(''.join(data))) # TO DO - clean from unnecessary chars - now testing with all
vocab_size = len(vocab)
print('vocab: {} unique characters'.format(vocab_size))

data_n = len(data)

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

processed_data = [preprocess_text(sentence, BEGIN_CHAR, END_CHAR).encode('utf-8') for sentence in data]

sentences_dataset = tf.data.Dataset.from_tensor_slices(processed_data)
char2idx_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([key.encode() for key in char2idx.keys()], dtype=tf.string),
            values=tf.constant([val for val in char2idx.values()], dtype=tf.int32)),
        default_value=tf.constant(-1),
        name="char2idx")
dataset = sentences_dataset.map(lambda x: char2idx_table.lookup(tf.strings.unicode_split(x, input_encoding="UTF-8")))
dataset = dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE//2, padded_shapes = [PADDING_CONST], padding_values=0, drop_remainder=False)

generator = RNN_generator(vocab_size, gen_EMBEDDING_DIM, gen_RNN_UNITS)
discriminator = RNN_discriminator(vocab_size, disc_EMBEDDING_DIM, disc_RNN_UNITS, BATCH_SIZE)

begin_char_id = tf.reshape(tf.constant(char2idx[BEGIN_CHAR], dtype = tf.int32), shape = (1,))
end_char_id = tf.constant(char2idx[END_CHAR], dtype = tf.int32)

REINFORCE_loop_size = tf.constant(data_n//BATCH_SIZE, dtype = tf.int32)

def tf_REINFORCE_loop_while_body(i, Generator_model, Discriminator_model, gamma, optimizer, num_trajectories, begin_char_id, end_char_id, padding_shape):
    """
    tf.while_loop created because of: warning-large-unrolled-loop-detected
    """
    tf_REINFORCE_step(Generator_model = Generator_model, 
                      Discriminator_model = Discriminator_model, 
                      gamma = gamma, 
                      optimizer = optimizer,
                      num_trajectories = num_trajectories, 
                      begin_char_id = begin_char_id, 
                      end_char_id = end_char_id,
                      padding_shape = padding_shape)
    return (i+1, )

@tf.function
def _train_step():           
        for _, batch_seq in enumerate(dataset):
            trajectories = tf_MC_rollouts(start_seq = begin_char_id, 
                                          num_trajectories = BATCH_SIZE//2, 
                                          Generator_model = generator, 
                                          end_char_id = end_char_id,
                                          maximum_iterations = PADDING_CONST)
            
            trjs = tf.ragged.stack(trajectories)

            padded_trajectories = trjs.to_tensor(default_value=0, name=None, shape = (BATCH_SIZE//2, PADDING_CONST))
            
            loss_disc = train_discriminator(loss = loss_discriminator_GP_0_line, 
                    Discriminator_model = discriminator, 
                    opt = disc_OPT, 
                    batch_original_x = batch_seq,
                    batch_gen_x = padded_trajectories,
                    lam = LAM)

        i = tf.constant(0)
        tf.while_loop(cond = lambda i: tf.math.less(i, REINFORCE_loop_size),
                      body = lambda i: tf_REINFORCE_loop_while_body(i = i,
                                                         Generator_model = generator, 
                                                         Discriminator_model = discriminator, 
                                                         gamma = GAMMA, 
                                                         optimizer = gen_OPT,
                                                         num_trajectories = NUM_MC_ROLLOUTS, 
                                                         begin_char_id = begin_char_id, 
                                                         end_char_id = end_char_id,
                                                         padding_shape = (BATCH_SIZE//2, PADDING_CONST)),
                      loop_vars = [i],
                      name = 'while_REINFORCE_epochs')

        return None

   
if disc_RESTORE or gen_RESTORE:
    recent_time = datetime.min
    for filename in os.listdir("tf_logs"):
        try:
            tim = datetime.strptime(filename, "%Y_%m_%d__%H_%M_%S")
        except:
            continue
        if tim >= recent_time: 
            recent_time = tim
    start_time = recent_time.strftime("%Y_%m_%d__%H_%M_%S")
else:
    start_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
logdir = os.path.join('tf_logs', start_time)
writer = tf.summary.create_file_writer(logdir)
disc_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=discriminator)
disc_ckpt_manager = tf.train.CheckpointManager(disc_ckpt, os.path.join('tf_logs', 'tf_ckpts_disc'), max_to_keep=10)
gen_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=generator)
gen_ckpt_manager = tf.train.CheckpointManager(gen_ckpt, os.path.join('tf_logs', 'tf_ckpts_gen'), max_to_keep=10)
    
# Restore Discriminator parameters:
if disc_RESTORE:
    disc_ckpt.restore(disc_ckpt_manager.latest_checkpoint)
    if disc_ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(disc_ckpt_manager.latest_checkpoint))
else:
    print("Initializing Discriminator from scratch.")

# Restore Generator parameters:
if gen_RESTORE:
    gen_ckpt.restore(gen_ckpt_manager.latest_checkpoint)
    if gen_ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(gen_ckpt_manager.latest_checkpoint))
else:
    print("Initializing Generator from scratch.")


test = gen_text(BEGIN_CHAR, generator, END_CHAR, idx2char, char2idx, return_text = True)
print(test)

with writer.as_default():
    for epoch in range(N_EPOCHS):
        if epoch == 0:
            tf.summary.trace_on(graph=True, profiler=True)
            _train_step()
            tf.summary.trace_export(
                name="trace",
                step=0,
                profiler_outdir=logdir)
        else:
            _train_step()

        # Save checkpoint after k epochs:
        disc_ckpt.step.assign_add(1)
        gen_ckpt.step.assign_add(1)
        # if epoch % 2 == 0:
        disc_save_path = disc_ckpt_manager.save()
        gen_save_path = gen_ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(disc_ckpt.step), disc_save_path))
        print("Saved checkpoint for step {}: {}".format(int(gen_ckpt.step), gen_save_path))
        generated_text = gen_text(BEGIN_CHAR, generator, END_CHAR, idx2char, char2idx, return_text = True)
        print(generated_text)
        tf.summary.text(name = 'Generated_text', 
                        data = generated_text[0], 
                        step = gen_ckpt.step, 
                        description=None)