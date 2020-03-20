import tensorflow as tf

import numpy as np
import os
import time
from datetime import datetime

from RNN_model import RNN_model, loss, train_step

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
BATCH_SIZE = 64
BUFFER_SIZE = 10000
embedding_dim = 256
rnn_units = 1024
opt = tf.keras.optimizers.Adam()
restore = False
N_EPOCHS = 1000
seq_length = 100


text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print ('Length of text: {} characters'.format(len(text)))

# The unique characters in the file
vocab = sorted(set(text))
vocab_size = len(vocab)
print ('{} unique characters'.format(len(vocab)))


# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
    
if __name__ == '__main__':    
    examples_per_epoch = len(text)//(seq_length+1)
    
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    model = RNN_model(vocab_size = vocab_size, 
                  embedding_dim = embedding_dim, 
                  rnn_units = rnn_units, 
                  batch_size = BATCH_SIZE)
    
    
    # One sample:
    for input_example_batch, target_example_batch in dataset.take(1):
      example_batch_predictions = model(input_example_batch)
    
    print(model.summary())
    
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    
    print(sampled_indices)
    
    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
    
    example_batch_loss  = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())
    
    
    
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    
    if restore:
        # get most recent log
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
    writer = tf.summary.create_file_writer(os.path.join('tf_logs', start_time))
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join('tf_logs', 'tf_ckpts'), max_to_keep=10)
    
    # Restore parameters:
    if restore:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing model from scratch.")
    
    @tf.function
    def _train_step(batch_labels, batch_input):
        return train_step(loss, model, opt, batch_labels, batch_input)
    
    
    with writer.as_default():
        # with tf.summary.record_if(True):
            for epoch in range(N_EPOCHS):
                hidden = model.reset_states()
                for (batch_n, (batch_seq, batch_labels)) in enumerate(dataset):
                    loss_val = _train_step(batch_labels, batch_seq)
                    if batch_n % 100 == 0:
                        print('Epoch: {}   Batch: {}   Loss: {}'.format(epoch, batch_n, loss_val))
    
                # Save checkpoint after k epochs:
                ckpt.epoch.assign_add(1)
                if epoch % 50 == 0:
                    save_path = ckpt_manager.save()
                    print("Saved checkpoint for epoch {}: {}".format(int(ckpt.epoch), save_path))

            