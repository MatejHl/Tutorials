import os
import tensorflow as tf
from RNN_model import RNN_model
from RNN import *

model = RNN_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# NOTE: Better would be to use saved_model
ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join('tf_logs', 'tf_ckpts'), max_to_keep=10)

# Restore parameters:
ckpt.restore(ckpt_manager.latest_checkpoint)
if ckpt_manager.latest_checkpoint:
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))

# To run the model with a different batch_size, the model needs to be rebuild:
# def build(self, input_shape) in tensorflow.python.keras.engine.network.Network
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/network.py
model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string, num_generate = 1000, temperature = 1.0):
    """
    Low temperatures results in more predictable text.
    Higher temperatures results in more surprising text.
    Experiment to find the best setting.
    """
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    

    text_generated = []

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        print(predictions)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
    
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
    
        text_generated.append(idx2char[predicted_id])
    
    return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"ROMEO: ", temperature = 1.0))
