import matplotlib.pyplot as plt
import numpy as np
from math import floor
import os
from GAN_model import *


def show_image(image_matrix):
    plt.figure()
    plt.imshow(image_matrix)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def prepro_dataset(images):
    """
    """
    images = (images - np.mean(images)) / np.std(images)
    # images = images / np.max(images)
    images = images.astype('float32')
    return images


learning_rate = 0.0002
batch_size = 52
epochs = 30000
original_batch_size = floor(batch_size/2)
gen_batch_size = floor(batch_size/2)
noise_shape = 100
disc_restore = True
gen_restore = True

mnist = tf.keras.datasets.mnist
(train_images, _), (test_images, _) = mnist.load_data()

train_images = prepro_dataset(train_images)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images) 

train_dataset = train_dataset.shuffle(train_images.shape[0])
train_dataset = train_dataset.batch(original_batch_size)
train_dataset = train_dataset.prefetch(batch_size * 4)

# Init:
image_shape = train_images[0].shape

discriminator = Discriminator(image_shape)
generator = Generator(noise_shape, image_shape)

opt_disc = tf.optimizers.Adam(learning_rate=learning_rate)
opt_gen = tf.optimizers.Adam(learning_rate=learning_rate)

from datetime import datetime
start_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
writer = tf.summary.create_file_writer(os.path.join('tf_logs', start_time))
disc_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=discriminator)
disc_ckpt_manager = tf.train.CheckpointManager(disc_ckpt, os.path.join('tf_logs', 'tf_ckpts_disc'), max_to_keep=3)
gen_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=generator)
gen_ckpt_manager = tf.train.CheckpointManager(gen_ckpt, os.path.join('tf_logs', 'tf_ckpts_gen'), max_to_keep=3)

# Restore Discriminator parameters:
if disc_restore:
    disc_ckpt.restore(disc_ckpt_manager.latest_checkpoint)
    if disc_ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(disc_ckpt_manager.latest_checkpoint))
else:
    print("Initializing Discriminator from scratch.")

# Restore Generator parameters:
if gen_restore:
    gen_ckpt.restore(gen_ckpt_manager.latest_checkpoint)
    if gen_ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(gen_ckpt_manager.latest_checkpoint))
else:
    print("Initializing Discriminator from scratch.")

# Func to train Discriminator
def training_Discriminator(batch_original_x):
    input_noise = tf.random.normal(shape = (gen_batch_size, noise_shape),
                     mean=0.0,
                     stddev=1.0,
                     dtype=tf.dtypes.float32,
                     seed=None,
                     name='input_noise')
    batch_gen_x = generator(input_noise)

    train_discriminator(loss = loss_discriminator, 
                        Discriminator_model = discriminator, 
                        opt = opt_disc, 
                        batch_original_x = batch_original_x, 
                        batch_gen_x = batch_gen_x)
    loss_disc = loss_discriminator(Discriminator_model = discriminator, 
                                   batch_original_x = batch_original_x, 
                                   batch_gen_x = batch_gen_x)
    return loss_disc

# Func to train Generator
def training_Generator():
    noise_batch_x = tf.random.normal(shape = (batch_size, noise_shape),
                     mean=0.0,
                     stddev=1.0,
                     dtype=tf.dtypes.float32,
                     seed=None,
                     name='noise_batch_x')
    
    train_generator(loss = loss_generator, 
                    Generator_model = generator, 
                    opt = opt_gen, 
                    noise_batch_x = noise_batch_x, 
                    Discriminator_model = discriminator)
    loss_gen = loss_generator(Generator_model = generator, 
                              noise_batch_x = noise_batch_x, 
                              Discriminator_model = discriminator)
    return loss_gen

@tf.function
def training_GAN(batch_original_x):
    loss_disc = training_Discriminator(batch_original_x)
    loss_gen = training_Generator()
    return loss_disc, loss_gen



with writer.as_default():
    with tf.summary.record_if(True):
        for epoch in range(epochs):
            for step, batch_original_x in enumerate(train_dataset):
                
                if step == 0: # Save training graph
                    tf.summary.trace_on(graph=True, profiler=False)
                    loss_disc, loss_gen = training_GAN(batch_original_x)
                    print('exporting model')
                    tf.summary.trace_export(name="disc_model_trace", step=epoch, profiler_outdir=os.path.join('tf_logs', start_time))
                else:
                    loss_disc, loss_gen = training_GAN(batch_original_x)
                tf.summary.scalar('loss_disc', loss_disc, step=step)
                tf.summary.scalar('loss_gen', loss_gen, step=step)
                  
                if step%50 == 0:
                    print('epoch:{}   step: {}'.format(epoch, step))
                    print('loss_disc: {}'.format(loss_disc))
                    print('loss_gen: {}'.format(loss_gen))

        # Save checkpoint after k epochs:
        disc_ckpt.step.assign_add(1)
        gen_ckpt.step.assign_add(1)
        if epoch % 1 == 0:
          disc_save_path = disc_ckpt_manager.save()
          gen_save_path = gen_ckpt_manager.save()
          print("Saved checkpoint for step {}: {}".format(int(disc_ckpt.step), disc_save_path))
          print("Saved checkpoint for step {}: {}".format(int(gen_ckpt.step), gen_save_path))


# backend.clear_session()