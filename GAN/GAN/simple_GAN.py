import matplotlib.pyplot as plt
import numpy as np
from math import floor
import os
from GAN_model import *

# pareto_dist = np.random.pareto(a=2, size=10000)
# pareto_dist = pareto_dist[np.where(pareto_dist < 10)[0]]
# _ = plt.hist(pareto_dist, bins='auto')
# plt.show()



learning_rate = 0.00004
batch_size = 2048
epochs = 30000
original_batch_size = floor(batch_size/2)
gen_batch_size = floor(batch_size/2)
noise_shape = 5
eps_disc = 0.001
eps_gen = 0.001
disc_restore = True
gen_restore = True


train_pareto = np.random.pareto(a=2, size=(100000,1))
train_dataset = tf.data.Dataset.from_tensor_slices(train_pareto) 

train_dataset = train_dataset.shuffle(train_pareto.shape[0])
train_dataset = train_dataset.batch(original_batch_size)
train_dataset = train_dataset.prefetch(batch_size * 4)

# Init:

discriminator = simple_Discriminator()
generator = simple_Generator(noise_shape)

opt_disc = tf.optimizers.Adam(learning_rate=learning_rate)
opt_gen = tf.optimizers.Adam(learning_rate=learning_rate)

from datetime import datetime
start_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
writer = tf.summary.create_file_writer(os.path.join('paret_test', 'tf_logs', start_time))
disc_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=discriminator)
disc_ckpt_manager = tf.train.CheckpointManager(disc_ckpt, os.path.join('paret_test', 'tf_logs', 'tf_ckpts_disc'), max_to_keep=3)
gen_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=generator)
gen_ckpt_manager = tf.train.CheckpointManager(gen_ckpt, os.path.join('paret_test', 'tf_logs', 'tf_ckpts_gen'), max_to_keep=3)

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
@tf.function
def training_Discriminator(batch_original_x):
    input_noise = tf.random.normal(shape = (gen_batch_size, noise_shape),
                     mean=0.0,
                     stddev=1.0,
                     dtype=tf.dtypes.float32,
                     seed=None,
                     name='input_noise')
    batch_gen_x = generator(input_noise)

    train_discriminator(loss = loss_discriminator_cross_entropy, 
                        Discriminator_model = discriminator, 
                        opt = opt_disc, 
                        batch_original_x = batch_original_x, 
                        batch_gen_x = batch_gen_x)
    loss_disc = loss_discriminator_cross_entropy(Discriminator_model = discriminator, 
                                   batch_original_x = batch_original_x, 
                                   batch_gen_x = batch_gen_x)
    return loss_disc

# Func to train Generator
@tf.function
def training_Generator(noise_batch_x):    
    train_generator(loss = loss_generator_cross_entropy, 
                    Generator_model = generator, 
                    opt = opt_gen, 
                    noise_batch_x = noise_batch_x, 
                    Discriminator_model = discriminator)
    loss_gen = loss_generator_cross_entropy(Generator_model = generator, 
                              noise_batch_x = noise_batch_x, 
                              Discriminator_model = discriminator)
    return loss_gen

mean_loss_disc = 0.0
mean_loss_gen = 0.0
with writer.as_default():
    with tf.summary.record_if(True):
        for epoch in range(epochs):
            print('epoch:{}'.format(epoch))
            if mean_loss_disc >= eps_disc or epoch % 40 == 0:
                mean_loss_disc = 0.0
                disc_steps_n = 0
                for disc_step, batch_original_x in enumerate(train_dataset):

                    loss_disc = training_Discriminator(batch_original_x)
                    tf.summary.scalar('loss_disc', loss_disc, step=disc_step)
                    mean_loss_disc += loss_disc
                    disc_steps_n +=1
                mean_loss_disc = mean_loss_disc/disc_steps_n
                print('mean_loss_disc: {}'.format(mean_loss_disc))

            if mean_loss_gen >= eps_gen or epoch % 40 == 0:
                mean_loss_gen = 0.0
                for gen_step in range(disc_steps_n):
                    noise_batch_x = tf.random.normal(shape = (batch_size, noise_shape),
                         mean=0.0,
                         stddev=1.0,
                         dtype=tf.dtypes.float32,
                         seed=None,
                         name='noise_batch_x')
                    loss_gen = training_Generator(noise_batch_x)
                    tf.summary.scalar('loss_gen', loss_gen, step=gen_step)
                    mean_loss_gen += loss_gen
                mean_loss_gen = mean_loss_gen/disc_steps_n
                print('mean_loss_gen: {}'.format(mean_loss_gen))
            
            # Save checkpoint after k epochs:
            disc_ckpt.step.assign_add(1)
            gen_ckpt.step.assign_add(1)
            if epoch % 100 == 99:
              disc_save_path = disc_ckpt_manager.save()
              gen_save_path = gen_ckpt_manager.save()
              print("Saved checkpoint for step {}: {}".format(int(disc_ckpt.step), disc_save_path))
              print("Saved checkpoint for step {}: {}".format(int(gen_ckpt.step), gen_save_path))
            if epoch%1000 == 0:
                normal_dist_input = tf.random.normal(shape = (100000, noise_shape),
                     mean=0.0,
                     stddev=1.0,
                     dtype=tf.dtypes.float32,
                     seed=None,
                     name='noise_batch_x')
                plt.subplot(2, 1, 1)
                gen_dist = generator(normal_dist_input).numpy().flatten()
                gen_dist = gen_dist[gen_dist <= 3.0]
                _ = plt.hist(generator(normal_dist_input).numpy().flatten(), bins='auto')
                plt.subplot(2, 1, 2)
                pareto_dist = train_pareto[train_pareto <= 3.0]
                _ = plt.hist(pareto_dist, bins='auto')
                plt.show()


# backend.clear_session()

# if epoch == 0:
#     tf.summary.trace_on(graph=True, profiler=False)
# if epoch == 0:
#     print('exporting  model')
#     tf.summary.trace_export(name="model_trace", step=epoch, profiler_outdir=os.path.join('tf_logs', start_time))