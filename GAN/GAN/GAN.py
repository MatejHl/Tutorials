import numpy as np
from math import floor
import os
from GAN_model import *
from datetime import datetime


def prepro_dataset(images):
    """
    """
    images = (images - np.mean(images)) / np.std(images)
    # images = images / np.max(images)
    images = images.astype('float32')
    return images



learning_rate = 0.00001
half_batch_size = 104
epochs = 30000
# original_batch_size = floor(batch_size/2)
# gen_batch_size = floor(batch_size/2)
noise_shape = 50
lam = tf.constant(100.0)
eps_disc = 0.00001
eps_gen = 0.00001
try_disc_mod = 10
try_gen_mod =  10
disc_restore = True
gen_restore = True

if __name__ == '__main__':    
    mnist = tf.keras.datasets.mnist
    (train_images, _), (test_images, _) = mnist.load_data()
    
    train_images = prepro_dataset(train_images)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images) 
    
    train_dataset = train_dataset.shuffle(train_images.shape[0])
    train_dataset = train_dataset.batch(half_batch_size)
    train_dataset = train_dataset.prefetch(half_batch_size * 8)
    
    # Init:
    image_shape = train_images[0].shape
    
    discriminator = Discriminator(image_shape)
    generator = Generator(noise_shape, image_shape)
    
    opt_disc = tf.optimizers.Adam(learning_rate=learning_rate)
    opt_gen = tf.optimizers.Adam(learning_rate=learning_rate)
    
    if disc_restore or gen_restore:
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
    disc_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=discriminator)
    disc_ckpt_manager = tf.train.CheckpointManager(disc_ckpt, os.path.join('tf_logs', 'tf_ckpts_disc'), max_to_keep=10)
    gen_ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=generator)
    gen_ckpt_manager = tf.train.CheckpointManager(gen_ckpt, os.path.join('tf_logs', 'tf_ckpts_gen'), max_to_keep=10)
    
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
        print("Initializing Generator from scratch.")
    
    # Func to train Discriminator
    @tf.function
    def training_Discriminator(batch_original_x, batch_gen_x):
        loss_disc = train_discriminator(loss = loss_discriminator_GP_0_line, 
                            Discriminator_model = discriminator, 
                            opt = opt_disc, 
                            batch_original_x = batch_original_x, 
                            batch_gen_x = batch_gen_x,
                            lam = lam)
        return loss_disc
    
    # Func to train Generator
    @tf.function
    def training_Generator(noise_batch_x):
        loss_gen = train_generator(loss = loss_generator_cross_entropy, 
                        Generator_model = generator, 
                        opt = opt_gen, 
                        noise_batch_x = noise_batch_x, 
                        Discriminator_model = discriminator)
        return loss_gen
    
    
    mean_loss_disc = 1.0
    mean_loss_gen = 1.0
    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                print('epoch:{} (ckpt_step:{})'.format(epoch, disc_ckpt.step.numpy()))
                if mean_loss_disc >= eps_disc or epoch % try_disc_mod == 0:
                    mean_loss_disc = 0.0
                    disc_steps_n = 0
                    for disc_step, batch_original_x in enumerate(train_dataset):
                        input_noise = tf.random.normal(shape = (batch_original_x.shape[0], noise_shape),
                                                       mean=0.0,
                                                       stddev=1.0,
                                                       dtype=tf.dtypes.float32,
                                                       seed=None,
                                                       name='input_noise')
                        batch_gen_x = generator(input_noise)
                        loss_disc = training_Discriminator(batch_original_x, batch_gen_x)
                        tf.summary.scalar('loss_disc', loss_disc, step=disc_step)
                        mean_loss_disc += loss_disc
                        disc_steps_n +=1
                    mean_loss_disc = mean_loss_disc/disc_steps_n
                    print('mean_loss_disc: {}'.format(mean_loss_disc))
                
                if mean_loss_gen >= eps_gen or epoch % try_gen_mod == 0:
                    mean_loss_gen = 0.0
                    for gen_step in range(disc_steps_n):
                        noise_batch_x = tf.random.normal(shape = (half_batch_size*2, noise_shape),
                                                     mean=0.0,
                                                     stddev=1.0,
                                                     dtype=tf.dtypes.float32,
                                                     seed=None,
                                                     name='noise_batch_x')
                        loss_gen = training_Generator(noise_batch_x)
                        tf.summary.scalar('loss_gen', loss_gen, step=gen_step) #gen_ckpt.step gives tensor with current step
                        mean_loss_gen += loss_gen
                    mean_loss_gen = mean_loss_gen/disc_steps_n
                    print('mean_loss_gen: {}'.format(mean_loss_gen))
    
                # Save checkpoint after k epochs:
                disc_ckpt.step.assign_add(1)
                gen_ckpt.step.assign_add(1)
                if epoch % 50 == 0:
                  disc_save_path = disc_ckpt_manager.save()
                  gen_save_path = gen_ckpt_manager.save()
                  print("Saved checkpoint for step {}: {}".format(int(disc_ckpt.step), disc_save_path))
                  print("Saved checkpoint for step {}: {}".format(int(gen_ckpt.step), gen_save_path))
                  img_in_noise = tf.random.normal(shape = (1, noise_shape),
                         mean=0.0,
                         stddev=1.0,
                         dtype=tf.dtypes.float32,
                         seed=None,
                         name='img_in_noise')
                  img = tf.expand_dims(input = generator(img_in_noise), 
                                 axis = -1, name=None)
                  tf.summary.image("Generated data", img, step=gen_ckpt.step.numpy())
    
    
    # backend.clear_session()
    
    # if epoch == 0:
    #     tf.summary.trace_on(graph=True, profiler=False)
    # if epoch == 0:
    #     print('exporting  model')
    #     tf.summary.trace_export(name="model_trace", step=epoch, profiler_outdir=os.path.join('tf_logs', start_time))