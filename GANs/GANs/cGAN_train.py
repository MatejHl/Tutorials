import os
import tensorflow as tf
from datetime import datetime


from utils import prepro_dataset
from cGAN_model import *

if __name__ == '__main__':    
    # Hyperparams:
    hparams = {'BATCH_SIZE' : 100,
               'N_EPOCHS' : 25000,
               'DISC_LEARNING_RATE' : 0.00001,
               'GEN_LEARNING_RATE' : 0.00001,
               'NOISE_SHAPE' : 50,
               'LAMBDA' : 100.0}

    disc_restore = True
    gen_restore = True

    opt_disc = tf.optimizers.Adam(learning_rate=hparams['DISC_LEARNING_RATE'])
    opt_gen = tf.optimizers.Adam(learning_rate=hparams['GEN_LEARNING_RATE'])

    # Data
    mnist = tf.keras.datasets.mnist
    (train_images, y_train), (test_images, y_test) = mnist.load_data()

    train_images = prepro_dataset(train_images)
    def get_dataset():
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, y_train)) 
        train_dataset = train_dataset.shuffle(train_images.shape[0])
        train_dataset = train_dataset.batch(hparams['BATCH_SIZE']//2)
        train_dataset = train_dataset.prefetch(hparams['BATCH_SIZE'] * 4)
        return train_dataset

    # Init:
    image_shape = train_images[0].shape
    
    discriminator = Discriminator(image_shape)
    generator = Generator(hparams['NOISE_SHAPE'], image_shape)
    
    

    start_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    logdir = os.path.join('_model_files', 'cGAN', 'tf_logs')
    if disc_restore or gen_restore:
        recent_time = datetime.min
        for filename in os.listdir(logdir):
            try:
                tim = datetime.strptime(filename, "%Y_%m_%d__%H_%M_%S")
            except:
                continue
            if tim >= recent_time: 
                recent_time = tim
        start_time = recent_time.strftime("%Y_%m_%d__%H_%M_%S")
    else:
        start_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    logdir = os.path.join(logdir, start_time)
    writer = tf.summary.create_file_writer(logdir)
    disc_ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype = tf.int64), model=discriminator)
    disc_ckpt_manager = tf.train.CheckpointManager(disc_ckpt, os.path.join(logdir, 'ckpts_disc'), max_to_keep=10)
    gen_ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype = tf.int64), model=generator)
    gen_ckpt_manager = tf.train.CheckpointManager(gen_ckpt, os.path.join(logdir, 'ckpts_gen'), max_to_keep=10)
    
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


    @tf.function
    def train_epoch(hparams):
        with writer.as_default():
            train_dataset = get_dataset()
            for step, batch_original in train_dataset.enumerate(start = 0):
                # Discriminator:
                batch_original_x, batch_label = batch_original
                input_noise = tf.random.normal(shape = (tf.shape(batch_original_x)[0], hparams['NOISE_SHAPE']),
                                               mean=0.0,
                                               stddev=1.0,
                                               dtype=tf.dtypes.float32,
                                               seed=None,
                                               name='input_noise')
                batch_gen_x = generator(input_noise, batch_label)
                loss_disc = train_discriminator(loss = loss_discriminator_GP_0_line, 
                                                Discriminator_model = discriminator, 
                                                opt = opt_disc, 
                                                batch_original_x = batch_original_x, 
                                                batch_gen_x = batch_gen_x,
                                                batch_label = batch_label,
                                                lam = hparams['LAMBDA'])
                tf.summary.scalar('loss_disc', loss_disc, step=disc_ckpt.step)
                disc_ckpt.step.assign_add(1)
                # Generator:
                noise_batch_x = tf.random.normal(shape = (tf.shape(batch_original_x)[0]*2, hparams['NOISE_SHAPE']),
                                             mean=0.0,
                                             stddev=1.0,
                                             dtype=tf.dtypes.float32,
                                             seed=None,
                                             name='noise_batch_x')
                loss_gen = train_generator(loss = loss_generator_cross_entropy, 
                                           Generator_model = generator, 
                                           opt = opt_gen, 
                                           noise_batch_x = noise_batch_x, 
                                           batch_label = tf.concat([batch_label, batch_label], axis = 0), 
                                           Discriminator_model = discriminator)
                tf.summary.scalar('loss_gen', loss_gen, step = gen_ckpt.step) #gen_ckpt.step gives tensor with current step
                if gen_ckpt.step%100 == 0:
                    label = tf.constant(8, shape = (1,))
                    img_in_noise = tf.random.normal(shape = (1, hparams['NOISE_SHAPE']),
                                                    mean=0.0,
                                                    stddev=1.0,
                                                    dtype=tf.dtypes.float32,
                                                    seed=None,
                                                    name='img_in_noise')
                    img = tf.expand_dims(input = generator(img_in_noise, label), axis = -1)
                    tf.summary.image("Generated data", img, step = gen_ckpt.step)
                gen_ckpt.step.assign_add(1)


    for epoch in range(hparams['N_EPOCHS']):
        train_epoch(hparams)
        # Save checkpoint after k epochs:
        if epoch % 50 == 0:
          disc_save_path = disc_ckpt_manager.save()
          gen_save_path = gen_ckpt_manager.save()
          print("Saved checkpoint for step {}: {}".format(int(disc_ckpt.step), disc_save_path))
          print("Saved checkpoint for step {}: {}".format(int(gen_ckpt.step), gen_save_path))

        