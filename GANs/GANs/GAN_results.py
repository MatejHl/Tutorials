import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from GAN_model import *
from GAN_train import prepro_dataset, noise_shape
from recover_latent_variable import recover_latent_var


def show_image(image_matrix):
    plt.figure()
    plt.imshow(image_matrix, cmap='gray')
    plt.colorbar()
    plt.grid(False)
    plt.show()


def animate_set(img_set, fig):
    img=img_set[0]
    im=plt.imshow(img, cmap='gray')
    for i in range(1,len(img_set)):
        img=img_set[i]
        im.set_data(img)
        time.sleep(1e-1)
        fig.canvas.draw()                         # redraw the canvas


def animate(img_set):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    win = fig.canvas.manager.window
    fig.canvas.manager.window.after(100, animate_set, img_set, fig)
    plt.show()


disc_restore = False
gen_restore = True


# Data to init. (For future, needs to save image_shape and noise_shape)
mnist = tf.keras.datasets.mnist
(train_images, _), (test_images, _) = mnist.load_data()
train_images = prepro_dataset(train_images)

# Init:
image_shape = train_images[0].shape

discriminator = Discriminator(image_shape)
generator = Generator(noise_shape, image_shape)




# Restore Discriminator parameters:
if disc_restore:
    disc_ckpt = tf.train.Checkpoint(model=discriminator)
    disc_ckpt_manager = tf.train.CheckpointManager(disc_ckpt, os.path.join('tf_logs', 'tf_ckpts_disc'), max_to_keep=10)
    disc_ckpt.restore(disc_ckpt_manager.latest_checkpoint)
    if disc_ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(disc_ckpt_manager.latest_checkpoint))

# Restore Generator parameters:
if gen_restore:
    gen_ckpt = tf.train.Checkpoint(model=generator)
    gen_ckpt_manager = tf.train.CheckpointManager(gen_ckpt, os.path.join('tf_logs', 'tf_ckpts_gen'), max_to_keep=10)
    gen_ckpt.restore(gen_ckpt_manager.latest_checkpoint)
    if gen_ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(gen_ckpt_manager.latest_checkpoint))



bool_animate = False


num_8 = tf.constant(
[[-2.5654566 , -1.3136631 , -1.5747982,  -3.1907828 , -3.3250546 , -2.5843987 ,
  -3.7153816 , -2.2779257 , -1.3887918,  -1.8555608 , -2.361919  , -2.8120785 ,
  -1.9476472 , -1.8278158 , -2.3266387,  -2.3302152 , -1.3865023 , -1.8556075 ,
  -1.155669  , -1.9475249 , -1.868443 ,  -3.1469457 , -0.88471854, -3.7180696 ,
  -1.0095948 , -1.4184244 , -1.0701287,  -2.1111104 , -1.4819247 , -2.5068636 ,
  -2.4160485 , -2.719998  , -4.351657 ,  -2.5432312 , -0.96354294, -0.70940864,
   0.22343683,  0.29687405, -2.0302794,  -2.8294225 , -2.8005657 , -5.0437226 ,
  -3.070631  , -1.7803143 , -2.3363934,  -0.90363455, -1.9201237 , -0.75773895,
  -2.51243   , -2.0243647 ]])
num_6 = tf.constant(
[[-0.44338012,  1.9369183,   2.2956052 ,  1.1464255 ,  1.2084175,   2.1184032,
   2.8499804 ,  1.0780071,   1.9924209 ,  2.3835795 ,  0.1336999,   1.2748327,
   1.5602367 ,  1.3064737,   1.0471075 ,  3.1120377 ,  1.2723343,   3.1925116,
   1.9391081 ,  2.4322796,   2.2751732 ,  3.1798248 ,  2.905013 ,   0.9054816,
   2.2461197 ,  2.567008 ,   1.5049369 ,  0.70587575,  2.1910577,   2.1461346,
   0.7438786 ,  1.6991   ,   2.2295094 ,  2.5637076 ,  2.6750202,   3.4418554,
   1.8876096 ,  2.3103535,   0.48998642,  2.709138  ,  2.4655483,   1.5876253,
   0.00629795,  1.2509828,   1.5543135 ,  0.6805968 ,  2.403572 ,   2.1035345,
   1.6757075 ,  3.01126   ]])

img_in_noise_begin = tf.random.normal(shape = (1, noise_shape),
                         mean=-2.0,
                         stddev=1.0,
                         dtype=tf.dtypes.float32,
                         seed=None,
                         name='img_in_noise_begin')

img_in_noise_end = tf.random.normal(shape = (1, noise_shape),
                         mean=2.0,
                         stddev=1.0,
                         dtype=tf.dtypes.float32,
                         seed=None,
                         name='img_in_noise_end')

# img_in_noise_begin = -5.0*tf.ones(shape = (1, noise_shape))
# img_in_noise_end = 5.0*tf.ones(shape = (1, noise_shape))

# img_in_noise_begin = num_6
# img_in_noise_end = num_8

if bool_animate:
    img_set = []
    for alpha in tf.range(start=0.0, limit=1.0, delta=0.01, dtype=None, name='range'):
        img_in_noise = alpha*img_in_noise_begin + (1-alpha)*img_in_noise_end
        img = generator(img_in_noise)
        img_set.append(np.squeeze(img.numpy()))
        # show_image(np.squeeze(img.numpy()))

    animate(img_set)


input_noise = 2.0*tf.ones(shape = (1, noise_shape))
input_noise = img_in_noise_end
img = generator(input_noise)

recovered_noise = recover_latent_var(input = img, 
                                     Generator_model = generator, 
                                     z = tf.random.normal(shape = (1, noise_shape),
                                                          mean=0.0,
                                                          stddev=1.0,
                                                          dtype=tf.dtypes.float32),
                                     alpha = tf.constant(0.001))

print(input_noise)
print(recovered_noise)

show_image(np.squeeze(generator(input_noise).numpy()))
show_image(np.squeeze(generator(recovered_noise[0]).numpy()))


