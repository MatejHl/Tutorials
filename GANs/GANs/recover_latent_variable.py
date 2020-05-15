import tensorflow as tf

""" 
Intuitive approcah:
"""



def SGD_step(alpha, input, Generator_model):
    def SGD_step_inner(z):
        with tf.GradientTape() as tape:
            tape.watch(z)
            diff = tf.reshape(input - Generator_model(z), [-1])
            obj_func = tf.tensordot(diff, diff, axes=1)
        grad = tape.gradient(obj_func, z)
        return tf.math.scalar_mul(alpha, grad, name='SGD_step')
    return SGD_step_inner

@tf.function
def recover_latent_var(input, Generator_model, z = None, alpha = None):
    """
    Intuitive approach to get latent space represatation for given input.

    z* = argmin_{z} \|input - f(z)\|^2
    """
    sgd_step = SGD_step(alpha, input, Generator_model)
    
    z_opt = tf.while_loop(cond = lambda z: tf.less(0.01, tf.norm(sgd_step(z))) ,
                          body = lambda z: (tf.subtract(z, sgd_step(z)), ),
                          loop_vars = [z],
                          maximum_iterations = 1000)

    return z_opt


if __name__=='__main__':
    import os
    from GAN_model import *
    from GAN import prepro_dataset, noise_shape

    # Data to init. (For future, needs to save image_shape and noise_shape)
    mnist = tf.keras.datasets.mnist
    (train_images, _), (test_images, _) = mnist.load_data()
    train_images = prepro_dataset(train_images)
    
    # Init:
    image_shape = train_images[0].shape
    generator = Generator(noise_shape, image_shape)

    # Restore Generator parameters:
    gen_ckpt = tf.train.Checkpoint(model=generator)
    gen_ckpt_manager = tf.train.CheckpointManager(gen_ckpt, os.path.join('tf_logs', 'tf_ckpts_gen'), max_to_keep=10)
    gen_ckpt.restore(gen_ckpt_manager.latest_checkpoint)
    if gen_ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(gen_ckpt_manager.latest_checkpoint))

    input_noise = 2.0*tf.ones(shape = (1, noise_shape))
    img = generator(input_noise)

    recovered_noise = recover_latent_var(input = img, 
                                         Generator_model = generator, 
                                         z = 0.5*tf.ones(shape = (1, noise_shape)),
                                         alpha = tf.constant(0.001))

    print(input_noise)
    print(recovered_noise)
