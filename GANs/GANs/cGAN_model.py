import tensorflow as tf

class Discriminator(tf.keras.Model):
    """
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.img_shape = tf.TensorShape(img_shape)
        
        self.reshape = tf.keras.layers.Reshape(input_shape = self.img_shape,
                                               target_shape = (tf.math.reduce_prod(self.img_shape),),
                                               name = 'reshape')

        self.concat = tf.keras.layers.Concatenate(axis=-1, name = 'concat')

        self.dense_1 = tf.keras.layers.Dense(units = 512,
                                             activation = tf.nn.leaky_relu,
                                             use_bias = True,
                                             name = 'dense_1')

        self.dense_2 = tf.keras.layers.Dense(units = 256,
                                             activation = tf.nn.leaky_relu,
                                             use_bias = True,
                                             name = 'dense_2')

        self.dense_out = tf.keras.layers.Dense(units = 1,
                                               activation = tf.nn.sigmoid,
                                               use_bias = True,
                                               name = 'dense_out')

    @tf.function
    def call(self, input, label):
        label = tf.one_hot(indices = label, depth = 10)
        x = self.reshape(input)
        x = self.concat([x, label])
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_out(x)
        return x



class Generator(tf.keras.Model):
    """
    """
    def __init__(self, noise_shape, img_shape):
        super(Generator, self).__init__()

        self.noise_shape = tf.TensorShape(noise_shape)
        self.img_shape = tf.TensorShape(img_shape)

        self.concat_input = tf.keras.layers.Concatenate(axis = -1, name = 'concat')

        self.dense_input = tf.keras.layers.Dense(input_shape = (noise_shape + 1, ),
                                                 units = 256,
                                                 activation = tf.nn.leaky_relu,
                                                 use_bias = True,
                                                 name = 'dense_input')

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(momentum = 0.8,
                                                               name = 'batch_norm_1')
        self.dense_1 = tf.keras.layers.Dense(units = 512,
                                            activation = tf.nn.leaky_relu,
                                            use_bias = True,
                                            name = 'dense_1')

        self.batch_norm_2 = tf.keras.layers.BatchNormalization(momentum = 0.8,
                                                               name = 'batch_norm_2')
        self.dense_2 = tf.keras.layers.Dense(units = 1024,
                                            activation = tf.nn.leaky_relu,
                                            use_bias = True,
                                            name = 'dense_2')
        
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(momentum = 0.8,
                                                               name = 'batch_norm_3')
        self.dense_3 = tf.keras.layers.Dense(units = tf.math.reduce_prod(img_shape),
                                            activation = tf.math.tanh,
                                            use_bias = True,
                                            name = 'dense_3')

        self.reshape_to_img = tf.keras.layers.Reshape(target_shape = img_shape,
                                                      name = 'reshape_to_img')
        

    def call(self, input_noise, label):
        label = tf.one_hot(indices = label, depth = 10)
        x = self.concat_input([input_noise, label])
        x = self.dense_input(x)
        x = self.batch_norm_1(x)
        x = self.dense_1(x)
        x = self.batch_norm_2(x)
        x = self.dense_2(x)
        x = self.batch_norm_3(x)
        x = self.dense_3(x)
        x = self.reshape_to_img(x)
        return x


def loss_discriminator_cross_entropy(Discriminator_model, batch_original_x, batch_gen_x, batch_label):
    """
    Notes:
    ------
    WARNING:
    This naive implementation is numerically unstable. For more complex usecases use logits directly by using
    tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output) 
    or
    tf.keras.losses.binary_crossentropy(from_logits = True)
    which uses former under the hood.
    """
    loss_original = tf.math.reduce_sum(tf.math.log(tf.squeeze(Discriminator_model(batch_original_x, batch_label))))
    loss_gen = tf.math.reduce_sum(tf.math.log(1-tf.squeeze(Discriminator_model(batch_gen_x, batch_label))))
    BCE_loss = - tf.math.add(loss_original, loss_gen, name="disc_loss_cross_entropy")
    # WARNING: See notes above.
    return BCE_loss

def loss_discriminator_GP_0_line(Discriminator_model, batch_original_x, batch_gen_x, batch_label, lam, alpha = None):
    '''
    Binary cross-entropy loss with gradient penalty (GP-0) proposed in ref [1].

    Notes:
    ------
    This approach assumes that paths betwwen real datapoints and fake datapoints are line segments. 
    Thus assuming that all line segments from fake to real datapoint are in support.
    C = \{\alpha*x + (1-\alpha)*y| x \in supp(p_g), y \in supp(p_r), \alpha \in (0,1)\} \in supp(p_g) union supp(p_r)
    -> There is and idea in appendix of the paper how to get other paths 

    Reference:
    ----------
    [1] Improving Generalization and Stability of Generative Adversarial Networks
        https://arxiv.org/abs/1902.03984
    '''
    batch_original_x_shape = tf.shape(batch_original_x)
    alpha_shape = (batch_original_x_shape[0], ) + tuple([1 for _ in range(len(batch_original_x_shape)-1)])
    if alpha is None:
        alpha = tf.random.uniform(shape = alpha_shape,
                                  minval = 0.0,
                                  maxval = 1.0,
                                  name = 'alpha')
    else:
        alpha = tf.reshape(tensor = alpha,
                           shape = alpha_shape,
                           name = 'alpha')
    interpolates = alpha*batch_original_x + (1-alpha)*batch_gen_x
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        disc_value = Discriminator_model(interpolates, batch_label)
    grad = tape.gradient(disc_value, interpolates)
    grad = tf.reshape(grad, shape=(batch_original_x_shape[0], -1))
    grad_sq_norm = tf.math.square(tf.norm(grad, axis=-1))
    GP_0 = tf.math.scalar_mul(lam, tf.reduce_mean(grad_sq_norm), name='Gradient_penalty_0')
    BCE_loss = loss_discriminator_cross_entropy(Discriminator_model, batch_original_x, batch_gen_x, batch_label)
    GP_0_loss = BCE_loss + GP_0
    return GP_0_loss


def loss_generator_cross_entropy(Generator_model, noise_batch_x, batch_label, Discriminator_model):
    """
    Notes:
    ------
    Note on instability holds here as well.
    """
    batch_gen_x = Generator_model(noise_batch_x, batch_label)
    BCE_loss = - tf.math.reduce_sum(tf.math.log(tf.squeeze(Discriminator_model(batch_gen_x, batch_label))))
    return BCE_loss


def train_discriminator(loss, Discriminator_model, opt, batch_original_x, batch_gen_x, batch_label, **kwargs):
    """
    """
    with tf.GradientTape() as tape:
        loss_disc = loss(Discriminator_model, batch_original_x, batch_gen_x, batch_label, **kwargs)
        gradients = tape.gradient(loss_disc, Discriminator_model.trainable_variables)
        gradient_variables = zip(gradients, Discriminator_model.trainable_variables)
        opt.apply_gradients(gradient_variables)
    return loss_disc


def train_generator(loss, Generator_model, opt, noise_batch_x, batch_label, Discriminator_model, **kwargs):
    """
    """
    with tf.GradientTape() as tape:
        loss_gen = loss(Generator_model, noise_batch_x, batch_label, Discriminator_model, **kwargs)
        gradients = tape.gradient(loss_gen, Generator_model.trainable_variables)
        gradient_variables = zip(gradients, Generator_model.trainable_variables)
        opt.apply_gradients(gradient_variables)
    return loss_gen