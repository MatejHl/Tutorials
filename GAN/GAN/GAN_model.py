import tensorflow as tf


# ----------------------------------------

class simple_Discriminator(tf.keras.Model):
    """

    """
    def __init__(self):
        super(simple_Discriminator, self).__init__()

        self.input_dense = tf.keras.layers.Dense(input_shape = (1,),
                                             units = 5,
                                              activation = tf.nn.leaky_relu,
                                              use_bias = False)
        
        self.dense_1 = tf.keras.layers.Dense(units = 7,
                                              activation = tf.nn.leaky_relu,
                                              use_bias = True)

        self.dense_2 = tf.keras.layers.Dense(units = 4,
                                              activation = tf.nn.leaky_relu,
                                              use_bias = True)

        self.dense_output = tf.keras.layers.Dense(units = 1,
                                              activation = tf.nn.sigmoid,
                                              use_bias = False)

    @tf.function
    def call(self, input_features):
        """
        """
        # with tf.name_scope('disc_model'):
        d_in = self.input_dense(input_features)
        d_1 = self.dense_1(d_in)
        d_2 = self.dense_2(d_1)
        d_out = self.dense_output(d_2)
        return d_out


class simple_Generator(tf.keras.Model):
    """
    
    """
    def __init__(self, noise_shape):
        super(simple_Generator, self).__init__()
        # input in ~ R^5
        self.dense_input = tf.keras.layers.Dense(input_shape = (noise_shape, ),
                                                units = 7,
                                                activation = tf.nn.leaky_relu,
                                                use_bias = True)

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(momentum = 0.8)
        self.dense_1 = tf.keras.layers.Dense(units = 4,
                                            activation = tf.nn.leaky_relu,
                                            use_bias = True)

        self.dense_output = tf.keras.layers.Dense(units = 1,
                                            activation = tf.nn.leaky_relu,
                                            use_bias = True)
        # Output should be in R

    @tf.function
    def call(self, input_noise):
        """
        """
        # with tf.name_scope('gen_model'):
        d_in = self.dense_input(input_noise)
        b_1 = self.batch_norm_1(d_in)
        d_1 = self.dense_1(b_1)
        d_out = self.dense_output(d_1)
        return d_out


# ----------------------------------------

class Discriminator(tf.keras.Model):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = tf.TensorShape(img_shape)

        self.reshape_to_dense = tf.keras.layers.Reshape(input_shape = img_shape, 
                                                        target_shape = (tf.math.reduce_prod(img_shape),))

        self.dense_1 = tf.keras.layers.Dense(units = 512,
                                              activation = tf.nn.leaky_relu,
                                              use_bias = True)

        self.dense_2 = tf.keras.layers.Dense(units = 256,
                                              activation = tf.nn.leaky_relu,
                                              use_bias = True)

        self.dense_output = tf.keras.layers.Dense(units = 1,
                                              activation = tf.nn.sigmoid,
                                              use_bias = True)

    @tf.function
    def call(self, input_features):
        """
        """
        # with tf.name_scope('disc_model'):
        to_dense = self.reshape_to_dense(input_features)
        d_1 = self.dense_1(to_dense)
        d_2 = self.dense_2(d_1)
        d_out = self.dense_output(d_2)
        return d_out

class Generator(tf.keras.Model):
    """
    
    """
    def __init__(self, noise_shape, img_shape):
        super(Generator, self).__init__()
        self.noise_shape = tf.TensorShape(noise_shape)
        self.img_shape = tf.TensorShape(img_shape)

        self.dense_input = tf.keras.layers.Dense(input_shape = (noise_shape, ),
                                                units = 256,
                                                activation = tf.nn.leaky_relu,
                                                use_bias = True)

        self.batch_norm_1 = tf.keras.layers.BatchNormalization(momentum = 0.8)
        self.dense_1 = tf.keras.layers.Dense(units = 512,
                                            activation = tf.nn.leaky_relu,
                                            use_bias = True)

        self.batch_norm_2 = tf.keras.layers.BatchNormalization(momentum = 0.8)
        self.dense_2 = tf.keras.layers.Dense(units = 1024,
                                            activation = tf.nn.leaky_relu,
                                            use_bias = True)
        
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(momentum = 0.8)
        self.dense_3 = tf.keras.layers.Dense(units = tf.math.reduce_prod(img_shape),
                                            activation = tf.math.tanh,
                                            use_bias = True)

        self.reshape_to_img = tf.keras.layers.Reshape(target_shape = img_shape)

    @tf.function
    def call(self, input_noise):
        """
        """
        # with tf.name_scope('gen_model'):
        d_in = self.dense_input(input_noise)
        b_1 = self.batch_norm_1(d_in)
        d_1 = self.dense_1(b_1)
        b_2 = self.batch_norm_2(d_1)
        d_2 = self.dense_2(b_2)
        b_3 = self.batch_norm_3(d_2)
        d_3 = self.dense_3(b_3)
        to_img = self.reshape_to_img(d_3)
        return to_img



# ---------------------- #
# -------- Loss -------- #
# ---------------------- #
@tf.function
def loss_discriminator_ls(Discriminator_model, batch_original_x, batch_gen_x):
    """
    Leaast squares Disc loss: E_{} (g((f(y))^2 + (g(x)-1)^2)
    x - observed data
    f(y) - generated data
    """
    # with tf.name_scope("disc_loss"):
    loss_original = tf.reduce_mean(tf.square(tf.subtract(Discriminator_model(batch_original_x), tf.ones(batch_original_x.shape[0]))))
    loss_gen = tf.reduce_mean(tf.square(tf.subtract(Discriminator_model(batch_gen_x), tf.zeros(batch_gen_x.shape[0]))))
    LS_loss = tf.math.add(loss_original, loss_gen)
    return LS_loss

@tf.function
def loss_discriminator_cross_entropy(Discriminator_model, batch_original_x, batch_gen_x):
    """
    Binary cross entropy loss for Discriminator.
    """
    loss_original = tf.keras.losses.binary_crossentropy(y_true = tf.ones(batch_original_x.shape[0]),
                                                        y_pred = tf.reshape(Discriminator_model(batch_original_x), shape = (batch_original_x.shape[0],)),
                                                        from_logits=False,
                                                        label_smoothing=0)
    loss_gen = tf.keras.losses.binary_crossentropy(y_true = tf.zeros(batch_gen_x.shape[0]),
                                                   y_pred = tf.reshape(Discriminator_model(batch_gen_x), shape = (batch_gen_x.shape[0],)) ,
                                                   from_logits=False,
                                                   label_smoothing=0)
    BCE_loss = tf.math.add(loss_original, loss_gen, name="disc_loss_cross_entropy")
    return BCE_loss

@tf.function
def loss_discriminator_GP_0_line(Discriminator_model, batch_original_x, batch_gen_x, lam, alpha = None):
    """
    Binary cross-entropy loss with gradient penalty (GP-0) proposed in ref [1].

    Notes:
    ------
    This approach assumes that paths betwwen real datapoints and fake datapoints are line segments. 
    Thus assuming that all line segments from fake to real datapoint are in support.
    C = \{\alpha*x + (1-\alpha)*y| x \in supp(p_g), y \in supp(p_r), \alpha \in (0,1)\} \in supp(p_g) \union supp(p_r)
    -> There is and idea in appendix of the paper how to get other paths 

    Reference:
    ----------
    [1] Improving Generalization and Stability of Generative Adversarial Networks
        https://arxiv.org/abs/1902.03984
    """
    alpha_shape = (batch_original_x.shape[0], ) + tuple([1 for _ in range(len(batch_original_x.shape)-1)])
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
        disc_value = Discriminator_model(interpolates)
    grad = tape.gradient(disc_value, interpolates)
    grad = tf.reshape(grad, shape=(batch_original_x.shape[0], -1))
    grad_sq_norm = tf.math.square(tf.norm(grad, axis=-1))
    GP_0 = tf.math.scalar_mul(lam, tf.reduce_mean(grad_sq_norm), name='Gradient_penalty_0')
    BCE_loss = loss_discriminator_cross_entropy(Discriminator_model, batch_original_x, batch_gen_x)
    GP_0_loss = BCE_loss + GP_0
    return GP_0_loss

@tf.function
def loss_generator_ls(Generator_model, noise_batch_x, Discriminator_model):
    """
    Leaast squares Generator loss: E_{x\sim N(0,1)} (g(f(x)) - 1)^2
    """
    # with tf.name_scope("gen_loss"):
    LS_loss = tf.reduce_mean(tf.square(tf.subtract(Discriminator_model(Generator_model(noise_batch_x)), tf.ones(noise_batch_x.shape[0]))))
    return LS_loss

@tf.function
def loss_generator_cross_entropy(Generator_model, noise_batch_x, Discriminator_model):
    """
    Binary cross entropy loss for Generator: E_{x\sim N(0,1)} binary_cross_entropy(g(f(x)), 1)
    """
    BCE_loss = tf.keras.losses.binary_crossentropy(y_true = tf.ones(noise_batch_x.shape[0]),
                                               y_pred = tf.reshape(Discriminator_model(Generator_model(noise_batch_x)), shape = (noise_batch_x.shape[0],)),
                                               from_logits=False,
                                               label_smoothing=0)
    return BCE_loss



# ----------------------- #
# -------- Train -------- #
# ----------------------- #
def train_discriminator(loss, Discriminator_model, opt, batch_original_x, batch_gen_x, **kwargs):
    # with tf.name_scope("disc_train"):
    with tf.GradientTape() as tape:
        loss_disc = loss(Discriminator_model, batch_original_x, batch_gen_x, **kwargs)
        gradients = tape.gradient(loss_disc, Discriminator_model.trainable_variables)
        gradient_variables = zip(gradients, Discriminator_model.trainable_variables)
        opt.apply_gradients(gradient_variables)
    return loss_disc

def train_generator(loss, Generator_model, opt, noise_batch_x, Discriminator_model, **kwargs):
    # with tf.name_scope("gen_train"):
    with tf.GradientTape() as tape:
        loss_gen = loss(Generator_model, noise_batch_x, Discriminator_model, **kwargs)
        gradients = tape.gradient(loss_gen, Generator_model.trainable_variables)
        gradient_variables = zip(gradients, Generator_model.trainable_variables)
        opt.apply_gradients(gradient_variables)
    return loss_gen