import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

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

@tf.function
def loss_discriminator(Discriminator_model, batch_original_x, batch_gen_x):
    """
    loss for the discrimnator.

    Parameters:
    -----------

    """
    # with tf.name_scope("disc_loss"):
    loss_original = tf.reduce_mean(tf.square(tf.subtract(Discriminator_model(batch_original_x), tf.ones(batch_original_x.shape[0]))))
    loss_gen = tf.reduce_mean(tf.square(tf.subtract(Discriminator_model(batch_gen_x), tf.zeros(batch_gen_x.shape[0]))))
    loss = tf.math.add(loss_original, loss_gen)
    return loss

@tf.function
def train_discriminator(loss, Discriminator_model, opt, batch_original_x, batch_gen_x):
    # with tf.name_scope("disc_train"):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(Discriminator_model, batch_original_x, batch_gen_x), Discriminator_model.trainable_variables)
        gradient_variables = zip(gradients, Discriminator_model.trainable_variables)
        opt.apply_gradients(gradient_variables)

        

class Generator(tf.keras.Model):
    """
    
    """
    def __init__(self, noise_shape, img_shape):
        super(Generator, self).__init__()

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

@tf.function
def loss_generator(Generator_model, noise_batch_x, Discriminator_model):
    """
    Generator loss: E_{x\sim N(0,1)} (g(f(x)) - 1)^2

    Parameters:
    -----------

    """
    # with tf.name_scope("gen_loss"):
    loss = tf.reduce_mean(tf.square(tf.subtract(Discriminator_model(Generator_model(noise_batch_x)), tf.ones(noise_batch_x.shape[0]))))
    return loss

@tf.function
def train_generator(loss, Generator_model, opt, noise_batch_x, Discriminator_model):
  # with tf.name_scope("gen_train"):
  with tf.GradientTape() as tape:
      gradients = tape.gradient(loss(Generator_model, noise_batch_x, Discriminator_model), Generator_model.trainable_variables)
      gradient_variables = zip(gradients, Generator_model.trainable_variables)
      opt.apply_gradients(gradient_variables)