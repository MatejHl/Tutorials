import tensorflow as tf


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

