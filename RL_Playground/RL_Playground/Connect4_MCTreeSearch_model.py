import tensorflow as tf


class PolicyHead(tf.keras.Model):
    """
    Policy output layer type.
    """
    def __init__(self,
                 n_actions,
                 filter_num,
                 kernel_size,
                 relu_negative_slope,
                 name = 'PolicyHead'):
        super(PolicyHead, self).__init__(name = name)
        self.n_actions = n_actions
        self.filter_num = filter_num # 2
        self.kernel_size = kernel_size # (1, 1)
        self.relu_negative_slope = relu_negative_slope # 0.0


        self.conv = tf.keras.layers.Conv2D(filters = self.filter_num, 
                                              kernel_size = self.kernel_size, 
                                              strides = (1, 1), 
                                              padding = 'same', 
                                              data_format = 'channels_last',
                                              dilation_rate = (1, 1), 
                                              activation = None, 
                                              use_bias = True,
                                              kernel_initializer = 'glorot_uniform', 
                                              bias_initializer = 'zeros',
                                              kernel_regularizer = None, 
                                              bias_regularizer = None, 
                                              activity_regularizer = None,
                                              kernel_constraint = None, 
                                              bias_constraint = None,
                                              name = 'conv')

        self.batch_normalization = tf.keras.layers.BatchNormalization(axis = -1, 
                                                                      momentum = 0.99, 
                                                                      epsilon = 0.001, 
                                                                      center = True, 
                                                                      scale = True,
                                                                      beta_initializer = 'zeros', 
                                                                      gamma_initializer = 'ones',
                                                                      moving_mean_initializer = 'zeros',  
                                                                      moving_variance_initializer = 'ones',
                                                                      beta_regularizer = None, 
                                                                      gamma_regularizer = None, 
                                                                      beta_constraint = None,
                                                                      gamma_constraint = None, 
                                                                      renorm = False, 
                                                                      renorm_clipping = None, 
                                                                      renorm_momentum = 0.99,
                                                                      fused = None, 
                                                                      trainable = True, 
                                                                      virtual_batch_size = None, 
                                                                      adjustment = None, 
                                                                      name = 'batch_normalization')

        self.relu = tf.keras.layers.ReLU(max_value = None, 
                                            negative_slope = self.relu_negative_slope, 
                                            threshold = 0,
                                            name = 'relu')

        self.flatten = tf.keras.layers.Flatten(data_format = 'channels_last',
                                               name = 'flatten')

        self.out_dense = tf.keras.layers.Dense(units = self.n_actions, 
                                               activation = None, 
                                               use_bias = True, 
                                               kernel_initializer = 'glorot_uniform',
                                               bias_initializer = 'zeros', 
                                               kernel_regularizer = None, 
                                               bias_regularizer = None,
                                               activity_regularizer = None, 
                                               kernel_constraint = None, 
                                               bias_constraint = None,
                                               name = 'out_dense')

    def call(self, input, training = False):
        """
        """
        x = self.conv(input)
        x = self.batch_normalization(x, training = training)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.out_dense(x)
        return x



class ValueHead(tf.keras.Model):
    """
    Value output layer type.
    """
    def __init__(self,
                 filter_num,
                 kernel_size,
                 in_relu_negative_slope,
                 in_dense_units,
                 out_relu_negative_slope,
                 name = 'ValueHead'):
        super(ValueHead, self).__init__(name = name)
        self.filter_num = filter_num # 1
        self.kernel_size = kernel_size # (1, 1)
        self.in_relu_negative_slope = in_relu_negative_slope # 0.0
        self.in_dense_units = in_dense_units # 256
        self.out_relu_negative_slope = out_relu_negative_slope # 0.0


        self.conv = tf.keras.layers.Conv2D(filters = self.filter_num, 
                                           kernel_size = self.kernel_size, 
                                           strides = (1, 1), 
                                           padding = 'same', 
                                           data_format = 'channels_last',
                                           dilation_rate = (1, 1), 
                                           activation = None, 
                                           use_bias = True,
                                           kernel_initializer = 'glorot_uniform', 
                                           bias_initializer = 'zeros',
                                           kernel_regularizer = None, 
                                           bias_regularizer = None, 
                                           activity_regularizer = None,
                                           kernel_constraint = None, 
                                           bias_constraint = None,
                                           name = 'conv')

        self.batch_normalization = tf.keras.layers.BatchNormalization(axis = -1, 
                                                                      momentum = 0.99, 
                                                                      epsilon = 0.001, 
                                                                      center = True, 
                                                                      scale = True,
                                                                      beta_initializer = 'zeros', 
                                                                      gamma_initializer = 'ones',
                                                                      moving_mean_initializer = 'zeros',  
                                                                      moving_variance_initializer = 'ones',
                                                                      beta_regularizer = None, 
                                                                      gamma_regularizer = None, 
                                                                      beta_constraint = None,
                                                                      gamma_constraint = None, 
                                                                      renorm = False, 
                                                                      renorm_clipping = None, 
                                                                      renorm_momentum = 0.99,
                                                                      fused = None, 
                                                                      trainable = True, 
                                                                      virtual_batch_size = None, 
                                                                      adjustment = None, 
                                                                      name = 'batch_normalization')

        self.in_relu = tf.keras.layers.ReLU(max_value = None, 
                                            negative_slope = self.in_relu_negative_slope, 
                                            threshold = 0,
                                            name = 'in_relu')

        self.flatten = tf.keras.layers.Flatten(data_format = 'channels_last',
                                               name = 'flatten')

        self.in_dense = tf.keras.layers.Dense(units = self.in_dense_units, 
                                              activation = None, 
                                              use_bias = True, 
                                              kernel_initializer = 'glorot_uniform',
                                              bias_initializer = 'zeros', 
                                              kernel_regularizer = None, 
                                              bias_regularizer = None,
                                              activity_regularizer = None, 
                                              kernel_constraint = None, 
                                              bias_constraint = None,
                                              name = 'in_dense')

        self.out_relu = tf.keras.layers.ReLU(max_value = None, 
                                            negative_slope = self.out_relu_negative_slope, 
                                            threshold = 0,
                                            name = 'out_relu')

        self.out_dense = tf.keras.layers.Dense(units = 1, 
                                               activation = tf.math.tanh, # tanh activation 
                                               use_bias = True, 
                                               kernel_initializer = 'glorot_uniform',
                                               bias_initializer = 'zeros', 
                                               kernel_regularizer = None, 
                                               bias_regularizer = None,
                                               activity_regularizer = None, 
                                               kernel_constraint = None, 
                                               bias_constraint = None,
                                               name = 'out_dense')
        
    def call(self, input, training = False):
        x = self.conv(input)
        x = self.batch_normalization(x, training = training)
        x = self.in_relu(x)
        x = self.flatten(x)
        x = self.in_dense(x)
        x = self.out_relu(x)
        x = self.out_dense(x)
        return x



class ResidualBlock(tf.keras.Model):
    """
    Feature extraction layer type.
    """
    def __init__(self, 
                 in_filter_num, 
                 in_kernel_size,
                 in_relu_negative_slope,
                 out_filter_num,
                 out_kernel_size,
                 out_relu_negative_slope,
                 name = 'ResidualBlock'):
        super(ResidualBlock, self).__init__(name = name)
        self.in_filter_num = in_filter_num # 256
        self.in_kernel_size = in_kernel_size # (3, 3)
        self.in_relu_negative_slope = in_relu_negative_slope # 0.0
        self.out_filter_num = out_filter_num # 256
        self.out_kernel_size = out_kernel_size # (3, 3)
        self.out_relu_negative_slope = out_relu_negative_slope # 0.0


        self.in_conv = tf.keras.layers.Conv2D(filters = self.in_filter_num, 
                                              kernel_size = self.in_kernel_size, 
                                              strides = (1, 1), 
                                              padding = 'same', 
                                              data_format = 'channels_last',
                                              dilation_rate = (1, 1), 
                                              activation = None, 
                                              use_bias = True,
                                              kernel_initializer = 'glorot_uniform', 
                                              bias_initializer = 'zeros',
                                              kernel_regularizer = None, 
                                              bias_regularizer = None, 
                                              activity_regularizer = None,
                                              kernel_constraint = None, 
                                              bias_constraint = None,
                                              name = 'in_conv')
        
        self.in_batch_normalization = tf.keras.layers.BatchNormalization(axis = -1, 
                                                                         momentum = 0.99, 
                                                                         epsilon = 0.001, 
                                                                         center = True, 
                                                                         scale = True,
                                                                         beta_initializer = 'zeros', 
                                                                         gamma_initializer = 'ones',
                                                                         moving_mean_initializer = 'zeros',  
                                                                         moving_variance_initializer = 'ones',
                                                                         beta_regularizer = None, 
                                                                         gamma_regularizer = None, 
                                                                         beta_constraint = None,
                                                                         gamma_constraint = None, 
                                                                         renorm = False, 
                                                                         renorm_clipping = None, 
                                                                         renorm_momentum = 0.99,
                                                                         fused = None, 
                                                                         trainable = True, 
                                                                         virtual_batch_size = None, 
                                                                         adjustment = None, 
                                                                         name = 'in_batch_normalization')

        self.in_relu = tf.keras.layers.ReLU(max_value = None, 
                                            negative_slope = self.in_relu_negative_slope, 
                                            threshold = 0,
                                            name = 'in_relu')

        self.out_conv = tf.keras.layers.Conv2D(filters = self.out_filter_num, 
                                               kernel_size = self.out_kernel_size, 
                                               strides = (1, 1), 
                                               padding = 'same', 
                                               data_format = 'channels_last',
                                               dilation_rate = (1, 1), 
                                               activation = None, 
                                               use_bias = True,
                                               kernel_initializer = 'glorot_uniform', 
                                               bias_initializer = 'zeros',
                                               kernel_regularizer = None, 
                                               bias_regularizer = None, 
                                               activity_regularizer = None,
                                               kernel_constraint = None, 
                                               bias_constraint = None,
                                               name = 'out_conv')

        self.out_batch_normalization = tf.keras.layers.BatchNormalization(axis = -1, 
                                                                         momentum = 0.99, 
                                                                         epsilon = 0.001, 
                                                                         center = True, 
                                                                         scale = True,
                                                                         beta_initializer = 'zeros', 
                                                                         gamma_initializer = 'ones',
                                                                         moving_mean_initializer = 'zeros',  
                                                                         moving_variance_initializer = 'ones',
                                                                         beta_regularizer = None, 
                                                                         gamma_regularizer = None, 
                                                                         beta_constraint = None,
                                                                         gamma_constraint = None, 
                                                                         renorm = False, 
                                                                         renorm_clipping = None, 
                                                                         renorm_momentum = 0.99,
                                                                         fused = None, 
                                                                         trainable = True, 
                                                                         virtual_batch_size = None, 
                                                                         adjustment = None, 
                                                                         name = 'out_batch_normalization')

        # skip connection
        self.concat = tf.keras.layers.Concatenate(axis=-1,
                                                  name = 'concat')

        self.out_relu = tf.keras.layers.ReLU(max_value = None, 
                                             negative_slope = self.out_relu_negative_slope, 
                                             threshold = 0,
                                             name = 'out_relu')


    def call(self, input, training = False):
        """
        """
        x = self.in_conv(input)
        x = self.in_batch_normalization(x, training = training)
        x = self.in_relu(x)
        x = self.out_conv(x)
        x = self.out_batch_normalization(x, training = training)
        x = self.concat([x, input])
        x = self.out_relu(x)
        return x
        



class Connect4_MCTreeSearch_model(tf.keras.Model):
    """
    """
    def __init__(self,
                 board_shape,
                 filter_num,
                 kernel_size,
                 relu_negative_slope,
                 num_residual_layers,
                 residual_in_filter_num,
                 residual_in_kernel_size,
                 residual_in_relu_negative_slope,
                 residual_out_relu_negative_slope,
                 n_actions,
                 policy_filter_num,
                 policy_kernel_size,
                 policy_relu_negative_slope,
                 value_filter_num,
                 value_kernel_size,
                 value_in_relu_negative_slope,
                 value_in_dense_units,
                 value_out_relu_negative_slope,
                 name = 'Connect4_MCTreeSearch_model'):
        super(Connect4_MCTreeSearch_model, self).__init__(name = name)
        self.board_shape = board_shape
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.relu_negative_slope = relu_negative_slope
        self.num_residual_layers = num_residual_layers
        self.residual_in_filter_num = residual_in_filter_num
        self.residual_in_kernel_size = residual_in_kernel_size
        self.residual_in_relu_negative_slope = residual_in_relu_negative_slope
        self.residual_out_relu_negative_slope = residual_out_relu_negative_slope
        self.n_actions = n_actions
        self.policy_filter_num = policy_filter_num
        self.policy_kernel_size = policy_kernel_size
        self.policy_relu_negative_slope = policy_relu_negative_slope
        self.value_filter_num = value_filter_num
        self.value_kernel_size = value_kernel_size
        self.value_in_relu_negative_slope = value_in_relu_negative_slope
        self.value_in_dense_units = value_in_dense_units
        self.value_out_relu_negative_slope = value_out_relu_negative_slope

        self.conv = tf.keras.layers.Conv2D(input_shape = self.board_shape,
                                           filters = self.filter_num,
                                           kernel_size = self.kernel_size, 
                                           strides = (1, 1), 
                                           padding = 'same', 
                                           data_format = 'channels_last',
                                           dilation_rate = (1, 1), 
                                           activation = None, 
                                           use_bias = True,
                                           kernel_initializer = 'glorot_uniform', 
                                           bias_initializer = 'zeros',
                                           kernel_regularizer = None, 
                                           bias_regularizer = None, 
                                           activity_regularizer = None,
                                           kernel_constraint = None, 
                                           bias_constraint = None,
                                           name = 'conv')

        self.batch_normalization = tf.keras.layers.BatchNormalization(axis = -1, 
                                                                         momentum = 0.99, 
                                                                         epsilon = 0.001, 
                                                                         center = True, 
                                                                         scale = True,
                                                                         beta_initializer = 'zeros', 
                                                                         gamma_initializer = 'ones',
                                                                         moving_mean_initializer = 'zeros',  
                                                                         moving_variance_initializer = 'ones',
                                                                         beta_regularizer = None, 
                                                                         gamma_regularizer = None, 
                                                                         beta_constraint = None,
                                                                         gamma_constraint = None, 
                                                                         renorm = False, 
                                                                         renorm_clipping = None, 
                                                                         renorm_momentum = 0.99,
                                                                         fused = None, 
                                                                         trainable = True, 
                                                                         virtual_batch_size = None, 
                                                                         adjustment = None, 
                                                                         name = 'batch_normalization')

        self.relu = tf.keras.layers.ReLU(max_value = None, 
                                             negative_slope = self.relu_negative_slope, 
                                             threshold = 0,
                                             name = 'relu')

        self.residual_layers = []
        for i in range(self.num_residual_layers):
            self.residual_layers.append(ResidualBlock(in_filter_num = self.residual_in_filter_num, 
                                                      in_kernel_size = self.residual_in_kernel_size,
                                                      in_relu_negative_slope = self.residual_in_relu_negative_slope,
                                                      out_filter_num = self.filter_num, # the same as for input conv
                                                      out_kernel_size = self.kernel_size, # the same as for input conv
                                                      out_relu_negative_slope = self.residual_out_relu_negative_slope,
                                                      name = 'residual_' + str(i)))

        self.policy_head = PolicyHead(n_actions = self.n_actions,
                                      filter_num = self.policy_filter_num,
                                      kernel_size = self.policy_kernel_size,
                                      relu_negative_slope = self.policy_relu_negative_slope,
                                      name = 'policy')

        self.value_head = ValueHead(filter_num = self.value_filter_num,
                                    kernel_size = self.value_kernel_size,
                                    in_relu_negative_slope = self.value_in_relu_negative_slope,
                                    in_dense_units = self.value_in_dense_units,
                                    out_relu_negative_slope = self.value_out_relu_negative_slope,
                                    name = 'value')

    @tf.function
    def call(self, input, training = False):
        """
        """
        x = self.conv(input)
        x = self.batch_normalization(x, training = training)
        x = self.relu(x)
        for residual in self.residual_layers:
            x = residual(x, training = training)
        logits = self.policy_head(x, training = training)
        value = self.value_head(x, training = training)
        return value, logits
        
    # def convertToModelInput(self, states):
    #     """
    #     This is to transform raw output of state to correct shape and type for model.
    #     It is used in Agent class.
    #     """
    #     return tf.cast(tf.stack([tf.reshape(state.trinary, shape = self.board_shape) for state in states]), dtype = tf.float32, name = 'convertToModelInput')
        
